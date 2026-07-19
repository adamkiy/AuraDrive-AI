#!/usr/bin/env bash
#
# AuraDrive — launcher
# ====================
# Starts the system and, if needed, installs the Python dependencies and makes
# sure a local Ollama server and model are ready. This is the launcher for every
# platform: macOS and Linux directly, Windows under Git Bash or WSL. The one
# thing that differs between them is where a virtualenv keeps its interpreter,
# and that is detected below rather than assumed.
#
# Layout this script assumes:
#   phase_B/run.sh             this file
#   phase_B/AuraDrive.command  a macOS wrapper so Finder has something to open
#   phase_B/src/               the eleven modules and requirements.txt
#   phase_B/logs/              every JSONL log the run produces
#   phase_B/.venv/             created by --venv
#
# Usage:
#   ./run.sh                 Run. Installs deps only if missing; ensures Ollama.
#   ./run.sh --venv          Use/create an isolated .venv (recommended on a
#                            fresh machine, e.g. the Jetson). Installs into it.
#   ./run.sh --install       Force a (re)install of requirements before running.
#   ./run.sh --setup         Do setup (deps + model pull) and EXIT without running.
#   ./run.sh --skip-ollama   Do not manage Ollama (agent calls fall back to the
#                            deterministic cold layer if Ollama is unavailable).
#   ./run.sh --help          Show this help.
#
# Environment overrides (all optional):
#   AURADRIVE_MODEL           primary model      (default llama3.2:1b)
#   AURADRIVE_FALLBACK_MODEL  fallback model     (default llama3.2:latest)
#   AURADRIVE_OLLAMA_URL      Ollama endpoint    (default http://localhost:11434)
#   AURADRIVE_CAMERA_INDEX    camera index       (default 0)
#   AURADRIVE_PYTHON          python interpreter (default: first python3 >= 3.10)
#
# Notes:
#   * Requires Python 3.10+ (the perception layer uses `X | None` signatures)
#     and Ollama 0.5+ (the agent sends a JSON Schema, which needs structured
#     outputs). The full dependency set is documented in src/requirements.txt.
#   * The script only stops an Ollama server that IT started; an already-running
#     server (the macOS app, or a systemd service) is left untouched.
#   * Audio: macOS uses afplay and say, Windows uses winsound and PowerShell
#     System.Speech, all of them standard on their platform. On Linux no backend
#     is resolved, so the system runs in full but silently and says so on stderr.
#   * On a Jetson, MediaPipe/OpenCV may need JetPack-specific builds; if the pip
#     install of mediapipe fails there, install the vendor wheel and re-run with
#     --skip-ollama-style manual setup. On Apple Silicon the pip wheels work.
#
set -euo pipefail

# ---------- defaults (override via environment) ----------
MODEL="${AURADRIVE_MODEL:-llama3.2:1b}"
FALLBACK_MODEL="${AURADRIVE_FALLBACK_MODEL:-llama3.2:latest}"
OLLAMA_URL="${AURADRIVE_OLLAMA_URL:-http://localhost:11434}"
CAMERA_INDEX="${AURADRIVE_CAMERA_INDEX:-0}"
MIN_PY_MINOR=10            # require Python 3.10+

USE_VENV=0
FORCE_INSTALL=0
SETUP_ONLY=0
MANAGE_OLLAMA=1

# ---------- helpers ----------
log() { printf '[run] %s\n' "$*"; }
err() { printf '[run][ERROR] %s\n' "$*" >&2; }

show_help() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; }

# ---------- arg parsing ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --venv)         USE_VENV=1 ;;
    --install)      FORCE_INSTALL=1 ;;
    --setup)        SETUP_ONLY=1; FORCE_INSTALL=1 ;;
    --skip-ollama)  MANAGE_OLLAMA=0 ;;
    -h|--help)      show_help; exit 0 ;;
    *)              err "Unknown option: $1 (try --help)"; exit 2 ;;
  esac
  shift
done

# ---------- resolve the layout ----------
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
LOGS="$ROOT/logs"
[ -f "$SRC/main.py" ]         || { err "src/main.py not found under $ROOT."; exit 1; }
[ -f "$SRC/requirements.txt" ] || { err "src/requirements.txt not found."; exit 1; }
mkdir -p "$LOGS"

# ---------- pick a Python interpreter (>= 3.10) ----------
pick_python() {
  local cand
  for cand in "${AURADRIVE_PYTHON:-}" python3 python; do
    [ -n "$cand" ] || continue
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, $MIN_PY_MINOR) else 1)" 2>/dev/null; then
      echo "$cand"; return 0
    fi
  done
  return 1
}
BASE_PY="$(pick_python)" || {
  err "Need Python 3.${MIN_PY_MINOR}+ on PATH (set AURADRIVE_PYTHON to point at one)."
  exit 1
}
log "Using interpreter: $("$BASE_PY" -c 'import sys,shutil;print(shutil.which(sys.executable) or sys.executable, sys.version.split()[0])')"

# ---------- virtual environment ----------
# An existing .venv is used automatically. Anything else would be a trap: the
# environment sits right here, already provisioned, and ignoring it because a
# flag was omitted would send pip at the system interpreter instead. --venv
# therefore means "create one if it is missing", not "use the one that exists".
# Setting AURADRIVE_PYTHON explicitly overrides this and wins.
venv_python() {   # bin/ on Unix, Scripts/ on Windows; look rather than assume
  if [ -x "$1/bin/python" ]; then
    echo "$1/bin/python"
  elif [ -x "$1/Scripts/python.exe" ]; then
    echo "$1/Scripts/python.exe"
  fi
}

if [ -n "${AURADRIVE_PYTHON:-}" ]; then
  PY="$BASE_PY"
  log "Using the interpreter from AURADRIVE_PYTHON (ignoring any .venv)."
elif [ -n "$(venv_python "$ROOT/.venv")" ]; then
  PY="$(venv_python "$ROOT/.venv")"
  log "Using existing virtualenv $ROOT/.venv"
elif [ "$USE_VENV" -eq 1 ]; then
  log "Creating virtualenv .venv ..."
  "$BASE_PY" -m venv "$ROOT/.venv"
  FORCE_INSTALL=1
  PY="$(venv_python "$ROOT/.venv")"
  [ -n "$PY" ] || { err "Virtualenv created but no interpreter found inside it."; exit 1; }
  log "Using virtualenv $ROOT/.venv"
else
  PY="$BASE_PY"
  log "No .venv present; using the system interpreter. Run with --venv for an isolated one."
fi

# ---------- Python dependencies ----------
# Functional check: MediaPipe is notorious for "installed but broken" states
# where `import mediapipe` succeeds yet `mp.solutions` is missing. A plain
# presence check (find_spec) is NOT enough — we actually exercise the package.
deps_ok() {
  "$PY" - <<'PYEOF'
import sys
try:
    import numpy            # noqa: F401
    import cv2              # noqa: F401
    import mediapipe as mp
    _ = mp.solutions.face_mesh   # the line that breaks on a partial install
except Exception as exc:
    sys.stderr.write(f"[deps] functional import check failed: {type(exc).__name__}: {exc}\n")
    sys.exit(1)
sys.exit(0)
PYEOF
}

_pip_install() {  # $@ -> extra pip flags
  if ! "$PY" -m pip install "$@" -r "$SRC/requirements.txt"; then
    log "pip failed; retrying with --break-system-packages (PEP 668) ..."
    "$PY" -m pip install --break-system-packages "$@" -r "$SRC/requirements.txt" || true
  fi
}

if [ "$FORCE_INSTALL" -eq 1 ] || ! deps_ok; then
  log "Installing Python requirements ..."
  "$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
  _pip_install
  if ! deps_ok; then
    log "Still not functional (a broken in-place package is likely) — forcing a clean reinstall ..."
    _pip_install --force-reinstall --no-cache-dir
  fi
  if ! deps_ok; then
    err "MediaPipe/OpenCV are installed but NOT functional in this interpreter ('mp.solutions' missing)."
    err "Most reliable fix: run in a clean virtualenv  ->  ./run.sh --venv"
    err "If that still fails, this Python is likely arch-mismatched; use a Homebrew Python:"
    err "  brew install python@3.11 && AURADRIVE_PYTHON=\$(brew --prefix)/bin/python3.11 ./run.sh --venv"
    exit 1
  fi
  log "Dependencies functional."
else
  log "Python dependencies already present and functional."
fi

# ---------- Ollama lifecycle ----------
OLLAMA_STARTED=0
OLLAMA_PID=""
cleanup() {
  if [ "$OLLAMA_STARTED" -eq 1 ] && [ -n "$OLLAMA_PID" ]; then
    log "Stopping the Ollama server this script started (pid $OLLAMA_PID) ..."
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

ollama_up()      { ollama list >/dev/null 2>&1; }
model_present()  { ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -Fxq "$1"; }

# Offer to install Ollama, never do it silently. Installing system software
# needs administrator rights and would otherwise happen at the worst possible
# moment, seconds before a demonstration, behind a sudo prompt nobody expected.
# So: name the exact command for this platform, and run it only if the operator
# says yes at an interactive terminal. Anywhere non-interactive this just prints
# the command and returns failure, leaving the decision to a human.
install_ollama() {
  local cmd=""
  case "$(uname -s)" in
    Darwin)
      command -v brew >/dev/null 2>&1 && cmd="brew install ollama"
      ;;
    Linux)
      cmd="curl -fsSL https://ollama.com/install.sh | sh"
      ;;
    MINGW*|MSYS*|CYGWIN*)
      command -v winget >/dev/null 2>&1 && cmd="winget install --id Ollama.Ollama -e"
      ;;
  esac

  if [ -z "$cmd" ]; then
    err "Install it from https://ollama.com/download, then re-run."
    return 1
  fi

  err "It can be installed with:  $cmd"
  if [ ! -t 0 ]; then
    err "Not an interactive terminal, so nothing was installed. Run that command, then re-run."
    return 1
  fi

  printf '[run] Run that command now? This may ask for your password. [y/N] '
  read -r reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) err "Skipped. Install it yourself, then re-run."; return 1 ;;
  esac

  log "Installing Ollama ..."
  if ! sh -c "$cmd"; then
    err "The install did not succeed. Use https://ollama.com/download instead."
    return 1
  fi
  command -v ollama >/dev/null 2>&1 || {
    err "Ollama installed but is not on PATH yet. Open a new terminal and re-run."
    return 1
  }
  log "Ollama installed."
  return 0
}

if [ "$MANAGE_OLLAMA" -eq 1 ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    err "Ollama is not installed — the LLM agent layer needs it."
    install_ollama || {
      err "(Or re-run with --skip-ollama to exercise the deterministic cold layer only.)"
      exit 1
    }
  fi

  if ollama_up; then
    log "Ollama server already running."
  else
    log "Starting 'ollama serve' in the background ..."
    ollama serve >/tmp/auradrive_ollama.log 2>&1 &
    OLLAMA_PID=$!
    OLLAMA_STARTED=1
    for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
    ollama_up || { err "Ollama did not become ready (see /tmp/auradrive_ollama.log)."; exit 1; }
    log "Ollama server is up."
  fi

  # Structured outputs, which the agent depends on, arrived in Ollama 0.5. On
  # anything older the JSON Schema in the request is ignored, every reply fails
  # validation and the agent falls back to the cold layer on every frame. That
  # is invisible from the outside, so say it loudly here rather than let the
  # reasoning layer quietly contribute nothing.
  ver="$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)"
  if [ -n "$ver" ]; then
    major="${ver%%.*}"; rest="${ver#*.}"; minor="${rest%%.*}"
    if [ "$major" -eq 0 ] && [ "$minor" -lt 5 ] 2>/dev/null; then
      err "Ollama $ver is too old: the agent needs structured outputs, added in 0.5."
      err "The system will run, but every model reply will fail validation and the"
      err "reasoning layer will contribute nothing. Update from https://ollama.com/download"
    else
      log "Ollama version $ver."
    fi
  fi

  if model_present "$MODEL" || model_present "$FALLBACK_MODEL"; then
    log "Model available ($MODEL or $FALLBACK_MODEL)."
  else
    log "No model found; pulling '$MODEL' (one-time download, ~1.3 GB) ..."
    ollama pull "$MODEL"
  fi
else
  log "Skipping Ollama management (--skip-ollama)."
  log "If Ollama is down, every agent call falls back to the cold layer (this is safe, just no LLM reasoning)."
fi

# ---------- export knobs so main.py AND the agent subprocess inherit them ----------
export AURADRIVE_MODEL="$MODEL"
export AURADRIVE_FALLBACK_MODEL="$FALLBACK_MODEL"
export AURADRIVE_OLLAMA_URL="$OLLAMA_URL"
export AURADRIVE_CAMERA_INDEX="$CAMERA_INDEX"
# The agent runs as a subprocess with its own cwd, so point its log at logs/
# explicitly; the other three follow the working directory set below.
export AURADRIVE_AGENT_LOG="$LOGS/agent_decision_log.jsonl"

# ---------- setup-only mode ----------
if [ "$SETUP_ONLY" -eq 1 ]; then
  log "Setup complete (--setup). Not launching."
  exit 0
fi

# ---------- run ----------
log "Launching AuraDrive. Press 'q' in the video window to quit."
log "Logs for this session: $LOGS"
cd "$LOGS"                     # all four JSONL logs land here
set +e
"$PY" "$SRC/main.py"
rc=$?
set -e
log "AuraDrive exited (code $rc)."
exit $rc
