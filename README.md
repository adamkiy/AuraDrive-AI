# AuraDrive — Edge-AI Driver Fatigue Detection

On-device driver drowsiness detection combining a deterministic
AHP-weighted signal-fusion engine with a local LLM reasoning agent.
Nothing leaves the device: perception, the decision engine and the
language model all run on the in-vehicle unit.

Braude College of Engineering — Final Project 2026
Team 26-1-D-10 · Adam Kayal & Maor Tzur · Supervisor: Mr. Ilya Zeldner

## Demo video

[Watch the demo](https://drive.google.com/file/d/1baDizSanrvPVadC55TprDTa3OGW5xCEn/view?usp=sharing)

## Repository structure

```
phase_A/                 Phase A submission (initial prototype)
phase_B/                 Phase B, the current system
├── AuraDrive.command    double-click launcher for macOS
├── run.sh               the launcher, for macOS, Linux and Windows
├── src/                 the eleven modules and requirements.txt
└── logs/sample/         recorded JSONL logs backing the measured results
```

## Quick start

Before anything else, install [Python 3.10 or newer](https://www.python.org/downloads/)
and [Ollama](https://ollama.com/download), and make sure a camera is connected.
Everything else is automatic.

**macOS** — open the `phase_B` folder and double-click **AuraDrive.command**.

**Windows 10/11** — open Git Bash in the `phase_B` folder and run `./run.sh`.
Git Bash comes with [Git for Windows](https://git-scm.com/download/win); it is
what lets one launcher serve every platform. WSL works too.

**Linux** — `./run.sh` from the `phase_B` folder.

On a machine that has never run AuraDrive the launcher performs the first-time
setup by itself: it creates an isolated Python environment, installs the
dependencies, starts a local Ollama server if one is not already running, and
downloads the reasoning model. That download is roughly 1.3 GB and happens only
once. Later launches skip all of it and start in seconds.

Press `q` in the video window to end a session.

There is one launcher rather than one per platform. `run.sh` resolves its own
location, picks a suitable interpreter, and detects whether a virtualenv keeps
its interpreter in `bin/` or in `Scripts/`, which is the difference between a
Unix and a Windows environment. `AuraDrive.command` is a thin macOS wrapper
around it so Finder has something to open, not a second implementation.

### Two things macOS will ask for

**The camera.** The first launch triggers a camera permission prompt for
Terminal. If it is dismissed, the system starts and then fails to open the
camera. Grant it under System Settings → Privacy & Security → Camera, then
launch again.

**Gatekeeper.** A copy of this repository that arrived as a downloaded ZIP is
quarantined, and double-clicking may be refused with a warning about an
unidentified developer. Right-click the file and choose **Open** instead, which
offers the option to run it anyway. A repository obtained with `git clone` is
not quarantined and does not have this problem.

### If double-clicking does nothing

On **macOS**, the executable bit does not survive a ZIP download. One command
restores it:

```bash
chmod +x ~/Desktop/AuraDrive-repo/phase_B/AuraDrive.command
```

The shortcut repairs `run.sh` the same way on its own, so this only ever needs
doing for the `.command` file itself.

### Windows notes

Run `./run.sh` from Git Bash rather than from `cmd.exe` or PowerShell, neither
of which can execute a shell script. If `permission denied` appears, run
`chmod +x run.sh` once.

Alert tones use `winsound` and speech uses PowerShell with System.Speech. Both
ship with the operating system, so there is nothing extra to install for audio.

The camera is claimed exclusively while a session runs, so close Teams, Zoom or
the Camera app first; otherwise OpenCV fails to open the device.

## Moving or renaming the project folder

The launchers resolve their own location at startup, so the repository can sit
anywhere and can be renamed freely. Both `AuraDrive.command` and `run.sh` work
from any path, and neither depends on the folder being called `AuraDrive-repo`
or living on the Desktop.

One rule makes that true: **keep the launchers next to `src/`.** Moving
`AuraDrive.command` on its own, for example onto the Desktop or into the Dock,
breaks it, because it looks for `run.sh` beside itself. To launch from
somewhere convenient, make an alias rather than a copy: right-click the file,
choose Make Alias, and move the alias wherever you like. An alias keeps
pointing at the original.

Note that an IDE is a separate matter. PyCharm stores an absolute interpreter
path in its own settings, so moving the folder will break the configured
interpreter there even though the launchers still work. Re-select the
interpreter at `phase_B/.venv/bin/python` if that happens, or
`phase_B/.venv/Scripts/python.exe` on Windows.

## Running from the command line

The shortcut is a wrapper around `run.sh`, so anything it does can be done
directly.

```bash
cd phase_B
./run.sh --venv      # first time: isolated environment + one-time model pull
./run.sh             # every run after that
```

The same two commands work on macOS, Linux, and Windows under Git Bash or WSL.
The flags are:

| Flag | Effect |
|---|---|
| `--venv` | Create and use an isolated `.venv` |
| `--setup` | Prepare dependencies and model, then exit without launching |
| `--install` | Force a dependency reinstall |
| `--skip-ollama` | Run the deterministic layer alone, without the model |
| `--help` | Show the full usage text |

Requires Python 3.10 or newer, a driver-facing camera, and
[Ollama](https://ollama.com/download).

An existing `.venv` beside the launcher is used automatically, so `--venv` only
ever matters on a machine that does not have one yet. Setting
`AURADRIVE_PYTHON` overrides the choice of interpreter entirely.

### Checking a machine before a demonstration

`--setup` prepares everything and exits without opening the camera, which is
the safe way to confirm a new machine is ready:

```bash
./run.sh --setup
```

It installs the dependencies, verifies that MediaPipe is genuinely functional
rather than merely importable, and pulls the model if it is absent. Running it
in advance means the 1.3 GB download cannot surprise anyone mid-demonstration.

## How it works

Three layers of decreasing trust decide what the driver hears:

1. **Reflex** — a continuous eye closure of two seconds or more latches an
   EMERGENCY straight from Python, bypassing every other layer.
2. **Deterministic cold engine** — an AHP-weighted fusion of PERCLOS, closure
   duration, blink rate and yawns, with bounded context multipliers for time
   of day and trip length. This is the safety baseline.
3. **LLM refinement** — a local Llama 3.2 agent reads the behavioural
   trajectory and may adjust the baseline, but only through an arbiter that
   accepts any escalation and permits de-escalation of at most one tier, and
   only when the baseline is already mild.

A missed detection is a crash and a false alarm is an inconvenience, so the
language model is never allowed to weaken a high-risk deterministic verdict.

Measured on the development laptop: perception sustains 30.3 FPS while a
single agent inference takes a median of 11.7 seconds, and the camera loop is
never blocked. The raw records are in `phase_B/logs/sample/`.

## Where the logs go

Each session writes four JSONL files to `phase_B/logs/`: every frame's metrics,
each model call and its result, the frame-by-frame agent rendezvous, and the
audit trail of decisions actually shown to the driver. They are ignored by git,
since a long session produces a large sensor log. A trimmed sample of each is
committed under `logs/sample/` as the evidence behind the measured figures
above.

## Configuration

Behaviour is tuned through environment variables, so no code edit is needed:

| Variable | Default | Effect |
|---|---|---|
| `AURADRIVE_CAMERA_INDEX` | `0` | Selects the capture device |
| `AURADRIVE_MODEL` | `llama3.2:1b` | Primary Ollama model |
| `AURADRIVE_OLLAMA_URL` | `http://localhost:11434` | Local model endpoint |
| `AURADRIVE_SPEAK_FULL` | `0` | Speak the agent's advisory instead of the fixed phrase |
| `AURADRIVE_AUDIO` / `_TTS` | `1` / `1` | Enable alert tones / spoken messages |

The full reference is in Appendix B of the project book.

## Status

This is a research and demonstration prototype. It does not control the
vehicle, and it is neither a certified automotive safety product nor a
clinical device. It is meant to be used alongside an attentive driver,
never in place of one.
