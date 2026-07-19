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
phase_A/              Phase A submission (initial prototype)
phase_B/              Phase B, the current system
├── run.sh            launcher for macOS and Linux
├── run.bat           launcher for Windows 10/11
├── src/              the eleven modules and requirements.txt
└── logs/sample/      recorded JSONL logs backing the measured results
```

## Running it

One launcher does everything: it installs the Python dependencies, makes
sure a local Ollama server and model are ready, then starts the monitor.

**macOS and Linux**

```bash
cd phase_B
./run.sh --venv      # first time: isolated environment + one-time model pull
./run.sh             # every run after that
```

**Windows 10/11**

```bat
cd phase_B
run.bat --venv
run.bat
```

Both accept `--setup` (prepare only), `--install` (force a dependency
reinstall), `--skip-ollama` (run the deterministic layer alone) and
`--help`. Press `q` in the video window to stop a session.

Requires Python 3.10 or newer, a driver-facing camera, and
[Ollama](https://ollama.com/download). The model is pulled automatically on
first run, a one-time download of roughly 1.3 GB.

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
