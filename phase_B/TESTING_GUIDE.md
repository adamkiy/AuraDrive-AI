# AuraDrive Validation Suite

## Purpose

`run_validation_suite.py` tests the defined software behavior of AuraDrive v8.1 without changing the runtime system. It creates JSON and Markdown reports under `validation_reports/` so every failure is visible and traceable.

A passing report means the tested contracts and synthetic scenarios behaved as expected. It does **not** certify the system for real-road use, medical diagnosis, or automotive safety compliance.

## Commands

```bat
python run_validation_suite.py
python run_validation_suite.py --live-agent
python run_validation_suite.py --camera --camera-seconds 10
python run_validation_suite.py --live-agent --camera --camera-seconds 10
```

The default command is offline. It does not call a camera or Ollama.

## Automatic coverage

- Decision schema and command ordering.
- Cold decision thresholds, including no false Emergency from weighted signals.
- Reflex latch: one emergency per continuous eye-closure event.
- Directional pitch and circular roll-angle arithmetic.
- Text-only LLM input: System Log, Facts Text and Frame Narrative.
- LLM output schema, Post-Microsleep guard and agent Emergency cap.
- Cold-versus-Agent arbitration: floor, acceptance and escalation cap.
- Bounded temporal recovery.
- No-Reference architecture contract and preserved face-mesh overlay source.
- `latest-only` queue semantics.
- SensorDB, exact `frame_id` rendezvous and audit logging.
- T1 reflex, T2 reflex-skip, T3 context refresh, T4 agent logging, T5 successful arbitration and T5 timeout fallback.

## Optional live tests

`--live-agent` verifies only that the local Ollama service and selected model can return a valid agent response. It reports a failure if the model cannot respond before the configured live-test timeout.

`--camera` opens the selected camera, samples its sensor output, checks the required keys and confirms that a face is detected at least once. Run this only while the vehicle is stationary.

## Manual visual checklist

Use the checklist in `MANUAL_VALIDATION_CHECKLIST.md` after the automated suite passes. It covers visual overlays, camera framing, lighting and behaviors that cannot be fully reproduced by synthetic metrics.
