# Manual Validation Checklist — AuraDrive

Perform these checks only in a stationary environment. Do not operate or test the system while driving.

## A. Startup and visual overlay

- [ ] Run `python main.py` and confirm the camera window opens.
- [ ] Confirm face contour/landmark overlay is visible.
- [ ] Confirm EAR, PERCLOS, blink rate, mouth state, pitch and roll appear in the debug overlay.
- [ ] Confirm the window exits cleanly with `Q`.
- [ ] Confirm no repeated Python exceptions appear in the terminal.

## B. Face and lighting resilience

- [ ] Face centered, normal indoor lighting: `no_face=false` for most frames.
- [ ] Temporarily move out of frame: overlay reports no face without crashing.
- [ ] Return to frame: tracking recovers without restarting the program.
- [ ] Test brighter and dimmer stationary lighting; record any tracking loss.

## C. Eye and mouth behavior

- [ ] Normal blinking does not create Emergency.
- [ ] A short stationary eye closure does not create Emergency.
- [ ] Speak normally: mouth movement is not recorded as repeated yawning.
- [ ] A sustained open-mouth/yawn-like movement is reflected in the mouth state and System Log.

## D. Head-pose calibration

- [ ] Hold a neutral posture during initial calibration.
- [ ] Confirm `Head_Pose_Calibrated=true` after calibration.
- [ ] Move the head upward: it must not be marked as head-down.
- [ ] Move downward for more than the configured persistence interval: it may be marked as head-down.
- [ ] Tilt left/right: roll delta changes without jumps near ±180 degrees.

## E. Safety and agent behavior

- [ ] Review `final_decision_log.jsonl`: a continuous reflex event produces one `T1_reflex` event, not one per frame.
- [ ] Review `agent_decision_log.jsonl`: the input includes System Log, Facts Text and Frame Narrative.
- [ ] Review `agent_rendezvous_log.jsonl`: each agent result matches the requested `frame_id`.
- [ ] Verify an agent timeout results in `cold_fallback`, not a crash.
- [ ] Verify no `reference_engine` field or source is present in logs or project files.

## F. Evidence to retain for the report

- [ ] Latest automatic JSON/Markdown report from `validation_reports/`.
- [ ] A short console capture from `python run_validation_suite.py`.
- [ ] One successful `python run_validation_suite.py --live-agent` report.
- [ ] One stationary camera run and the resulting sensor/final/agent logs.
- [ ] A table of observed latency, timeout count, fallback count and duplicate-reflex count.
