# AuraDrive v8.1 Validation Harness

Copy the four files in this archive into the root folder of **AuraDrive v8.1 No-Reference**:

- `run_validation_suite.py`
- `run_full_validation.bat`
- `TESTING_GUIDE.md`
- `MANUAL_VALIDATION_CHECKLIST.md`

The harness does not modify runtime decision logic. It adds offline, integration, optional live-agent, and optional camera validation.

## Commands

```bat
python run_validation_suite.py
python run_validation_suite.py --live-agent
python run_validation_suite.py --camera --camera-seconds 10
python run_validation_suite.py --live-agent --camera --camera-seconds 10
```

Each run creates `validation_reports/*.json` and `validation_reports/*.md`.

Run camera checks only while stationary.
