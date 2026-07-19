# controller.py
import json
import subprocess
import cv2
from sensor import EyeBlinkSensor

AGENT_SCRIPT = "llm_agent_stream.py"  # call llm_agent_stream.py


def start_agent():
    # -u => unbuffered stdout (חשוב ל-stream)
    return subprocess.Popen(
        ["python", "-u", AGENT_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # line-buffered
    )


def sanitize_payload(payload: dict) -> dict:
    """
    Ensure payload contains exactly the fields the LLM agent expects.
    Your agent expects keys like:
    Driver_State, EAR, Eyes Closed Duration, Blinks/min, PERCLOS
    """
    return {
        "Driver_State": payload.get("Driver_State", payload.get("driver_status", payload.get("state", "Unknown"))),
        "EAR": payload.get("EAR", payload.get("ear")),
        "Eyes Closed Duration": payload.get("Eyes Closed Duration", payload.get("eye_closure_duration", payload.get("closed_ms", 0))),
        "Blinks/min": payload.get("Blinks/min", payload.get("blink_rate", payload.get("blinks_per_min", 0))),
        "PERCLOS": payload.get("PERCLOS", payload.get("perclos", 0.0)),
    }


def send_to_agent(proc: subprocess.Popen, payload: dict) -> dict | None:
    if proc.stdin is None or proc.stdout is None:
        return None

    clean = sanitize_payload(payload)

    # send NDJSON line
    proc.stdin.write(json.dumps(clean) + "\n")
    proc.stdin.flush()

    # read one NDJSON decision line
    line = proc.stdout.readline()
    if not line:
        return None

    line = line.strip()
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # אם ה-agent הדפיס משהו שהוא לא JSON
        return {"status": "error", "raw": line}


def main():
    agent = start_agent()

    sensor = EyeBlinkSensor(debug=True)
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            sensor.process_frame(frame)

            payload = getattr(sensor, "last_risk_payload", None)
            if payload:
                decision = send_to_agent(agent, payload)

                if decision:
                    print("AGENT DECISION:", decision)

                # חשוב: לא לשלוח שוב את אותו אירוע
                sensor.last_risk_payload = None

            cv2.imshow("AuraDrive", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # אם יש שגיאות, נדפיס stderr לפני סגירה
        try:
            if agent.stderr:
                err = agent.stderr.read().strip()
                if err:
                    print("AGENT STDERR:\n", err)
        except Exception:
            pass

        try:
            agent.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()