#!/usr/bin/env python3
import sys
import json
import time
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, Tuple, List
from collections import deque

###################
# CONSTANTS
###################

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:latest"
REQUEST_TIMEOUT_SEC = 15
MAX_LLM_RESPONSE_CHARS = 20000
MAX_LLM_RETRIES = 1
MAX_LLM_TOKENS = 256

CONVERSATION_LOG_FILE = "conversation_log.jsonl"
CONVERSATION_HISTORY_SECONDS = 5

REQUIRED_INPUT_FIELDS = [
    "Driver_State",
    "EAR",
    "Eyes Closed Duration",
    "Blinks/min",
    "PERCLOS",
]

VALID_DRIVER_STATES = {"EYES_OPEN", "EYES_CLOSED"}

ALLOWED_OUTPUT_STATUS = {"ok", "error", "needs_clarification"}

#############################################################
# system output commands.
# STATUS_CHECK is used by the CODE ONLY (protocol/fallback),
#  NOT by the LLM in order to promise vaild input and output.
#############################################################
WRAPPER_ALLOWED_COMMANDS = {
    "STATUS_CHECK",
    "NO_ACTION",
    "GENTLE_ALERT",
    "MODERATE_ALERT",
    "URGENT_ALERT",
    "EMERGENCY_ALERT",
}

##################################
# LLM-level commands (model output).
##################################
LLM_ALLOWED_COMMANDS = {
    "NO_ACTION",
    "GENTLE_ALERT",
    "MODERATE_ALERT",
    "URGENT_ALERT",
    "EMERGENCY_ALERT",
}

ALLOWED_SEVERITY = {"LOW", "MEDIUM", "HIGH"}
MAX_MESSAGE_CHARS = 200
PROTOCOL_SEVERITY = "LOW"

# ############################################
# SEVERITY POLICY (STRICT ENFORCEMENT FOR LLM)
# ############################################

LLM_COMMAND_TO_SEVERITY = {
    "NO_ACTION": "LOW",
    "GENTLE_ALERT": "LOW",
    "MODERATE_ALERT": "MEDIUM",
    "URGENT_ALERT": "HIGH",
    "EMERGENCY_ALERT": "HIGH",
}

# ###################
# SYSTEM PROMPT (LLM) 
# ###################

SYSTEM_PROMPT = """You are a deterministic rule-execution engine for driver fatigue decisions.
You MUST follow the rules EXACTLY as written.
No creativity, no medical interpretation, no probabilistic reasoning.

IMPORTANT:
- Input validity, schema, and missing values are handled OUTSIDE the model.
- You MUST assume all required input fields exist and are valid.
- Ignore any extra keys such as "case", "expected", "meta".
- Your ONLY responsibility is to apply the decision rules below.
- Output ONLY a single JSON object and NOTHING ELSE.

════════════════════════════════════════════════════════
STRICT OUTPUT FORMAT (MANDATORY)
════════════════════════════════════════════════════════
Return exactly ONE JSON object with EXACTLY these top-level keys:

{
  "command": "NO_ACTION" | "GENTLE_ALERT" | "MODERATE_ALERT" | "URGENT_ALERT" | "EMERGENCY_ALERT",
  "sleep": <integer 0-60000>,
  "severity": "LOW" | "MEDIUM" | "HIGH",
  "args": {
    "reason": <string>,
    "message": <string max 200 chars>
  }
}

Hard constraints:
- No extra keys (NO "output", NO "status", NO "notes", NO arrays).
- args must contain BOTH "reason" and "message".
- args.message length ≤ 200 characters.
- sleep must be integer 0..60000.

════════════════════════════════════════════════════════
FIXED MAPPINGS (NO DEVIATION)
════════════════════════════════════════════════════════

Severity mapping:
- NO_ACTION        → LOW
- GENTLE_ALERT     → LOW
- MODERATE_ALERT   → MEDIUM
- URGENT_ALERT     → HIGH
- EMERGENCY_ALERT  → HIGH

Sleep mapping:
- NO_ACTION        → 10000
- GENTLE_ALERT     → 8000
- MODERATE_ALERT   → 5000
- URGENT_ALERT     → 0
- EMERGENCY_ALERT  → 0

════════════════════════════════════════════════════════
INPUT FIELDS (GUARANTEED VALID)
════════════════════════════════════════════════════════
You will receive a JSON object containing:
- Driver_State: "EYES_OPEN" or "EYES_CLOSED"
- EAR: number in [0,1]
- Eyes Closed Duration: number ≥ 0 (milliseconds)
- Blinks/min: number ≥ 0
- PERCLOS: number in [0,1]

════════════════════════════════════════════════════════
ABSOLUTE EDGE RULES (CRITICAL)
════════════════════════════════════════════════════════
1) STRICT INEQUALITIES ONLY
- ">" means strictly greater (not equal).
- "<" means strictly less (not equal).
Examples:
- Duration > 500 is FALSE when Duration == 500
- EAR < 0.24 is FALSE when EAR == 0.24
- PERCLOS > 0.15 is FALSE when PERCLOS == 0.15
- PERCLOS > 0.30 is FALSE when PERCLOS == 0.30

2) NO EXTRA LOGIC
- Do NOT infer fatigue beyond the rules.
- Do NOT reinterpret thresholds.
- Do NOT blend conditions.

════════════════════════════════════════════════════════
PRIORITY ENFORCEMENT (ABSOLUTE, NON-NEGOTIABLE)
════════════════════════════════════════════════════════
You MUST execute the decision as a strict IF / ELSE-IF chain.
The FIRST TRUE condition ALWAYS wins.

Define boolean flags EXACTLY:

S1 = (Driver_State == "EYES_CLOSED") AND (Eyes Closed Duration > 2000)

S2 = (Driver_State == "EYES_CLOSED") AND (Eyes Closed Duration > 500)

S3 = (PERCLOS > 0.30) AND (EAR < 0.20)

S4 = (PERCLOS > 0.30) OR ((EAR < 0.20) AND (Eyes Closed Duration > 200))

S5 = (Blinks/min < 8) OR (Blinks/min > 50)

S6 = (PERCLOS > 0.15) OR (EAR < 0.24) OR (Blinks/min < 10)

Decision chain (MUST FOLLOW EXACTLY):

IF S1 is TRUE:
  choose STEP 1
ELSE IF S2 is TRUE:
  choose STEP 2
ELSE IF S3 is TRUE:
  choose STEP 3
ELSE IF S4 is TRUE:
  choose STEP 4
ELSE IF S5 is TRUE:
  choose STEP 5
ELSE IF S6 is TRUE:
  choose STEP 6
ELSE:
  choose STEP 7

CRITICAL RULES:
- You are FORBIDDEN from selecting a later step if an earlier step is TRUE.
- You are FORBIDDEN from averaging or blending steps.
- If more than one Sx is TRUE, the earliest one MUST be chosen.
- If your chosen step is not the earliest TRUE step, your answer is WRONG.

════════════════════════════════════════════════════════
STEP OUTPUTS (FIXED, EXACT)
════════════════════════════════════════════════════════

STEP 1:
command = EMERGENCY_ALERT
severity = HIGH
sleep = 0
args.reason = "microsleep_critical"
args.message = "EMERGENCY: Eyes closed >2000ms. Pull over safely immediately."

STEP 2:
command = URGENT_ALERT
severity = HIGH
sleep = 0
args.reason = "eyes_closed_prolonged"
args.message = "URGENT: Eyes closed >500ms. Take action now (pull over safely)."

STEP 3:
command = URGENT_ALERT
severity = HIGH
sleep = 0
args.reason = "combined_high_risk"
args.message = "URGENT: High-risk fatigue indicators (PERCLOS>0.30 & EAR<0.20)."

STEP 4:
command = MODERATE_ALERT
severity = MEDIUM
sleep = 5000
args.reason = "moderate_confirmed"
args.message = "MODERATE: Fatigue risk detected. Stay alert and plan a short break."

STEP 5:
command = MODERATE_ALERT
severity = MEDIUM
sleep = 5000
args.reason = "blink_anomaly"
args.message = "MODERATE: Abnormal blink rate detected. Monitor and consider a break."

STEP 6:
command = GENTLE_ALERT
severity = LOW
sleep = 8000
args.reason = "early_warning"
args.message = "GENTLE: Early fatigue signs detected. Refocus and stay alert."

STEP 7:
command = NO_ACTION
severity = LOW
sleep = 10000
args.reason = "normal"
args.message = "Normal: No fatigue indicators detected."

════════════════════════════════════════════════════════
PRIORITY FEW-SHOT (CRITICAL EDGE CASES)
════════════════════════════════════════════════════════

Example 1 (S3 MUST override S4):
Input:
{"Driver_State":"EYES_OPEN","EAR":0.19,"Eyes Closed Duration":300,"Blinks/min":18,"PERCLOS":0.40}
Output:
{"command":"URGENT_ALERT","sleep":0,"severity":"HIGH","args":{"reason":"combined_high_risk","message":"URGENT: High-risk fatigue indicators (PERCLOS>0.30 & EAR<0.20)."}}

Example 2 (S2 MUST override S3/S4):
Input:
{"Driver_State":"EYES_CLOSED","EAR":0.19,"Eyes Closed Duration":800,"Blinks/min":18,"PERCLOS":0.80}
Output:
{"command":"URGENT_ALERT","sleep":0,"severity":"HIGH","args":{"reason":"eyes_closed_prolonged","message":"URGENT: Eyes closed >500ms. Take action now (pull over safely)."}}

════════════════════════════════════════════════════════
Now process the next input JSON and output exactly one JSON object.
════════════════════════════════════════════════════════
"""

# ##############################################################
# CONVERSATION LOG (LAST [CONVERSATION_HISTORY_SECONDS] SECONDS)
# ##############################################################


class ConversationLog:
    def __init__(self, history_seconds: int):
        self.history_seconds = history_seconds
        self.entries: deque = deque()
        self._lock = threading.Lock()

    def add_entry(self, entry: Dict[str, Any]) -> None:
        entry_timestamp = time.time()
        entry["log_timestamp"] = entry_timestamp
        entry["log_timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(entry_timestamp))

        with self._lock:
            self.entries.append(entry)
            self._cleanup_old_entries()
            self._write_log_file()

    def _cleanup_old_entries(self) -> None:
        cutoff_time = time.time() - self.history_seconds
        while self.entries and self.entries[0]["log_timestamp"] < cutoff_time:
            self.entries.popleft()

    def _write_log_file(self) -> None:
        try:
            with open(CONVERSATION_LOG_FILE, "w", encoding="utf-8") as f:
                for e in self.entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except Exception as ex:
            sys.stderr.write(f"[LOG ERROR] {ex}\n")
            sys.stderr.flush()

conversation_log = ConversationLog(CONVERSATION_HISTORY_SECONDS)

# ###################
# PROTOCOL VALIDATION
# ####################

def is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")

def validate_protocol_input(input_object: Any) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(input_object, dict):
        return "error", {"reason": "invalid_input_type", "detail": "Input must be a JSON object"}

    missing_fields = [f for f in REQUIRED_INPUT_FIELDS if f not in input_object]
    if missing_fields:
        return "error", {"reason": "missing_fields", "missing_fields": missing_fields}

    missing_values = [f for f in REQUIRED_INPUT_FIELDS if is_missing_value(input_object.get(f))]
    if missing_values:
        return "needs_clarification", {"reason": "missing_values", "missing_values": missing_values}

    driver_state = input_object.get("Driver_State")
    if not isinstance(driver_state, str):
        return "error", {"reason": "invalid_type", "field": "Driver_State", "detail": "Must be a string"}
    if driver_state not in VALID_DRIVER_STATES:
        return "error", {"reason": "invalid_value", "field": "Driver_State", "detail": f"Must be one of {sorted(VALID_DRIVER_STATES)}"}

    constraints = {
        "EAR": (0.0, 1.0),
        "PERCLOS": (0.0, 1.0),
        "Eyes Closed Duration": (0.0, None),
        "Blinks/min": (0.0, None),
    }

    invalid_fields: List[Dict[str, Any]] = []
    for name, (min_v, max_v) in constraints.items():
        v = input_object.get(name)
        if not isinstance(v, (int, float)):
            invalid_fields.append({"field": name, "problem": "invalid_type", "detail": "Must be numeric"})
            continue
        fv = float(v)
        if fv < min_v:
            invalid_fields.append({"field": name, "problem": "out_of_range", "detail": f"Must be >= {min_v}"})
            continue
        if max_v is not None and fv > max_v:
            invalid_fields.append({"field": name, "problem": "out_of_range", "detail": f"Must be <= {max_v}"})

    if invalid_fields:
        return "error", {"reason": "invalid_fields", "invalid_fields": invalid_fields}

    return "ok", {"reason": "input_ok"}

# ################
# OUTPUT BUILDERS
# ################

def _truncate_message(msg: Any) -> str:
    s = msg if isinstance(msg, str) else str(msg)
    return s[:MAX_MESSAGE_CHARS]

def build_final_output(status: str, severity: str, command: str, sleep_duration_ms: int, args: Dict[str, Any]) -> Dict[str, Any]:
    safe_args = dict(args) if isinstance(args, dict) else {"reason": "invalid_args"}
    if "message" not in safe_args:
        safe_args["message"] = "N/A"
    safe_args["message"] = _truncate_message(safe_args["message"])

    return {
        "status": status,
        "severity": severity,
        "sleep": int(sleep_duration_ms),
        "output": {
            "command": command,
            "args": safe_args
        }
    }

def protocol_error_response(details: Dict[str, Any]) -> Dict[str, Any]:
    args = {"reason": details.get("reason", "protocol_error"), **details, "message": "Input does not conform to required schema or value ranges"}
    return build_final_output("error", PROTOCOL_SEVERITY, "STATUS_CHECK", 0, args)

def protocol_needs_clarification_response(details: Dict[str, Any]) -> Dict[str, Any]:
    args = {"reason": details.get("reason", "missing_values"), **details, "message": "Input contains missing values and requires clarification"}
    return build_final_output("needs_clarification", PROTOCOL_SEVERITY, "STATUS_CHECK", 0, args)

def technical_error_response(reason: str, detail: str) -> Dict[str, Any]:
    return build_final_output("error", PROTOCOL_SEVERITY, "STATUS_CHECK", 0, {"reason": reason, "detail": detail, "message": "Technical error occurred"})

def invalid_llm_output_fallback(detail: str) -> Dict[str, Any]:
    return build_final_output(
        "error",
        "HIGH",
        "STATUS_CHECK",
        0,
        {"reason": "invalid_llm_output", "detail": detail, "message": "Invalid LLM output"}
    )

# ########################
# LLM DECISION VALIDATION
# ########################

def validate_llm_decision(decision_object: Any) -> Tuple[bool, str]:
    if not isinstance(decision_object, dict):
        return False, "Decision must be a JSON object"

    allowed_top_keys = {"command", "sleep", "severity", "args"}
    if set(decision_object.keys()) != allowed_top_keys:
        return False, f"Decision must contain exactly keys: {sorted(allowed_top_keys)}"

    command = decision_object.get("command")
    if command not in LLM_ALLOWED_COMMANDS:
        return False, f"Invalid LLM command: {command}"

    severity = decision_object.get("severity")
    if severity not in ALLOWED_SEVERITY:
        return False, f"Invalid severity: {severity}"

    sleep_duration_ms = decision_object.get("sleep")
    if not isinstance(sleep_duration_ms, int) or not (0 <= sleep_duration_ms <= 60000):
        return False, "sleep must be integer between 0 and 60000"

    args = decision_object.get("args")
    if not isinstance(args, dict):
        return False, "args must be a JSON object"

    if "reason" not in args:
        return False, "args.reason is required"
    if "message" not in args:
        return False, "args.message is required"

    msg = args.get("message")
    if not isinstance(msg, str) or len(msg) > MAX_MESSAGE_CHARS:
        return False, f"args.message must be string <= {MAX_MESSAGE_CHARS} chars"

    expected_severity = LLM_COMMAND_TO_SEVERITY.get(command)
    if severity != expected_severity:
        return False, f"severity must match policy: {command} -> {expected_severity}"

    return True, ""

# ###########
# LLM CALL
# ###########

def call_llm_for_decision(valid_input_object: Dict[str, Any], attempt: int = 0) -> str:
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(valid_input_object, ensure_ascii=False)},
        ],
        "options": {
            "temperature": 0,
            "num_predict": MAX_LLM_TOKENS,
            "repeat_penalty": 1.1,
        },
    }

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            raw = resp.read(MAX_LLM_RESPONSE_CHARS).decode("utf-8", errors="replace")

        parsed = json.loads(raw)
        content = parsed.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Missing decision content")
        return content

    except urllib.error.URLError as ex:
        if attempt < MAX_LLM_RETRIES:
            return call_llm_for_decision(valid_input_object, attempt + 1)
        raise ex

# ###########
# PIPELINE
# ###########

def process_input_line(raw_line: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    try:
        input_object = json.loads(raw_line)
    except json.JSONDecodeError as ex:
        return technical_error_response("invalid_json", str(ex)), None, None, "protocol_error"

    status, details = validate_protocol_input(input_object)

    if status == "needs_clarification":
        return protocol_needs_clarification_response(details), input_object, None, "protocol_needs_clarification"

    if status == "error":
        return protocol_error_response(details), input_object, None, "protocol_error"

    llm_raw: Optional[str] = None
    try:
        llm_raw = call_llm_for_decision(input_object)
        decision_object = json.loads(llm_raw)

        ok, err = validate_llm_decision(decision_object)
        if not ok:
            return invalid_llm_output_fallback(err), input_object, llm_raw, "llm_protocol_violation"

        final_output = build_final_output(
            "ok",
            decision_object["severity"],
            decision_object["command"],
            decision_object["sleep"],
            decision_object["args"],
        )
        return final_output, input_object, llm_raw, None

    except Exception as ex:
        return technical_error_response("llm_error", str(ex)), input_object, llm_raw, "technical_error"

# #############
# MAIN FUNCTION
# #############

def main() -> None:
    sys.stderr.write("[AGENT] Started (Option A + strict LLM keys + protocol severity const)\n")
    sys.stderr.flush()

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue

        start = time.time()
        output, input_obj, llm_raw, code = process_input_line(raw)
        latency_ms = int((time.time() - start) * 1000)

        print(json.dumps(output, ensure_ascii=False), flush=True)

        conversation_log.add_entry({
            "input": input_obj if input_obj is not None else {"raw": raw},
            "llm_response": llm_raw,
            "output": output,
            "latency_ms": latency_ms,
            "decision_source": code or "llm_autonomous"
        })

if __name__ == "__main__":
    main()
