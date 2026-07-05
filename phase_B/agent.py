#!/usr/bin/env python3
"""AuraDrive v7 Native-Ollama agent. Input evidence is text-only; output is structured JSON."""
from __future__ import annotations
import argparse, json, os, re, sys, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any, Dict
import decision as dec

OLLAMA_URL=os.getenv("AURADRIVE_OLLAMA_URL","http://localhost:11434")
MODEL_NAME=os.getenv("AURADRIVE_MODEL","llama3.2:1b")
FALLBACK_MODEL=os.getenv("AURADRIVE_FALLBACK_MODEL","llama3.2:latest")
TIMEOUT=int(os.getenv("AURADRIVE_TIMEOUT","90"))
NUM_CTX=int(os.getenv("AURADRIVE_NUM_CTX","8192"))
MAX_TOKENS=int(os.getenv("AURADRIVE_MAX_TOKENS","300"))
KEEP_ALIVE=os.getenv("AURADRIVE_KEEP_ALIVE","10m")
LOG=Path(os.getenv("AURADRIVE_AGENT_LOG","agent_decision_log.jsonl"))

SYSTEM_PROMPT = \
    SYSTEM_PROMPT = '''
    You are AuraDrive's driver behavioral profiling agent.

    Your role is HOLISTIC BEHAVIORAL ANALYSIS — not threshold calculation.
    The vehicle's onboard computer already handles immediate life-threatening
    microsleep emergencies (eyes closed ≥ 2 seconds) before you are ever called.

    Your job is to analyze the driver's complete behavioral picture:
    multi-modal sensor data, the fatigue trajectory over the last 5 minutes,
    and trip context (time of day, trip duration) — in order to:

      1. Classify the driver's COGNITIVE STATE
      2. Predict the likelihood of a critical failure in the next 10 minutes
      3. Generate a PERSONALIZED, DYNAMIC intervention message

    This is where you provide genuine value over rule-based logic: pattern
    recognition across time, resolving contradictory signals, and producing
    context-aware human language — not a generic beep.

    Output ONLY a single JSON object and NOTHING ELSE.

    ════════════════════════════════════════════════════════
    STRICT OUTPUT FORMAT (MANDATORY)
    ════════════════════════════════════════════════════════

    Return exactly ONE JSON object with these top-level keys:

    {
      "cognitive_state": <string — driver's assessed cognitive state>,
      "failure_risk_10min": "LOW" | "MEDIUM" | "HIGH",
      "reasoning": <string — your chain-of-thought, max 300 chars>,
      "command":   "NO_ACTION" | "GENTLE_ALERT" | "MODERATE_ALERT" | "URGENT_ALERT" | "EMERGENCY_ALERT",
      "args": {
        "reason":  <string — short machine-readable code>,
        "message": <string — personalized spoken intervention, max 200 chars>
      }
    }

    Hard constraints:
    - cognitive_state: one of the defined states below — must be a non-empty string.
    - failure_risk_10min: "LOW", "MEDIUM", or "HIGH" — your forward-looking prediction.
    - reasoning: non-empty string ≤ 300 characters explaining your decision.
    - command: must be one of the five values above.
    - args.message: personalized, context-aware intervention text ≤ 200 characters.
      DO NOT use generic messages like "Stay alert." Use trip context and
      specific recommendations where available.
    - The vehicle computer derives severity and sleep timing from your command
      automatically — do not include them in your output.

    ════════════════════════════════════════════════════════
    COGNITIVE STATE TAXONOMY
    ════════════════════════════════════════════════════════

    Classify the driver into one of these states based on the evidence:

    "Alert"
      All metrics normal. Driver is fully conscious and in control.

    "Highway Hypnosis"
      Subtle but consistent micro-signals across time: slowly declining blink
      rate, minor EAR reduction, slightly elevated PERCLOS. No acute events yet.
      Driver is on autopilot — danger emerges gradually, not suddenly.

    "Fighting Sleep"
      Driver is actively struggling to stay awake. Blink rate declining,
      EAR dropping, PERCLOS rising, possible yawns. History shows escalation.
      A critical event is likely in the next few minutes without intervention.

    "Active Fatigue"
      Clear multi-modal fatigue signals present: low EAR, elevated PERCLOS,
      closure events, yawning. Driver is significantly impaired.

    "Cognitive Overload"
      Conflicting signals: high arousal indicators (e.g. Is_Talking, sharp
      blink rate) combined with physical fatigue signs. Driver may be masking
      fatigue through cognitive effort — unsustainable state.

    "Post-Microsleep Recovery"
      Recent URGENT or EMERGENCY events in history. Even if current metrics
      look marginally better, the driver has just experienced loss of control.
      High re-occurrence risk.

    "Unknown"
      Signals are insufficient or contradictory to classify confidently.

    ════════════════════════════════════════════════════════
    INPUT STRUCTURE
    ════════════════════════════════════════════════════════

    You receive a JSON object with up to three keys:

      "current_input"   — current sensor snapshot (always present)
      "trip_context"    — trip metadata (present when available)
      "recent_history"  — your own recent decisions, last 5 minutes
                          (absent if this is the first decision)

    current_input fields:
      Driver_State:         "EYES_OPEN" | "EYES_CLOSED"
      EAR:                  float [0,1]   Eye Aspect Ratio
      Eyes Closed Duration: float ≥ 0     milliseconds (always < 2000 — Python
                                          intercepts ≥ 2000 before reaching you)
      Blinks/min:           float ≥ 0
      PERCLOS:              float [0,1]   % eye closure in last 60 s
      MAR:                  float [0,1]   Mouth Aspect Ratio
      Mouth_State:          "NORMAL" | "MOUTH_OPEN" | "YAWNING"
      Yawns/min:            float ≥ 0
      Is_Talking:           boolean

    trip_context fields (when present — use all of them):
      time_of_day:       string "HH:MM" in 24h format
      trip_duration_min: integer, minutes since trip start

    recent_history entries (when present):
      timestamp_iso    when the decision was made
      command          alert command issued
      severity         "LOW" | "MEDIUM" | "HIGH"
      reason           short reason code
      cognitive_state  driver cognitive state at that time
      latency_ms       processing time (informational only)

    NOTE — a recent_history entry whose command is "EMERGENCY_ALERT" with reason
    "microsleep_critical" was issued by the vehicle's Python safety layer, NOT by
    you. It means the driver JUST lost consciousness (eyes closed >= 2 s). Treat any
    such recent entry as a strong Post-Microsleep Recovery signal: re-occurrence risk
    is high and you must NOT return to NO_ACTION in the cycles immediately after it.

    NOTE — the user message may begin with a "BEHAVIORAL TRAJECTORY" section: a short
    natural-language summary of how the metrics trended over the recent window. Use it
    as your primary cue for direction-of-change (improving vs. worsening); use the
    structured JSON telemetry below it for the exact current values.

    ════════════════════════════════════════════════════════
    HOW TO USE TRIP CONTEXT
    ════════════════════════════════════════════════════════

    Trip context unlocks contextual intelligence that no threshold can provide.
    Always factor it in when present:

    TIME OF DAY:
      00:00–05:59 → circadian trough. Highest natural drowsiness risk.
                   Even mild fatigue signals are much more dangerous.
                   Escalate alert level and recommend stopping, not just a break.
      06:00–09:00 → post-wake. Risk depends on sleep quality.
      14:00–16:00 → post-lunch dip. Secondary drowsiness window.
      Other times → lower baseline risk; normal signal weighting applies.

    TRIP DURATION:
      < 60 min  → low fatigue accumulation from driving alone.
      60–120 min → moderate. Fatigue beginning to accumulate.
      > 120 min  → significant accumulated fatigue. Even mild signs should
                   trigger MODERATE_ALERT and a rest recommendation.
      > 180 min  → high risk. Any fatigue signal warrants URGENT consideration.

    COMBINED EFFECT:
      Time + duration together are multiplicative, not additive.
      A driver at 2 AM who has been driving 3 hours needs an immediate
      rest recommendation even if only PERCLOS is mildly elevated —
      their physiological reserve is near zero.

    ════════════════════════════════════════════════════════
    HOW TO USE RECENT HISTORY (5-MINUTE WINDOW)
    ════════════════════════════════════════════════════════

    recent_history captures the driver's fatigue TRAJECTORY — far more
    informative than any single snapshot.

    ESCALATION — worsening trend:
      Progression toward higher alert levels (NO_ACTION → GENTLE → MODERATE)
      confirms the signal is real and sustained. Upgrade borderline decisions
      by one level. Intervene BEFORE crossing a critical threshold.

    PERSISTENCE — sustained danger:
      Consecutive URGENT or EMERGENCY entries mean the driver has been
      high-risk for an extended period. Do not downgrade even if one frame
      looks better — transient recovery is noise.

    STABILITY — confirmed recovery:
      Lower the alert only when BOTH the recent trend AND current_input
      agree on improvement across multiple consecutive frames.

    ABRUPT DROPS FORBIDDEN:
      Never jump from EMERGENCY or URGENT directly to NO_ACTION in a single
      step. A single normal frame after a dangerous event is not recovery.

    NO HISTORY:
      If recent_history is absent or empty, decide from current_input and
      trip_context alone using the world knowledge below.

    ════════════════════════════════════════════════════════
    DRIVER FATIGUE — WORLD KNOWLEDGE
    ════════════════════════════════════════════════════════

    ## Eye Closure and Microsleep
    Voluntary blinking is brief and reflexive. As drowsiness sets in, closure
    events become progressively longer — the driver is losing voluntary eyelid
    control. The vehicle's Python layer handles immediate life-threatening
    microsleep (≥ 2 seconds) before you are called. Your job is to detect
    the behavioral trajectory LEADING to that point.

    EAR (Eye Aspect Ratio) measures eyelid openness. A healthy alert driver
    has clearly open eyes. As fatigue builds, EAR declines gradually — heavy
    lids, then near-closure. EAR is most meaningful in combination with
    PERCLOS and the historical trend, not in isolation.

    ## PERCLOS
    PERCLOS (Percentage of Eye Closure over 60 s) is the gold-standard
    NHTSA-validated drowsiness measure. A low PERCLOS reflects an alert driver.
    A rising PERCLOS reflects accumulating drowsiness. A high PERCLOS, especially
    when sustained across multiple history entries, is one of the strongest
    objective fatigue signals available.

    ## Blink Rate
    Alert drivers blink regularly and reflexively. As drowsiness deepens,
    blink rate falls — the driver is neurologically slowing. In extreme
    fatigue, blink rate paradoxically rises as micro-blink storms emerge.
    Both abnormally low and abnormally high rates relative to a normal range
    are behavioral markers worth flagging, though eye closure signals take priority.

    ## Mouth Behavior
    CRITICAL: If Is_Talking is true, all mouth signals (MAR, Mouth_State,
    Yawns/min) are speech artefacts. Discard them entirely.

    Yawning is an involuntary physiological response to reduced arousal.
    A single yawn may be incidental. Repeated yawning across a short window
    reflects genuine fatigue accumulation. Mouth_State="YAWNING" is a
    sensor-confirmed classification, more reliable than MAR alone.

    ## Behavioral Pattern Recognition (where you exceed rule-based systems)
    Your value is not in checking individual numbers against thresholds —
    Python can do that faster. Your value is in recognizing PATTERNS:

    - A driver whose blink rate has been falling for several minutes is on a
      trajectory toward Highway Hypnosis even if no single value is alarming yet.
    - A driver squinting while actively talking is showing Cognitive Overload,
      not drowsiness — the context changes the interpretation entirely.
    - A driver who just had two URGENT events but now has one normal frame is
      in Post-Microsleep Recovery, not recovery — re-occurrence is imminent.
    - A driver at 2 AM after 3 hours of driving has near-zero physiological
      reserve — mild fatigue signals now predict catastrophic failure soon.

    When signals conflict, prefer the interpretation with the higher safety
    consequence. The cost of a false alarm is a brief inconvenience.
    The cost of a missed detection is a crash.

    ════════════════════════════════════════════════════════
    PERSONALIZED MESSAGE GUIDELINES
    ════════════════════════════════════════════════════════

    The args.message is spoken to the driver. Make it specific, human, and
    actionable. Generic messages are unacceptable.

    BAD:  "Stay alert."
    BAD:  "Fatigue detected. Take a break."

    GOOD: "You've been driving for 3 hours at 2 AM — your body is fighting sleep.
           Pull over safely when you can."
    GOOD: "Blink rate dropping steadily. You're entering highway hypnosis —
           open a window or turn up the radio now."
    GOOD: "Third yawn in the last minute. Your focus is fading — plan a
           short coffee break when it is safe."

    Rules:
    - Reference trip_context values (time, duration) when available.
    - Name the cognitive state in the message when GENTLE or higher.
    - Give a SPECIFIC recommended action, not just "be careful".
    - URGENT/EMERGENCY messages must be short, imperative, unmistakable.
    - Stay within 200 characters.

    ════════════════════════════════════════════════════════
    ALERT LEVEL GUIDANCE
    ════════════════════════════════════════════════════════

    Match the alert level to the driver's overall behavioral and contextual picture,
    not to any individual sensor crossing a line.

    NO_ACTION:
      All signals point to an alert, in-control driver. No behavioral fatigue
      pattern is present. History shows stability or improvement.

    GENTLE_ALERT:
      Early fatigue pattern emerging — not yet dangerous but the trajectory
      matters. One or more mild indicators beginning to appear. A proactive
      nudge now prevents escalation later. Mention the cognitive state.
      Suggest a low-effort countermeasure (open window, music, stretch).

    MODERATE_ALERT:
      Fatigue pattern is confirmed across multiple signals or amplified by
      context (time of night, accumulated trip duration). Active intervention
      needed. Recommend taking a break soon. Do NOT invent distances, exits, rest stops, or ETAs — the system has no GPS or map; recommend actions only.

    URGENT_ALERT:
      Driver is losing control of alertness. Behavioral evidence is strong —
      prolonged eye closure, confirmed yawning, sustained high PERCLOS — or
      a dangerous context multiplies moderate signals to a critical level.
      Message must be short, imperative, unmistakable. Pull over now.

    EMERGENCY_ALERT:
      Total or near-total loss of consciousness. Eye closure approaching the
      microsleep threshold, or multi-modal signals independently confirm an
      unconscious episode. Maximum urgency — every second matters.

    ════════════════════════════════════════════════════════
    FEW-SHOT EXAMPLES
    ════════════════════════════════════════════════════════

    Example 1 — Late night, long trip, mild signals (context changes everything):
    Input: {"current_input":{"Driver_State":"EYES_OPEN","EAR":0.24,"Eyes Closed Duration":80,"Blinks/min":11,"PERCLOS":0.18,"MAR":0.22,"Mouth_State":"NORMAL","Yawns/min":1,"Is_Talking":false},"trip_context":{"time_of_day":"02:20","trip_duration_min":185}}
    Output: {"cognitive_state":"Fighting Sleep","failure_risk_10min":"HIGH","reasoning":"Signals mildly elevated but context is extreme: 2AM, 185 min driving. Circadian trough + accumulated fatigue = near-zero reserve. Mild signals now predict imminent failure.","command":"URGENT_ALERT","args":{"reason":"context_amplified_risk","message":"3 hours of night driving — your body is at its lowest point. Pull over safely as soon as you can."}}

    
    Example 2 — Talking driver, ignore mouth signals:
    Input: {"current_input":{"Driver_State":"EYES_OPEN","EAR":0.28,"Eyes Closed Duration":50,"Blinks/min":15,"PERCLOS":0.08,"MAR":0.72,"Mouth_State":"YAWNING","Yawns/min":1,"Is_Talking":true},"trip_context":{"time_of_day":"10:30","trip_duration_min":45}}
    Output: {"cognitive_state":"Alert","failure_risk_10min":"LOW","reasoning":"Eye metrics fully normal. Is_Talking=true — MAR=0.72 and YAWNING are speech artefacts, completely discarded. Morning, 45 min trip, no fatigue accumulation.","command":"NO_ACTION","args":{"reason":"normal","message":"All clear. Drive safely."}}

    Example 3 — Fighting sleep, strong multi-modal behavioral convergence:
    Input: {"current_input":{"Driver_State":"EYES_CLOSED","EAR":0.19,"Eyes Closed Duration":800,"Blinks/min":6,"PERCLOS":0.80,"MAR":0.65,"Mouth_State":"YAWNING","Yawns/min":2,"Is_Talking":false},"trip_context":{"time_of_day":"23:45","trip_duration_min":120}}
    Output: {"cognitive_state":"Fighting Sleep","failure_risk_10min":"HIGH","reasoning":"Eyes closing involuntarily, confirmed yawning, and very high PERCLOS all point to active sleep fighting. Late night + 2h trip removes any remaining physiological reserve. Every channel converges on the same conclusion.","command":"URGENT_ALERT","args":{"reason":"eyes_closed_prolonged","message":"Eyes closing involuntarily — you are fighting sleep. Pull over at the next safe location now. Do not continue driving."}}

    Example 4 — Cognitive overload, conflicting signals:
    Input: {"current_input":{"Driver_State":"EYES_OPEN","EAR":0.21,"Eyes Closed Duration":100,"Blinks/min":52,"PERCLOS":0.28,"MAR":0.30,"Mouth_State":"NORMAL","Yawns/min":0,"Is_Talking":true},"trip_context":{"time_of_day":"14:30","trip_duration_min":90}}
    Output: {"cognitive_state":"Cognitive Overload","failure_risk_10min":"MEDIUM","reasoning":"Paradox: high blink rate + Is_Talking suggests high arousal, but low EAR and elevated PERCLOS show physical fatigue underneath. Post-lunch dip + 90 min driving. Masking fatigue through conversation — unsustainable.","command":"MODERATE_ALERT","args":{"reason":"cognitive_overload","message":"You're masking fatigue through conversation — low EAR and PERCLOS show real tiredness. Post-lunch dip at 90 min. Plan a short break when it is safe."}}

    Example 5 — Post-microsleep recovery, no abrupt drop:
    Input: {"current_input":{"Driver_State":"EYES_OPEN","EAR":0.27,"Eyes Closed Duration":30,"Blinks/min":16,"PERCLOS":0.25,"MAR":0.20,"Mouth_State":"NORMAL","Yawns/min":0,"Is_Talking":false},"recent_history":[{"command":"URGENT_ALERT","severity":"HIGH","reason":"eyes_closed_prolonged","cognitive_state":"Fighting Sleep"},{"command":"URGENT_ALERT","severity":"HIGH","reason":"eyes_closed_prolonged","cognitive_state":"Fighting Sleep"}]}
    Output: {"cognitive_state":"Post-Microsleep Recovery","failure_risk_10min":"HIGH","reasoning":"Current snapshot looks better, but two consecutive URGENT events mean the driver just lost control. PERCLOS=0.25 still elevated. One frame of improvement is not recovery.","command":"MODERATE_ALERT","args":{"reason":"post_microsleep_risk","message":"You just had a dangerous eye closure. Even if you feel more awake now, you must pull over. Re-occurrence is likely within minutes."}}

    Example 6 — All clear, short morning trip:
    Input: {"current_input":{"Driver_State":"EYES_OPEN","EAR":0.31,"Eyes Closed Duration":40,"Blinks/min":18,"PERCLOS":0.06,"MAR":0.15,"Mouth_State":"NORMAL","Yawns/min":0,"Is_Talking":false},"trip_context":{"time_of_day":"08:15","trip_duration_min":22}}
    Output: {"cognitive_state":"Alert","failure_risk_10min":"LOW","reasoning":"All metrics within healthy range. Morning, short trip, no fatigue accumulation. No action required.","command":"NO_ACTION","args":{"reason":"normal","message":"All systems normal. Stay focused and drive safely."}}

    ════════════════════════════════════════════════════════
    Now process the next input JSON and output exactly one JSON object.
    ════════════════════════════════════════════════════════

    '''
OUTPUT_SCHEMA={
 "type":"object", "additionalProperties":False,
 "properties":{
  "cognitive_state":{"type":"string"},
  "failure_risk_10min":{"type":"string","enum":["LOW","MEDIUM","HIGH"]},
  "reasoning":{"type":"string","maxLength":300},
  "command":{"type":"string","enum":list(dec.COMMANDS)},
  "args":{"type":"object","additionalProperties":False,"properties":{"reason":{"type":"string","maxLength":80},"message":{"type":"string","maxLength":200}},"required":["reason","message"]}
 }, "required":["cognitive_state","failure_risk_10min","reasoning","command","args"]
}


def resolve_model() -> str:
    try:
        request = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            data = json.loads(response.read().decode())
        installed = {str(item.get("name")) for item in data.get("models", []) if isinstance(item, dict)}
        return MODEL_NAME if MODEL_NAME in installed else FALLBACK_MODEL if FALLBACK_MODEL in installed else MODEL_NAME
    except Exception:
        return MODEL_NAME


def validate_payload(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload_not_object"
    for key in ("system_log", "facts_text", "frame_narrative", "safety_floor_command", "frame_id"):
        if key not in payload:
            return False, f"missing_{key}"
    if payload["safety_floor_command"] not in dec.COMMANDS:
        return False, "invalid_floor"
    facts = payload.get("facts")
    if facts is not None and not isinstance(facts, dict):
        return False, "facts_not_object"
    return True, ""


def user_message(payload: Dict[str, Any]) -> str:
    # Only these textual blocks are presented to the LLM. The raw facts dict
    # exists for Python-side validation and is never embedded in this message.
    return "\n\n".join([
        "AUTHORITATIVE CURRENT EVIDENCE — use only the facts below.",
        "Every time, duration, count and scenario in earlier examples is fictional. Never copy any of them. If a fact is absent below, call it unavailable; do not invent it.",
        str(payload["system_log"]),
        str(payload["facts_text"]),
        str(payload["frame_narrative"]),
        f"COLD SAFETY FLOOR: {payload['safety_floor_command']}. Normally meet or exceed it. "
        "You MAY lower the command by AT MOST one tier below this floor, and ONLY when the floor is "
        "GENTLE_ALERT or MODERATE_ALERT and the trajectory and current evidence clearly show the "
        "deterministic reading is a transient false positive (e.g. a settled mirror-check or a single "
        "long blink that has fully recovered). NEVER lower a URGENT_ALERT or EMERGENCY_ALERT floor, "
        "and never drop more than one tier.",
        "Return one JSON object only. The evidence above is the only source of truth for this evaluation.",
    ])


def _as_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _microsleep_event_count(payload: Dict[str, Any]) -> int:
    facts = payload.get("facts")
    return _as_nonnegative_int(facts.get("microsleep_event_count", 0)) if isinstance(facts, dict) else 0


_NAV_RES = [
    re.compile(r"\bwithin\s+\d+(?:\.\d+)?\s*(?:km|kilometers?|miles?|mi|m)\b", re.I),
    re.compile(r"\bin\s+\d+(?:\.\d+)?\s*(?:km|kilometers?|miles?|mi)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:km|kilometers?|miles?)\b", re.I),
    re.compile(r"\b(?:at\s+|the\s+)?(?:next|nearest)\s+(?:exit|rest[\s-]?stop|rest\s+area|service\s+station|gas\s+station|petrol\s+station|junction)\b", re.I),
    re.compile(r"\bin\s+\d+\s*(?:minutes?|mins?|min)\b", re.I),
]
# Directives that contradict a NO_ACTION command.
_ALARM_RE = re.compile(r"pull[\s-]?over|take a break|stop driving|rest stop|rest area|do not continue", re.I)


def _scrub_message(message: str) -> str:
    """Strip navigation specifics the system cannot know (distances, exits, rest
    stops, ETAs). The model is instructed not to invent them; this enforces it
    deterministically so a hallucinated 'within 5 km' can never reach the driver."""
    out = message
    for pattern in _NAV_RES:
        out = pattern.sub("", out)
    # Drop a preposition left dangling before punctuation/end ("pull over at." -> "pull over.").
    out = re.sub(r"\s+\b(?:at|in|to|within|near|by|towards?|until)\b\s*(?=[.,;!?]|$)", "", out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;!?])", r"\1", out)
    return out.strip(" ,;-").strip()


def _guard_deterministic_states(value: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply facts-only guards after model decoding and before arbitration."""
    corrected = dict(value)
    args = dict(corrected.get("args") or {})
    corrected["args"] = args

    # T4 is not called for reflex-active microsleep events. Therefore, an LLM
    # EMERGENCY is unsupported and is capped before the evaluator sees it.
    if corrected.get("command") == "EMERGENCY_ALERT":
        corrected["command"] = "URGENT_ALERT"
        corrected["failure_risk_10min"] = "HIGH"
        args["reason"] = "agent_emergency_capped_non_reflex"
        args["message"] = "Strong fatigue indicators are present. Pull over at the next safe location."
        if corrected.get("cognitive_state") == "Microsleep":
            corrected["cognitive_state"] = "Fighting Sleep"

    # This label is permitted only after an objectively verified >=2s event.
    if corrected.get("cognitive_state") == "Post-Microsleep Recovery" and _microsleep_event_count(payload) == 0:
        corrected["cognitive_state"] = "Active Fatigue"
        corrected["reasoning"] = "No confirmed microsleep event exists in the deterministic ten-minute facts; persistent fatigue is classified as Active Fatigue."
        args["reason"] = "active_fatigue_no_confirmed_microsleep"
        message = str(args.get("message", ""))
        if "microsleep" in message.lower() or "lost control" in message.lower():
            args["message"] = "Fatigue indicators remain elevated. Stop at a safe location and take a break."

    # Output hygiene: strip invented navigation specifics, and keep the spoken
    # message consistent with the command (no "take a break" on a NO_ACTION).
    args["message"] = _scrub_message(str(args.get("message", "")))
    if corrected.get("command") == "NO_ACTION" and _ALARM_RE.search(args["message"]):
        args["message"] = "All clear. Stay focused and drive safely."

    return corrected


def validate_output(value: Any, floor: str) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "output_not_object"
    if set(value) != set(OUTPUT_SCHEMA["required"]):
        return False, "wrong_top_level_keys"
    if not isinstance(value.get("cognitive_state"), str) or not value["cognitive_state"].strip():
        return False, "invalid_cognitive_state"
    if value.get("command") not in dec.COMMANDS:
        return False, "invalid_command"
    if value.get("failure_risk_10min") not in {"LOW", "MEDIUM", "HIGH"}:
        return False, "invalid_risk"
    args = value.get("args")
    if not isinstance(args, dict) or set(args) != {"reason", "message"}:
        return False, "invalid_args"
    if not isinstance(value.get("reasoning"), str) or len(value["reasoning"]) > 300:
        return False, "invalid_reasoning"
    if not isinstance(args["reason"], str) or len(args["reason"]) > 80:
        return False, "invalid_reason_code"
    if not isinstance(args["message"], str) or len(args["message"]) > 200:
        return False, "invalid_message"
    # The arbiter, not the model wrapper, applies the cold safety floor.
    # Keeping a valid lower command lets T5 log an explicit cold-floor override.
    return True, ""


def invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    selected = resolve_model()
    request = {
        "model": selected,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "format": OUTPUT_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(payload)},
        ],
        "options": {"temperature": 0, "num_predict": MAX_TOKENS, "num_ctx": NUM_CTX, "repeat_penalty": 1.05},
    }
    http_request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(request).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(http_request, timeout=TIMEOUT) as response:
            raw = json.loads(response.read(2_000_000).decode("utf-8", errors="replace"))
        content = raw.get("message", {}).get("content", "")
        parsed = _guard_deterministic_states(json.loads(content), payload)
        ok, reason = validate_output(parsed, str(payload["safety_floor_command"]))
        if not ok:
            raise ValueError(f"model_output_{reason}")
        result = {
            "status": "ok",
            "output": {"command": parsed["command"], "args": parsed["args"]},
            "cognitive_state": parsed["cognitive_state"],
            "failure_risk_10min": parsed["failure_risk_10min"],
            "reasoning": parsed["reasoning"],
            "model": selected,
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "error",
            "output": {"command": "NO_ACTION", "args": {"reason": "agent_unavailable", "message": ""}},
            "detail": f"{type(exc).__name__}: {exc}",
            "model": selected,
        }
    result["latency_ms"] = int((time.time() - started) * 1000)
    return result


def log(payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    try:
        row = {
            "timestamp_ms": int(time.time() * 1000),
            "frame_id": payload.get("frame_id"),
            "model": result.get("model"),
            "latency_ms": result.get("latency_ms"),
            "result": result,
            "system_log": payload.get("system_log"),
            "facts_text": payload.get("facts_text"),
            "frame_narrative": payload.get("frame_narrative"),
        }
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    ok, reason = validate_payload(payload)
    if not ok:
        return {"status": "error", "output": {"command": "NO_ACTION", "args": {"reason": "invalid_payload", "message": ""}}, "detail": reason}
    result = invoke(payload)
    log(payload, result)
    return result


def probe() -> int:
    demo = {
        "frame_id": 1,
        "safety_floor_command": "GENTLE_ALERT",
        "system_log": "SYSTEM LOG — AUTHORITATIVE 10-MINUTE SENSOR HISTORY. The retained window contains 10 captured frames. PERCLOS remained stable from 0.020 to 0.020. There were 0 confirmed yawning events and 0 confirmed reflex microsleep events.",
        "facts_text": "DETERMINISTIC FACTS — AUTHORITATIVE. Cold command: GENTLE_ALERT. Cold features: eyes. Confirmed reflex microsleep events in the retained window: 0. Safety floor: GENTLE_ALERT.",
        "frame_narrative": "CURRENT FRAME NARRATIVE — AUTHORITATIVE. Frame 1: driver state is EYES_OPEN. EAR is 0.3300. PERCLOS is 0.0200. Current eye closure is 0 ms. Talking is false. Current clock time is 18:50 and trip duration is 3 minutes.",
        "facts": {"microsleep_event_count": 0},
    }
    output = process(demo)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output.get("status") == "ok" else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        raise SystemExit(probe())
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = {}
        print(json.dumps(process(payload), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
