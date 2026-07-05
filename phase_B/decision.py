"""AuraDrive v8 canonical decision contract — cold and agent only."""
from __future__ import annotations
from typing import Any, Dict

COMMANDS = ("NO_ACTION", "GENTLE_ALERT", "MODERATE_ALERT", "URGENT_ALERT", "EMERGENCY_ALERT")
COMMAND_RANK = {command: index for index, command in enumerate(COMMANDS)}
COMMAND_TO_SEVERITY = {
    "NO_ACTION": "LOW", "GENTLE_ALERT": "LOW", "MODERATE_ALERT": "MEDIUM",
    "URGENT_ALERT": "HIGH", "EMERGENCY_ALERT": "HIGH",
}
COMMAND_TO_SLEEP_MS = {
    "NO_ACTION": 10000, "GENTLE_ALERT": 8000, "MODERATE_ALERT": 5000,
    "URGENT_ALERT": 0, "EMERGENCY_ALERT": 0,
}
MICROSLEEP_THRESHOLD_MS = 2000
MAX_MESSAGE_CHARS = 200


def rank(command: Any) -> int:
    return COMMAND_RANK.get(str(command), COMMAND_RANK["EMERGENCY_ALERT"])


def make_decision(*, command: str, decision_source: str, reason: str, message: str,
                  cognitive_state: str = "Unknown", failure_risk_10min: str = "LOW",
                  evidence: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if command not in COMMAND_RANK:
        command = "EMERGENCY_ALERT"
        reason = f"invalid_command::{reason}"
    if failure_risk_10min not in {"LOW", "MEDIUM", "HIGH"}:
        failure_risk_10min = "HIGH"
    return {
        "schema_version": "8.0",
        "decision_source": str(decision_source),
        "command": command,
        "severity": COMMAND_TO_SEVERITY[command],
        "sleep_ms": COMMAND_TO_SLEEP_MS[command],
        "cognitive_state": str(cognitive_state or "Unknown"),
        "failure_risk_10min": failure_risk_10min,
        "message": str(message or "")[:MAX_MESSAGE_CHARS],
        "reason": str(reason or "unknown"),
        "evidence": dict(evidence or {}),
    }


def from_agent(agent: Dict[str, Any]) -> Dict[str, Any]:
    output = agent.get("output") if isinstance(agent, dict) else {}
    output = output if isinstance(output, dict) else {}
    args = output.get("args") if isinstance(output.get("args"), dict) else {}
    command = str(output.get("command", "EMERGENCY_ALERT"))
    return make_decision(
        command=command, decision_source="agent",
        reason=str(args.get("reason", "agent_reason")), message=str(args.get("message", "")),
        cognitive_state=str(agent.get("cognitive_state", "Unknown")),
        failure_risk_10min=str(agent.get("failure_risk_10min", "LOW")),
        evidence={"agent_reasoning": str(agent.get("reasoning", ""))[:300], "agent_raw": dict(agent)},
    )
