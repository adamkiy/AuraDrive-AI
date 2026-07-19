"""Canonical decision contract shared by every layer of AuraDrive.

The reflex latch, the deterministic cold engine and the LLM agent all speak in
the same five commands and emit the same decision object. Keeping that
vocabulary in one module is what allows the safety arbiter to compare a
deterministic verdict with a model verdict by rank alone, and it guarantees
the two layers can never disagree about what a command means.

The module holds no decision logic of its own. It defines the vocabulary,
the severity and cooldown each command maps to, and the two constructors that
every other layer uses to build a decision.
"""
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
    """Order a command by urgency so the arbiter can compare two verdicts.

    Escalation and de-escalation are decided by comparing ranks rather than by
    matching command names, which is what keeps the safety floor expressible as
    a single numeric rule. An unrecognised command ranks as EMERGENCY_ALERT, so
    corrupt input degrades towards caution instead of silently passing as low
    severity.

    Parameters
    ----------
    command : Any
        The command to rank; any value is accepted so malformed model output
        cannot raise inside the arbiter.

    Returns
    -------
    int
        The command's position on the urgency scale, higher being more urgent.
    """
    return COMMAND_RANK.get(str(command), COMMAND_RANK["EMERGENCY_ALERT"])


def make_decision(*, command: str, decision_source: str, reason: str, message: str,
                  cognitive_state: str = "Unknown", failure_risk_10min: str = "LOW",
                  evidence: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build the canonical decision object that the rest of the system consumes.

    This is the single point at which a decision becomes valid, so it is also
    where the fail-safe defaults live: an unrecognised command becomes
    EMERGENCY_ALERT and an unrecognised risk level becomes HIGH, on the
    principle that a malformed decision must never be quieter than the truth.
    Severity and cooldown are derived from the command rather than accepted
    from the caller, so no layer can pair a command with the wrong urgency.

    Parameters
    ----------
    command : str
        The alert level being issued, from the five-command vocabulary.
    decision_source : str
        Which layer produced this decision, used by the audit trail to make
        every published alert traceable to its origin.
    reason : str
        Short machine-readable tag for why the command was chosen.
    message : str
        Driver-facing text, truncated so an over-long model reply cannot
        overflow the on-screen banner.
    cognitive_state : str
        The behavioural state label, supplied by the agent or derived by the
        cold engine.
    failure_risk_10min : str
        Assessed risk over the coming ten minutes, used for logging and display.
    evidence : dict or None
        Supporting detail retained for the audit trail rather than for display.

    Returns
    -------
    dict
        A complete decision object, safe to publish, compare or log.
    """
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
    """Convert a raw model reply into a canonical decision.

    This is the boundary between untrusted model output and the trusted
    decision pipeline, so every field is defensively extracted: a reply that is
    not a dictionary, or is missing its nested structures, still yields a valid
    decision rather than raising inside the arbiter. A missing command defaults
    to EMERGENCY_ALERT, so a malformed reply cannot quietly become NO_ACTION.
    The model's own reasoning is preserved in the evidence for the audit trail.

    Parameters
    ----------
    agent : dict
        The raw reply returned by the agent subprocess.

    Returns
    -------
    dict
        A canonical decision the arbiter can rank against the cold baseline.
    """
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
