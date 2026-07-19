"""Cold-versus-agent safety arbiter with bounded temporal recovery."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import decision as dec


@dataclass
class Evaluation:
    final: Dict[str, Any]
    reason: str


class TemporalGuard:
    """Avoid abrupt de-escalation after a real alert while preserving immediate escalation.

    A graded alert (cold / agent) recovers gradually: held for `min_step_seconds`,
    then stepped down one tier at a time. A REFLEX microsleep EMERGENCY, however,
    is a point event — not a graded trajectory — so it is held only briefly
    (`reflex_release_seconds`) and then released DIRECTLY to the current
    deterministic truth, instead of decaying EMERGENCY -> URGENT -> ... over ~30 s.
    `peak_source` tracks who set the level currently held, so the reflex rule keeps
    applying across the held frames (where the published source reads temporal_hold).
    """

    def __init__(self, min_step_seconds: float = 8.0, reflex_release_seconds: float | None = None) -> None:
        """Configure how slowly the published alert level is allowed to fall.

        Only de-escalation is timed. Escalation is never delayed, because any
        wait before raising an alert is time the driver spends unwarned.

        Parameters
        ----------
        min_step_seconds : float
            How long a graded alert is held before it may step down one tier.
        reflex_release_seconds : float or None
            How long a reflex EMERGENCY is held before releasing to current
            truth; read from the environment when not supplied.

        Returns
        -------
        None
            The constructor only prepares internal state.
        """
        self.min_step_seconds = float(min_step_seconds)
        self.reflex_release_seconds = float(
            reflex_release_seconds if reflex_release_seconds is not None
            else os.getenv("AURADRIVE_REFLEX_RELEASE_SEC", "2.5")
        )
        self.last: Dict[str, Any] | None = None
        self.last_change = 0.0
        self.peak_source: str = ""   # who set the level currently held

    def apply(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Smooth a proposed decision in time, letting it rise freely but not fall.

        Escalation passes through untouched. A fall is treated differently
        depending on what set the current level: a reflex microsleep was a point
        event, so it is held briefly and then released straight to present truth,
        while a graded alert decays one tier at a time. A reasoned one-tier drop
        from the agent is also allowed through, since that is a judgement rather
        than the deterministic flicker this guard exists to absorb.

        Parameters
        ----------
        candidate : dict
            The decision the arbiter proposes to publish on this frame.

        Returns
        -------
        dict
            The decision actually published, which may be the candidate, the
            held previous level, or a single step down from it.
        """
        now = time.monotonic()
        if self.last is None:
            self.last = dict(candidate)
            self.last_change = now
            self.peak_source = str(candidate.get("decision_source"))
            return dict(candidate)

        old_rank = dec.rank(self.last.get("command"))
        candidate_rank = dec.rank(candidate.get("command"))
        final = dict(candidate)

        if candidate_rank < old_rank:
            elapsed = now - self.last_change
            deliberate_agent = str(candidate.get("decision_source")) == "agent"
            if self.peak_source == "reflex":
                # Point-event microsleep: hold briefly, then release straight to the
                # current decision (cold reflects the live eye state) — no slow decay.
                if elapsed < self.reflex_release_seconds:
                    final = dict(self.last)
                    final["decision_source"] = "temporal_hold"
                    final["reason"] = "post_microsleep_hold"
                else:
                    final = dict(candidate)
            elif deliberate_agent and (old_rank - candidate_rank) <= 1:
                # A reasoned one-tier agent de-escalation is a deliberate decision,
                # not cold flicker — let it through rather than holding the old level.
                final = dict(candidate)
            elif elapsed < self.min_step_seconds:
                final = dict(self.last)
                final["decision_source"] = "temporal_hold"
                final["reason"] = "bounded_recovery_hold"
            elif old_rank - candidate_rank > 1:
                step_command = dec.COMMANDS[old_rank - 1]
                final = dict(candidate)
                final["command"] = step_command
                final["severity"] = dec.COMMAND_TO_SEVERITY[step_command]
                final["sleep_ms"] = dec.COMMAND_TO_SLEEP_MS[step_command]
                final["decision_source"] = "temporal_recovery"
                final["reason"] = "bounded_recovery_step"

        if dec.rank(final.get("command")) != old_rank:
            self.last_change = now
            self.peak_source = str(final.get("decision_source"))
        self.last = dict(final)
        return final


_SOFTEN_CEILING = "MODERATE_ALERT"  # the LLM may soften only at/below this cold floor


def evaluate(cold: Dict[str, Any], agent: Dict[str, Any]) -> Evaluation:
    """Arbitrate two canonical decisions (cold oracle vs. agent). Cold is the
    deterministic safety floor; the agent refines it. Let delta = rank(agent) -
    rank(cold):

      * delta >= 0  -> ACCEPT the agent (it agrees or escalates). Escalation is
                       always allowed; large jumps (delta >= +3) are flagged for
                       audit but still pass.
      * delta < 0   -> the agent wants to SOFTEN. Governed by a HARD FLOOR:
          - cold >= URGENT_ALERT  -> the LLM may NEVER soften a high-risk floor.
                                     Hold the full cold command (urgent_floor_hold).
          - cold in {NO, GENTLE, MODERATE} -> at most ONE tier of softening is
                                     allowed (this is where deterministic false
                                     positives actually occur):
              delta == -1 -> accept the agent (agent_softened).
              delta <= -2 -> clamp to cold - 1 (agent_under_alert_capped).

    Rationale: a missed detection is a crash, a false alarm is an inconvenience,
    and the LLM is the less reliable component, so it can lower caution only in
    the mild band and never at URGENT or above. The agent itself is capped to
    URGENT upstream in agent.py, so an EMERGENCY here always originated from a
    deterministic layer, either the reflex latch or a cold score whose
    unamplified base cleared the calibration floor. Such a floor is caught by
    the same URGENT-or-above branch and held in full.

    Parameters
    ----------
    cold : dict
        The deterministic decision, which defines the floor for this frame.
    agent : dict
        The model's decision, already schema-validated and capped upstream.

    Returns
    -------
    Evaluation
        The decision to publish, paired with the arbitration reason that names
        which rule applied, so the audit trail records why and not just what.
    """
    cold_rank = dec.rank(cold.get("command"))
    agent_rank = dec.rank(agent.get("command"))
    delta = agent_rank - cold_rank

    # ── ACCEPT: agent agrees or escalates (escalation is always safe) ──
    if delta >= 0:
        large_jump = delta >= 3
        final = dict(agent)
        final["decision_source"] = "agent"
        final["reason"] = "agent_accepted_flagged" if large_jump else "agent_accepted"
        final["evidence"] = {
            **dict(agent.get("evidence") or {}),
            "cold_command": cold.get("command"),
            "arbitration": "agent_accepted",
            "tier_delta": delta,
            "large_escalation_flagged": large_jump,
        }
        return Evaluation(final=final, reason=final["reason"])

    # ── AGENT WANTS TO SOFTEN (delta < 0) ──
    mild = cold_rank <= dec.rank(_SOFTEN_CEILING)

    if not mild:
        # HARD FLOOR at URGENT+: the LLM may never lower a high-risk cold floor.
        final = dict(cold)
        final["decision_source"] = "safety_override"
        final["reason"] = "urgent_floor_hold"
        final["evidence"] = {
            **dict(cold.get("evidence") or {}),
            "agent_command": agent.get("command"),
            "arbitration": "urgent_hard_floor",
            "tier_delta": delta,
        }
        return Evaluation(final=final, reason="urgent_floor_hold")

    if delta == -1:
        # One tier of softening in the mild band — the agent earns its keep here.
        final = dict(agent)
        final["decision_source"] = "agent"
        final["reason"] = "agent_softened"
        final["evidence"] = {
            **dict(agent.get("evidence") or {}),
            "cold_command": cold.get("command"),
            "arbitration": "agent_softened_one_tier",
            "tier_delta": delta,
        }
        return Evaluation(final=final, reason="agent_softened")

    # delta <= -2 in the mild band: grant one tier, refuse the larger drop.
    command = dec.COMMANDS[max(cold_rank - 1, 0)]
    final = dict(cold)
    final["command"] = command
    final["severity"] = dec.COMMAND_TO_SEVERITY[command]
    final["sleep_ms"] = dec.COMMAND_TO_SLEEP_MS[command]
    final["decision_source"] = "safety_override"
    final["reason"] = "agent_under_alert_capped"
    final["evidence"] = {
        **dict(cold.get("evidence") or {}),
        "agent_command": agent.get("command"),
        "arbitration": "under_alert_override",
        "tier_delta": delta,
    }
    return Evaluation(final=final, reason="agent_under_alert_capped")
