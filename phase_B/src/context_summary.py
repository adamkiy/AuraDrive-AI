"""Deterministic aggregation of the retained window into evidence for the agent.

This module is the boundary between measurement and reasoning. It reduces the
ten-minute perception window to three blocks of prose plus a facts dictionary,
and those blocks are the only thing the language model ever sees.

The design rests on one principle: Python owns every calculation, and the model
interprets conclusions Python has already drawn. Trends are labelled here, event
counts are established here, and each measurement reaches the model inside a
sentence that already states what it means. The model is therefore asked to read
a described trajectory rather than to re-derive a verdict from raw figures, which
is exactly the task a small model is reliable at.

The facts dictionary is kept Python-side for validation and never sent. It is
what allows the agent guards to check a model claim against the record, most
importantly the microsleep count that authorises the recovery state label.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Sequence

import decision as dec


def n(value: Any, default: float = 0.0) -> float:
    """Coerce a metric to a usable number, substituting a default when it is not.

    Every field arriving from perception passes through here before it can reach
    a calculation. A missing key, a non-numeric value or a NaN produced by a bad
    landmark fit all resolve to the default, so a single malformed frame degrades
    the summary rather than propagating a corrupt figure into the evidence the
    agent is asked to trust.

    Parameters
    ----------
    value : Any
        The raw field taken from a frame record.
    default : float
        The value to substitute when the input cannot be trusted as a number.

    Returns
    -------
    float
        The coerced measurement, or the default if coercion was not possible.
    """
    try:
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def transitions(rows: Sequence[Dict[str, Any]], predicate) -> int:
    """Count how many times a condition began to hold across the window.

    Counting rising edges rather than matching frames is what turns a per-frame
    state into an event count. A two-second eye closure occupies roughly sixty
    frames, so counting frames would report sixty closures where one occurred,
    and every event figure given to the agent would be inflated by the frame
    rate. This is the function that makes counts such as yawns and microsleeps
    mean episodes.

    Parameters
    ----------
    rows : sequence of dict
        The retained frame records, in chronological order.
    predicate : callable
        Test applied to each record, defining the condition being counted.

    Returns
    -------
    int
        The number of distinct episodes in which the condition became true.
    """
    count = 0
    previous = False
    for row in rows:
        current = bool(predicate(row))
        if current and not previous:
            count += 1
        previous = current
    return count


def trend(values: List[float]) -> tuple[float, float, float, str]:
    """Reduce a metric series to its endpoints, its mean, and a direction label.

    The label is the point of the function. Deciding whether a series is rising
    is a numerical judgement, and it is made here rather than left to the model,
    so the agent receives a stated direction instead of a column of figures to
    compare. The deadband around the endpoints keeps ordinary measurement noise
    from being reported as a developing trend.

    Parameters
    ----------
    values : list of float
        The metric sampled across the retained window, in chronological order.

    Returns
    -------
    tuple
        The first and last samples, the mean, and a direction label of RISING,
        FALLING, STABLE, or UNKNOWN when the window holds no samples.
    """
    if not values:
        return 0.0, 0.0, 0.0, "UNKNOWN"
    first, last = values[0], values[-1]
    label = "RISING" if last > first + 0.02 else "FALLING" if last < first - 0.02 else "STABLE"
    return first, last, float(mean(values)), label


@dataclass(frozen=True)
class ContextSummary:
    system_log: str
    facts_text: str
    frame_narrative: str
    facts: Dict[str, Any]


def build_context_summary(
    frame: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    cold: Dict[str, Any],
    floor: str,
) -> ContextSummary:
    """Build the three evidence blocks and the facts the agent call depends on.

    This is where the window becomes something a language model can read. The
    history block describes the trajectory, the facts block states what the
    deterministic layers concluded and what floor applies, and the narrative
    block describes the current frame. The current frame is appended if the
    window has not caught up with it, so the agent is never asked to judge a
    frame it cannot see.

    Parameters
    ----------
    frame : dict
        The metrics for the frame being evaluated.
    history : sequence of dict
        The retained window, which may lag the current frame by a moment.
    cold : dict
        The deterministic decision for this frame, quoted to the agent so its
        starting point is explicit rather than inferred.
    floor : str
        The minimum command the agent is permitted to return, stated in the
        evidence so the constraint is visible to the model as well as enforced
        downstream by the arbiter.

    Returns
    -------
    ContextSummary
        The three prose blocks sent to the model, plus the facts dictionary
        retained for Python-side validation of the model's reply.
    """
    records = sorted([dict(row) for row in history if isinstance(row, dict)], key=lambda row: n(row.get("timestamp_ms")))
    if not records or records[-1].get("frame_id") != frame.get("frame_id"):
        records.append(dict(frame))

    perclos_values = [n(row.get("PERCLOS")) for row in records]
    blink_values = [n(row.get("Blinks/min")) for row in records]
    per_a, per_b, per_avg, per_trend = trend(perclos_values)
    blink_a, blink_b, blink_avg, blink_trend = trend(blink_values)

    yawn_events = transitions(records, lambda row: str(row.get("Mouth_State", "NORMAL")) == "YAWNING" and not bool(row.get("Is_Talking", False)))
    closure_events = transitions(records, lambda row: str(row.get("Driver_State")) == "EYES_CLOSED")
    pitch_events = transitions(records, lambda row: bool(row.get("Head_Pitch_Down_Active", False)))
    roll_events = transitions(records, lambda row: bool(row.get("Head_Roll_Active", False)))
    # Count only direct, threshold-confirmed microsleep episodes. This is the
    # sole fact that authorizes the Post-Microsleep Recovery label.
    microsleep_events = transitions(records, lambda row: n(row.get("Eyes Closed Duration")) >= dec.MICROSLEEP_THRESHOLD_MS)
    longest_closure = max([n(row.get("Eyes Closed Duration")) for row in records] or [0.0])
    face_loss = transitions(records, lambda row: bool(row.get("no_face", False)))

    current = {
        "frame_id": frame.get("frame_id"),
        "time": str(frame.get("time_of_day", "unknown")),
        "trip": int(n(frame.get("trip_duration_min"))),
        "state": str(frame.get("Driver_State", "UNKNOWN")),
        "ear": n(frame.get("EAR")),
        "perclos": n(frame.get("PERCLOS")),
        "closed": n(frame.get("Eyes Closed Duration")),
        "blink": n(frame.get("Blinks/min")),
        "mar": n(frame.get("MAR")),
        "mouth": str(frame.get("Mouth_State", "UNKNOWN")),
        "yawns": n(frame.get("Yawns/min")),
        "talking": bool(frame.get("Is_Talking", False)),
        "pitch": n(frame.get("Head_Pitch")),
        "pitch_base": n(frame.get("Head_Pitch_Baseline")),
        "pitch_delta": n(frame.get("Head_Pitch_Down_Delta")),
        "pitch_active": bool(frame.get("Head_Pitch_Down_Active", False)),
        "roll": n(frame.get("Head_Roll")),
        "roll_base": n(frame.get("Head_Roll_Baseline")),
        "roll_delta": n(frame.get("Head_Roll_Delta")),
        "roll_active": bool(frame.get("Head_Roll_Active", False)),
        "pose_calibrated": bool(frame.get("Head_Pose_Calibrated", False)),
        "face_detected": not bool(frame.get("no_face", False)),
    }

    system_log = (
        "SYSTEM LOG — AUTHORITATIVE 10-MINUTE SENSOR HISTORY\n"
        f"The retained window contains {len(records)} captured frames. PERCLOS moved from {per_a:.3f} to {per_b:.3f} (mean {per_avg:.3f}; trend {per_trend}). "
        f"Blink rate moved from {blink_a:.1f} to {blink_b:.1f} per minute (mean {blink_avg:.1f}; trend {blink_trend}). "
        f"There were {yawn_events} confirmed yawning events, {closure_events} eye-closure episodes, and the longest closure was {longest_closure:.0f} ms. "
        f"There were {microsleep_events} confirmed reflex microsleep event(s), {pitch_events} directional head-down episodes and {roll_events} lateral head-roll episodes. "
        f"Face tracking was lost in {face_loss} episode(s). Current clock time is {current['time']} and current trip duration is {current['trip']} minutes.\n"
        f"The deterministic cold decision is {cold.get('command')}; the minimum permitted command is {floor}."
    )

    facts_text = (
        "DETERMINISTIC FACTS — AUTHORITATIVE\n"
        f"Cold command: {cold.get('command')}; cold features: {', '.join(cold.get('evidence', {}).get('active_features', [])) or 'none'}; cold index: {cold.get('evidence', {}).get('cold_index', 'unavailable')}.\n"
        f"Confirmed reflex microsleep events in the retained window: {microsleep_events}. "
        f"Safety floor: {floor}. Head-pose calibration is {'complete' if current['pose_calibrated'] else 'not complete'}; face tracking is {'available' if current['face_detected'] else 'unavailable'}."
    )

    frame_narrative = (
        "CURRENT FRAME NARRATIVE — AUTHORITATIVE\n"
        f"Frame {current['frame_id']}: driver state is {current['state']}. EAR is {current['ear']:.4f}; PERCLOS is {current['perclos']:.4f}; current eye closure is {current['closed']:.0f} ms; blink rate is {current['blink']:.1f} per minute. "
        f"MAR is {current['mar']:.4f}; mouth state is {current['mouth']}; yawn count in the rolling minute is {current['yawns']:.1f}; talking is {str(current['talking']).lower()}. "
        f"Pitch is {current['pitch']:.2f} degrees with personal baseline {current['pitch_base']:.2f}; directional downward deviation is {current['pitch_delta']:.2f} degrees and head-down active is {str(current['pitch_active']).lower()}. "
        f"Roll is {current['roll']:.2f} degrees with personal baseline {current['roll_base']:.2f}; lateral roll deviation is {current['roll_delta']:.2f} degrees and roll-tilt active is {str(current['roll_active']).lower()}."
    )

    facts = {
        "current": current,
        "yawn_events": yawn_events,
        "closure_events": closure_events,
        "pitch_events": pitch_events,
        "roll_events": roll_events,
        "microsleep_event_count": microsleep_events,
        "longest_closure_ms": longest_closure,
        "perclos": (per_a, per_b, per_avg, per_trend),
        "blink": (blink_a, blink_b, blink_avg, blink_trend),
    }
    return ContextSummary(system_log, facts_text, frame_narrative, facts)
