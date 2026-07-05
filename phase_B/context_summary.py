"""Deterministic 10-minute aggregation. The LLM receives text outputs only."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Sequence

import decision as dec


def n(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def transitions(rows: Sequence[Dict[str, Any]], predicate) -> int:
    count = 0
    previous = False
    for row in rows:
        current = bool(predicate(row))
        if current and not previous:
            count += 1
        previous = current
    return count


def trend(values: List[float]) -> tuple[float, float, float, str]:
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
