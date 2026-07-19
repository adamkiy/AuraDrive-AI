"""Deterministic decision oracle and the safety-net fallback.

This engine grades fatigue from validated sensor metrics without reference to
the language model, and it is what the system falls back to whenever the model
is slow, unavailable or rejected. Being the last line of defence sets three
requirements: it must never trust an unvalidated number, never crash on bad
input, and be explicit about which of its thresholds rest on published evidence
and which are engineering judgement.

Method
------
Fatigue is scored on a 0 to 10 scale as a weighted fusion of PERCLOS, closure
duration, blink rate and yawn rate. The weights are not chosen by feel: they are
derived through the Analytic Hierarchy Process from a documented pairwise
comparison matrix, accepted only after its consistency ratio fell well below
Saaty's bound. Bounded context multipliers for time of day and trip duration
then amplify the base score, subject to a calibration floor that prevents
context alone from manufacturing an EMERGENCY out of weak physiological
evidence.

What is actually research-validated
-----------------------------------
Only two numbers here are backed by primary literature:

    PERCLOS graded bands   Grace (2001): moderate 0.08 to 0.14, severe above 0.14
    EAR closure point      Soukupova and Cech (2016): EAR = 0.2

Everything else, meaning the closure sub-bands, the blink bands, the yawn
thresholds, the multiplier values and the score band boundaries, is a reasoned
engineering choice rather than a validated constant. Each is tagged
[ENGINEERING] at its definition so a reader can tell exactly what rests on
evidence and what rests on judgement. The [ENGINEERING] values are not presented
as medically validated anywhere in the project report.

Design decisions worth knowing
------------------------------
EAR is deliberately absent from the score. EAR, closure duration and PERCLOS all
derive from the same raw signal, so scoring EAR as a third input would count the
same physiology three times. PERCLOS captures aggregate closure and closure
duration captures the severity of the current event; EAR remains in the payload
as context for the agent but does not inflate the deterministic score.

The cognitive state label is derived from whichever signal dominated the score,
not from the score band alone, so a closure-driven result and a blink-driven
result at the same severity are not described identically.

Command to severity mapping is imported from decision.py rather than duplicated
here, so this engine and the agent can never disagree about what a command means.

Sources
-------
[1] Dinges, D.F. and Grace, R. (1998). PERCLOS: A Valid Psychophysiological
    Measure of Alertness. FHWA Office of Motor Carriers, FHWA-MCRT-98-006.
[2] Grace, R. (2001). Drowsy Driver Monitor and Warning System. Carnegie Mellon
    Robotics Institute. PERCLOS bands: moderate 0.08 to 0.14, severe above 0.14.
[3] Soukupova, T. and Cech, J. (2016). Real-Time Eye Blink Detection using
    Facial Landmarks. 21st CVWW. EAR = 0.2, verified against Fig. 2.
[4] Stern, J.A., Boyer, D. and Schroeder, D. (1994). Blink Rate: A Possible
    Measure of Fatigue. Human Factors 36(2):285-297. Note that blink rate rises
    with fatigue, which does not support a low-only rule, so the bands here are
    symmetric.
[5] Williamson, A.M. and Feyer, A.M. (2000). Moderate sleep deprivation produces
    impairments equivalent to alcohol intoxication. Occup Environ Med
    57(10):649-655. 17 to 19 hours awake is roughly 0.05 percent BAC.

Claims not cited as validated
-----------------------------
Wierwille and Ellsworth (1994) originated the PERCLOS metric; the fact is
confirmed but the exact bibliographic record is unverified. McIntire et al.
(2014) is unverified. No physiological source for yawning frequency has been
confirmed, so yawn scoring is tagged [ENGINEERING] pending one.
"""


from __future__ import annotations
import os
from typing import Optional, Tuple

import decision as dec


# ════════════════════════════════════════════════════════════
#  VALID INPUT RANGES  (used for clamping — fail-safe, not trust)
# ════════════════════════════════════════════════════════════

_RANGES = {
    "PERCLOS":              (0.0, 1.0),
    "Blinks/min":           (0.0, 200.0),    # 200 is a generous hard ceiling
    "Eyes Closed Duration": (0.0, 60_000.0), # ms; clamp runaway values
    "Yawns/min":            (0.0, 60.0),
}

# Neutral defaults used when a value is missing or non-numeric.
# Chosen to be NON-alarming (so landmark noise doesn't fire false alerts)
# while still being flagged via input_valid=False.
_NEUTRAL_DEFAULTS = {
    "PERCLOS":              0.0,
    "Blinks/min":           15.0,   # mid-normal
    "Eyes Closed Duration": 0.0,
    "Yawns/min":            0.0,
}


def _safe_value(raw, key: str) -> Tuple[float, bool]:
    """
    Coerce `raw` to a float clamped into _RANGES[key].
    Returns (value, ok).  ok=False means the input was missing/non-numeric/
    out-of-range and was repaired — the caller records this.
    """
    lo, hi = _RANGES[key]
    if raw is None:
        return _NEUTRAL_DEFAULTS[key], False
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _NEUTRAL_DEFAULTS[key], False
    if v != v:                      # NaN check
        return _NEUTRAL_DEFAULTS[key], False
    ok = (lo <= v <= hi)
    return min(max(v, lo), hi), ok  # clamp regardless; report ok


# ════════════════════════════════════════════════════════════
#  SIGNAL SCORING FUNCTIONS  — each returns a score in [0, 10]
# ════════════════════════════════════════════════════════════

def _perclos_score(perclos: float) -> float:
    """
    [VALIDATED] PERCLOS bands anchored to Grace (2001) [2]:
        moderate 0.08–0.14, severe >0.14.
    Band SCORES are [ENGINEERING]:
        <0.08 → 0 ; 0.08–<0.15 → 4 ; 0.15–<0.25 → 8 ; ≥0.25 → 10
    The moderate band maps to 4 (was 5) so that Grace's *moderate* range plus a
    single co-signal lands in MODERATE_ALERT, not URGENT — URGENT requires the
    validated severe band (>=0.15) or genuine multi-signal convergence.
    """
    if perclos < 0.08:  return 0.0
    if perclos < 0.15:  return 4.0
    if perclos < 0.25:  return 8.0
    return 10.0


def _blink_score(blinks_per_min: float) -> float:
    """
    [ENGINEERING] Normal resting band ~10–18/min.  SYMMETRIC: both extremes
    scored equally — the literature on direction is mixed (Stern [4] reports
    a RISE with fatigue; highway-hypnosis work reports a fall).  Modest weight
    because this single signal is the most ambiguous.
    """
    b = blinks_per_min
    if 10.0 <= b <= 18.0:                          return 0.0
    if (8.0 <= b < 10.0)  or (18.0 < b <= 25.0):  return 3.0
    if (5.0 <= b < 8.0)   or (25.0 < b <= 35.0):  return 6.0
    return 9.0                                       # < 5 or > 35


def _closure_score(closed_ms: float) -> float:
    """
    [ENGINEERING] Eye-closure beyond a normal voluntary blink (~350–400 ms).
    The reflex layer intercepts ≥2000 ms, so this covers 400–1999 ms only.
    Sub-bands are a graded engineering judgement, not a published scale.
    """
    if closed_ms < 400:   return 0.0
    if closed_ms < 500:   return 2.5
    if closed_ms < 800:   return 5.0
    if closed_ms < 1200:  return 8.0
    return 10.0


def _yawn_score(yawns_per_min: int) -> float:
    """
    [ENGINEERING] Yawning is a widely accepted drowsiness marker, but no
    verified per-minute physiological threshold is currently cited [V3].
    Heuristic: 0→0, 1→3 (weak), ≥2→10 (strong convergent evidence).
    """
    if yawns_per_min == 0:  return 0.0
    if yawns_per_min == 1:  return 3.0
    return 10.0


# ════════════════════════════════════════════════════════════
#  CONTEXT MULTIPLIERS  — VALUES are [ENGINEERING]; the DIRECTION
#  (night + long trip = worse) is grounded in Williamson & Feyer [5].
#  Combined cap 2.5×.
# ════════════════════════════════════════════════════════════

_MAX_CONTEXT_MULTIPLIER = 2.5


def _time_of_day_multiplier(time_of_day: Optional[str]) -> float:
    """Amplify the base score according to circadian risk.

    The same physiological reading carries more risk at 03:00 than at midday,
    so the clock is treated as context rather than as evidence: it scales what
    the body is already showing but can never create a signal on its own. Any
    unparseable or out-of-range clock value returns a neutral multiplier, so a
    bad timestamp cannot silently inflate a decision. All bands are tagged
    [ENGINEERING] rather than presented as validated constants.

    Parameters
    ----------
    time_of_day : str or None
        The current clock time; anything unusable is treated as neutral.

    Returns
    -------
    float
        A multiplier of 1.0 or above, applied to the fused base score.
    """
    if time_of_day is None:
        return 1.0
    try:
        h = int(str(time_of_day).split(":")[0])
    except (ValueError, IndexError, AttributeError):
        return 1.0
    if not (0 <= h <= 23):
        return 1.0
    if h >= 23 or h < 6:   return 1.8   # [ENGINEERING] circadian nadir
    if 6 <= h < 9:          return 1.2   # [ENGINEERING] post-wake
    if 14 <= h < 16:        return 1.3   # [ENGINEERING] post-prandial dip
    return 1.0


def _trip_duration_multiplier(trip_min: Optional[int]) -> float:
    """Amplify the base score according to how long the driver has been driving.

    Time on task accumulates fatigue, so a given reading after three hours is
    more concerning than the same reading after ten minutes. Like the circadian
    multiplier this is context rather than evidence: it scales an existing
    signal and cannot manufacture one, and an unusable value returns neutral.
    All bands are tagged [ENGINEERING].

    Parameters
    ----------
    trip_min : int or None
        Minutes driven in this trip; anything unusable is treated as neutral.

    Returns
    -------
    float
        A multiplier of 1.0 or above, applied to the fused base score.
    """
    if trip_min is None:
        return 1.0
    try:
        t = float(trip_min)
    except (TypeError, ValueError):
        return 1.0
    if t < 60:     return 1.0
    if t < 120:    return 1.2    # [ENGINEERING]
    if t < 180:    return 1.35   # [ENGINEERING]
    return 1.5                    # [ENGINEERING]


# ════════════════════════════════════════════════════════════
#  WEIGHTS  (must sum to 1.0).  EAR deliberately does not participate.
#
#  Weights are DERIVED VIA AHP (Analytic Hierarchy Process), the same
#  methodology Wei et al. (2023) used.  Instead of choosing weights by feel,
#  we built a pairwise comparison matrix (Saaty 1-9 scale) with a documented
#  justification for every entry, took its principal eigenvector, and verified
#  consistency.  Consistency Ratio CR = 0.006 (< 0.10 → consistent).
#
#  Pairwise matrix (entry [i][j] = how much i outranks j), order
#  [perclos, closure, blink, yawn]:
#        perclos closure blink yawn
#  perc [ 1      2       5     4   ]   perclos vs closure=2 (validated gold std
#  clos [ 1/2    1       3     2   ]     slightly over acute event)
#  blnk [ 1/5    1/3     1     1   ]   perclos vs blink=5, vs yawn=4
#  yawn [ 1/4    1/2     1     1   ]   closure vs blink=3, vs yawn=2; blink~yawn
#
#  Eigenvector → perclos .509, closure .267, blink .103, yawn .121
#  (rounded below to 2 dp and renormalised to sum exactly 1.0).
#
#  NOTE: these AHP weights came out almost identical to the previous
#  by-feel weights (0.50/0.30/0.10/0.10) — which is reassuring, but now the
#  values are methodologically justified and the justification is auditable,
#  not asserted.  The pairwise JUDGEMENTS are still expert opinion; AHP makes
#  them explicit and checks them for internal consistency, it does not make
#  them "validated" in the clinical sense.
# ════════════════════════════════════════════════════════════

_WEIGHTS: dict[str, float] = {
    "perclos": 0.51,   # [AHP] validated bands; strongest fatigue biomarker
    "closure": 0.27,   # [AHP] acute event severity / leading edge
    "yawn":    0.12,   # [AHP] heuristic pending source [V3]
    "blink":   0.10,   # [AHP] ambiguous (bidirectional) → lowest weight
}
# Total: 1.00

assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"


# ════════════════════════════════════════════════════════════
#  SCORE → COMMAND  (band boundaries are [ENGINEERING])
#
#  EMERGENCY FLOOR (calibration safeguard):
#  Context multipliers (time of day, trip duration) are AMPLIFIERS, not
#  evidence.  Without this floor a merely-moderate physiological base could be
#  multiplied into the EMERGENCY band purely by context — e.g. PERCLOS=0.20
#  at 03:00 after 3 h reached EMERGENCY.  EMERGENCY should mean direct, strong
#  physiological evidence of (near) loss of consciousness, not "moderate signal
#  on a bad night".  So EMERGENCY from the score now REQUIRES the unamplified
#  base_score to reach EMERGENCY_MIN_BASE; otherwise the result is capped at
#  the top of the URGENT band.  Context can still escalate all the way to
#  URGENT ("pull over now") — it just cannot manufacture EMERGENCY alone.
#
#  Worked check with AHP weights:
#    PERCLOS 0.20 alone → 8.0×0.51 = 4.08 base  < 5.0 → capped at URGENT ✓
#    PERCLOS 0.25 alone → 10.0×0.51 = 5.10 base ≥ 5.0 → may reach EMERGENCY ✓
#    PERCLOS 0.20 + closure 1500ms → 4.08 + 2.70 = 6.78 base → convergence,
#                                    may reach EMERGENCY ✓ (two strong signals)
#  (Microsleep ≥2 s is handled earlier by the controller's reflex layer, so
#   most true EMERGENCIES come from there; this floor governs the rarer
#   score-driven EMERGENCY.)
# ════════════════════════════════════════════════════════════

EMERGENCY_MIN_BASE = 5.0    # base_score required before context can yield EMERGENCY
URGENT_CEILING     = 7.49   # top of the URGENT band (cap when floor not met)

# (upper_bound_inclusive, command)
_SCORE_BANDS: list[tuple[float, str]] = [
    (1.49, "NO_ACTION"),
    (2.99, "GENTLE_ALERT"),
    (5.49, "MODERATE_ALERT"),
    (7.49, "URGENT_ALERT"),
    (10.0, "EMERGENCY_ALERT"),
]


def _cognitive_state(
    command: str,
    dominant: str,
    is_talking: bool,
    adjusted: float,
) -> str:
    """
    Derive cognitive_state from the DOMINANT signal and severity, not from a
    hard-wired band label.  This is a coarse oracle label; the LLM
    produces the nuanced one when it is trusted.
    """
    if command == "NO_ACTION":
        return "Alert"
    if is_talking and adjusted >= 3.0:
        return "Cognitive Overload"

    high = command in ("URGENT_ALERT", "EMERGENCY_ALERT")
    if dominant == "blink":
        return "Highway Hypnosis"          # reduced arousal / autopilot
    if dominant in ("closure", "perclos"):
        return "Fighting Sleep" if high else "Active Fatigue"
    if dominant == "yawn":
        return "Active Fatigue"
    return "Unknown"


# ════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════

def compute_reference_decision(
    sensor:      dict,
    time_of_day: Optional[str] = None,
    trip_min:    Optional[int] = None,
) -> dict:
    """
    Compute a medically-grounded reference decision from the sensor payload.
    Never raises on bad input: malformed numbers are coerced/clamped and
    flagged via score_breakdown.input_valid.

    Returns
    -------
    {
        "command":         str,
        "severity":        "LOW" | "MEDIUM" | "HIGH",
        "cognitive_state": str,
        "adjusted_score":  float,
        "score_breakdown": dict,   # includes input_valid (bool) and repaired_fields
    }
    """
    if not isinstance(sensor, dict):
        sensor = {}

    if time_of_day is None:
        time_of_day = sensor.get("time_of_day")
    if trip_min is None:
        trip_min = sensor.get("trip_duration_min")

    is_talking = bool(sensor.get("Is_Talking", False))

    # ── Safe coercion + clamping of every numeric input ───────────────────
    perclos,   ok_p = _safe_value(sensor.get("PERCLOS"),              "PERCLOS")
    blinks,    ok_b = _safe_value(sensor.get("Blinks/min"),           "Blinks/min")
    closed_ms, ok_c = _safe_value(sensor.get("Eyes Closed Duration"), "Eyes Closed Duration")
    yawns_f,   ok_y = _safe_value(sensor.get("Yawns/min"),            "Yawns/min")

    repaired = [name for name, ok in [
        ("PERCLOS", ok_p), ("Blinks/min", ok_b),
        ("Eyes Closed Duration", ok_c), ("Yawns/min", ok_y),
    ] if not ok]
    input_valid = (len(repaired) == 0)

    # Discard yawn when talking (speech artefact; matches agent.py).
    yawns = 0 if is_talking else int(yawns_f)

    # ── Per-signal scores ──────────────────────────────────────────────────
    ps = _perclos_score(perclos)
    bs = _blink_score(blinks)
    cs = _closure_score(closed_ms)
    ys = _yawn_score(yawns)

    contributions = {
        "perclos": _WEIGHTS["perclos"] * ps,
        "closure": _WEIGHTS["closure"] * cs,
        "blink":   _WEIGHTS["blink"]   * bs,
        "yawn":    _WEIGHTS["yawn"]    * ys,
    }
    base_score = sum(contributions.values())
    dominant = max(contributions, key=contributions.get) if base_score > 0 else "none"

    # ── Context multipliers (jointly capped) ──────────────────────────────
    t_mult = _time_of_day_multiplier(time_of_day)
    d_mult = _trip_duration_multiplier(trip_min)
    combined_mult = min(t_mult * d_mult, _MAX_CONTEXT_MULTIPLIER)
    adjusted = min(base_score * combined_mult, 10.0)

    # ── EMERGENCY FLOOR ───────────────────────────────────────────────────
    # Context alone cannot manufacture an EMERGENCY: if the unamplified
    # physiological base is below EMERGENCY_MIN_BASE, cap the amplified score
    # at the top of URGENT.  (See _SCORE_BANDS header for the rationale.)
    emergency_floor_applied = False
    if base_score < EMERGENCY_MIN_BASE and adjusted > URGENT_CEILING:
        adjusted = URGENT_CEILING
        emergency_floor_applied = True

    # ── Score → command ───────────────────────────────────────────────────
    command = "EMERGENCY_ALERT"
    for upper, cmd in _SCORE_BANDS:
        if adjusted <= upper:
            command = cmd
            break

    cognitive_state = _cognitive_state(command, dominant, is_talking, adjusted)

    return {
        "command":         command,
        # severity now comes from the single source of truth in decision.py,
        # not a local copy — the oracle and the agent can never disagree.
        "severity":        dec.COMMAND_TO_SEVERITY[command],
        "cognitive_state": cognitive_state,
        "adjusted_score":  round(adjusted, 3),
        "score_breakdown": {
            "input_valid":     input_valid,
            "repaired_fields": repaired,
            "dominant_signal": dominant,
            "perclos_score":   round(ps, 2),
            "blink_score":     round(bs, 2),
            "closure_score":   round(cs, 2),
            "yawn_score":      round(ys, 2),
            "base_score":      round(base_score, 3),
            "time_mult":       t_mult,
            "duration_mult":   d_mult,
            "combined_mult":   round(combined_mult, 3),
            "emergency_floor_applied": emergency_floor_applied,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  COLD ADAPTER — wires the AHP reference engine into the runtime (tasks.py)
# ═════════════════════════════════════════════════════════════════════════════
"""
get_cold_decision(frame) is the entry point the runtime (tasks.py) calls for
the deterministic "cold" layer. It wraps compute_reference_decision() with two
additions:

  (1) HEAD-POSE CORROBORATION.
      The validated 4-criterion AHP matrix (perclos/closure/yawn/blink) is left
      UNTOUCHED. Head pose is treated as an INDEPENDENT calibrated posture
      signal: a sustained head-down nod (Head_Pitch_Down_Active) — the classic
      "nodding off" posture — escalates the fused command by exactly ONE tier,
      capped at URGENT (the non-reflex ceiling the rest of cold obeys). This
      keeps the AHP weights/CR defensible while incorporating the new signal.
      [TUNABLE] To fold pose INTO the AHP instead, re-derive the Saaty matrix
      for 6 criteria; this one-tier rule is the conservative default.

  (2) MICROSLEEP SAFETY NET  (defence in depth).
      In normal runtime T1's reflex latch handles >=2 s closure and T2 skips
      reflex-active frames, so this branch rarely fires — it exists so the cold
      layer is still safe if called directly on a microsleep frame.

Output shape matches what tasks.cold_baseline() and context_summary expect:
  command, reason, message, cognitive_state, failure_risk_10min, evidence{...}.
"""

_NON_REFLEX_CEILING = "URGENT_ALERT"

# ── Band hysteresis (same medicine as the EAR 0.20/0.23 thresholds) ──
# The score→command bands had no hysteresis, so a score hovering at a band
# edge (e.g. an integer blink count wobbling across 10/min moves the weighted
# sum ±0.3 around the 2.99 GENTLE/MODERATE edge) flapped the command and
# re-triggered audio. Rule: ESCALATIONS are instant, but a DOWNGRADE is
# accepted only after the score falls a margin BELOW the current band's lower
# edge. Safe direction only — the system is never slower to escalate.
# AURADRIVE_BAND_HYSTERESIS=0 restores the exact previous behavior.
_BAND_HYSTERESIS = float(os.getenv("AURADRIVE_BAND_HYSTERESIS", "0.25"))
_BAND_LOWER_EDGE = {          # lower edge of each graded band on the 0–10 score
    "NO_ACTION": 0.0, "GENTLE_ALERT": 1.50, "MODERATE_ALERT": 3.00,
    "URGENT_ALERT": 5.50, "EMERGENCY_ALERT": 7.50,
}
_last_graded_command: dict = {"command": None}


def _apply_band_hysteresis(command: str, adjusted: float) -> tuple[str, bool]:
    """Hold the previous (higher) command while the score sits within the
    hysteresis margin below its band's lower edge. Returns (command, held)."""
    prev = _last_graded_command["command"]
    held = False
    if (
        _BAND_HYSTERESIS > 0.0
        and prev is not None
        and dec.rank(command) < dec.rank(prev)
        and adjusted >= _BAND_LOWER_EDGE.get(prev, 10.0) - _BAND_HYSTERESIS
    ):
        command, held = prev, True
    _last_graded_command["command"] = command
    return command, held


def _cold_num(value, default: float = 0.0) -> float:
    """Coerce a frame field to a float, substituting a default when it is not one.

    The cold layer is the fallback the whole system relies on, so it must not be
    the layer that raises. Missing keys, non-numeric values and NaN produced by a
    failed landmark fit all resolve to a non-alarming default, keeping a single
    malformed frame from either crashing the engine or inventing an alert.

    Parameters
    ----------
    value : Any
        The raw field taken from the frame payload.
    default : float
        The non-alarming value substituted when coercion is not possible.

    Returns
    -------
    float
        The coerced measurement, or the default.
    """
    try:
        f = float(value)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def get_cold_decision(frame: dict) -> dict:
    """Graded-AHP deterministic decision for ONE frame (the cold layer).

    Input : frame dict from the sensor (metrics + time_of_day/trip_duration_min
            + Head_Pitch_Down_Active).
    Output: cold decision dict — command, cognitive_state, failure_risk_10min,
            reason, message, evidence{cold_index, active_features, breakdown}.
    """
    if not isinstance(frame, dict):
        frame = {}

    closed_ms = _cold_num(frame.get("Eyes Closed Duration"))

    # ── (2) Microsleep safety net — defence in depth ──
    if closed_ms >= dec.MICROSLEEP_THRESHOLD_MS:
        return {
            "command": "EMERGENCY_ALERT",
            "cognitive_state": "Microsleep",
            "failure_risk_10min": "HIGH",
            "reason": "cold_microsleep_safety_net",
            "message": f"Emergency: eyes closed for {int(closed_ms)} ms. Pull over safely immediately.",
            "evidence": {
                "cold_index": 10.0,
                "active_features": ["microsleep"],
                "closed_ms": round(closed_ms),
                "frame_id": frame.get("frame_id"),
            },
        }

    # ── Graded AHP fusion (reads time_of_day + trip_duration_min off the frame) ──
    ref = compute_reference_decision(frame)
    command = ref["command"]
    bd = ref.get("score_breakdown", {})

    # ── (1) Head-pose corroboration: sustained head-down escalates one tier ──
    head_down = bool(frame.get("Head_Pitch_Down_Active", False))
    head_roll = bool(frame.get("Head_Roll_Active", False))
    pose_escalated = False
    if head_down and command != "EMERGENCY_ALERT":
        bumped = dec.COMMANDS[min(dec.rank(command) + 1, dec.rank(_NON_REFLEX_CEILING))]
        if dec.rank(bumped) > dec.rank(command):
            command, pose_escalated = bumped, True

    # ── (3) Band hysteresis: hold the higher band while the score sits within
    #        the margin below its edge (stops boundary flapping; escalation stays instant) ──
    command, band_held = _apply_band_hysteresis(command, _cold_num(ref.get("adjusted_score")))

    # ── active features (consumed by the context-summary narrative) ──
    active = [s for s in ("perclos", "closure", "blink", "yawn")
              if _cold_num(bd.get(f"{s}_score")) > 0.0]
    if head_down:
        active.append("head_down")
    if head_roll:
        active.append("head_roll")

    failure_risk = ("HIGH" if command in ("URGENT_ALERT", "EMERGENCY_ALERT")
                    else "MEDIUM" if command == "MODERATE_ALERT" else "LOW")
    reason = "reference_ahp_fusion" if active else "reference_no_signal"
    message = ("" if command == "NO_ACTION"
               else "Deterministic fatigue signal detected. Stay attentive and follow the alert.")

    return {
        "command": command,
        "cognitive_state": ref.get("cognitive_state", "Unknown"),
        "failure_risk_10min": failure_risk,
        "reason": reason,
        "message": message,
        "evidence": {
            "cold_index": ref.get("adjusted_score"),
            "active_features": active,
            "pose_escalated": pose_escalated,
            "band_hold_applied": band_held,
            "score_breakdown": bd,
            "frame_id": frame.get("frame_id"),
        },
    }
