# sensor.py
"""
AuraDrive — Perception Layer (sensor.py)
=========================================
Responsibility: Extract ALL biometric signals from the camera frame and
expose them as a structured payload for the controller → agent pipeline.

This module owns ALL threshold-based decisions for the eye state machine
(EAR hysteresis, blink timing, PERCLOS window). These are deterministic
rules that Python should own — NOT the LLM.

The LLM (agent.py) receives the computed metrics and reasons about their
BEHAVIORAL MEANING across time. It never re-derives state from raw pixels.

Signals produced per frame:
  Eye:   EAR, Driver_State, Eyes Closed Duration, Blinks/min, PERCLOS
  Mouth: MAR, Mouth_State, Yawns/min, Is_Talking
"""

import cv2
import os
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

from pose import HeadPoseTracker



class EyeBlinkSensor:
    """
    Computes fatigue-relevant biometric metrics from a single camera frame.

    Design principles:
    - ear_fast  → used ONLY for state transitions and blink timing (responsive)
    - ear_slow  → used ONLY for SLEEPING confirmation (noise-resistant)
    - PERCLOS   → always includes the current ongoing closure (no lag)
    - Mouth     → MAR + temporal classification into NORMAL / MOUTH_OPEN / YAWNING
    - Talking   → sustained high MAR across multiple frames signals speech
    """

    # ── MediaPipe landmark indices ──
    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth landmarks — simple, well-validated MAR using only true lip points.
    #   13   = upper lip, inner edge centre  (top of mouth opening)
    #   14   = lower lip, inner edge centre  (bottom of mouth opening)
    #   61   = left outer mouth corner
    #   291  = right outer mouth corner
    # MAR = dist(13,14) / dist(61,291)  →  closed ~0.05, wide yawn ~0.7
    # Always in [0,1] for a real face; values above 1 mean landmark failure.
    MAR_LIP_TOP    = 13
    MAR_LIP_BOTTOM = 14
    MAR_CORNER_L   = 61
    MAR_CORNER_R   = 291

    # ── Eye state machine thresholds ──
    EAR_CLOSE_THRESH = 0.20   # OPEN → CLOSED  (Python owns this, not the LLM)
    EAR_OPEN_THRESH  = 0.23   # CLOSED → OPEN  (hysteresis gap prevents flicker)

    # ── Blink timing (ms) ──
    MIN_BLINK_MS = 60     # shorter than this = noise / micro-twitch
    MAX_BLINK_MS = 350    # longer than this = not a blink, partial closure

    # ── Drowsiness timing ──
    SLEEP_CONFIRM_MS = 1_000   # closure ≥ 1s + ear_slow confirms drowsiness
                                # at 100 km/h the car covers 27.8 m per second

    # ── PERCLOS window ──
    PERCLOS_WINDOW_MS = 60_000  # standard 60-second rolling window (NHTSA)

    # ── Mouth / yawn thresholds (calibrated for the simple MAR above) ──
    MAR_OPEN_THRESH  = 0.30   # below → NORMAL, above → mouth opened
    MAR_YAWN_THRESH  = 0.50   # above (sustained) → wide enough for a yawn
    YAWN_MIN_MS      = 1_500  # minimum open duration to count as a yawn
    YAWN_MAX_MS      = 6_000  # beyond this → probably just mouth open, not yawning
    YAWN_WINDOW_MS   = 60_000 # rolling window for Yawns/min

    # ── Eye-narrowing co-signal (distinguishes yawning from talking) ──
    # When a person yawns, eyes partially close (EAR drops). When a person
    # talks, eyes stay at normal openness. We use the minimum EAR observed
    # during a mouth-open period to tell them apart. A yawn requires both
    # a wide mouth AND eye narrowing; talking requires MAR activity AND
    # stable eyes.
    YAWN_EYE_NARROW_THRESH = 0.22  # min EAR during open period below this → yawn

    # ── Talking detection ──
    # Speech produces rapid, oscillating MAR fluctuation.
    # We detect it by measuring MAR variance over a short window:
    # high variance + moderate-to-high mean MAR = talking.
    TALKING_WINDOW_FRAMES = 20   # ~0.67 s at 30 FPS
    TALKING_MAR_MEAN_MIN  = 0.25 # mean MAR must be elevated (mouth is moving)
    TALKING_MAR_VAR_MIN   = 0.003 # variance threshold distinguishes speech from static open

    # ── Perception v2 (experimental, default OFF) ──
    # AURADRIVE_PERCEPTION_V2=1 enables behavior-changing math upgrades:
    #   * 3D landmark distances for EAR/MAR (pose-invariant: head pitch no
    #     longer foreshortens the vertical distances and fakes eye closure)
    #   * per-driver EAR threshold calibration (median of the first stable
    #     open-eye frames; fixed 0.20/0.23 becomes a fraction of the driver's
    #     own baseline)
    #   * median instead of mean for ear_slow (outlier-immune smoothing)
    # Downstream thresholds are calibrated for the default (v1) signal path,
    # so v2 stays opt-in until it is re-validated against recorded runs.
    EAR_CALIB_FRAMES     = 90     # ~3 s of clearly-open eyes
    EAR_CALIB_MIN_OPEN   = 0.24   # sample only unambiguous open-eye frames
    EAR_CLOSE_FRACTION   = 0.72   # close threshold as fraction of baseline
    EAR_OPEN_FRACTION    = 0.80   # open  threshold as fraction of baseline

    def __init__(self, debug: bool = True) -> None:
        """Build the perception layer and every buffer its state machines need.

        The sensor is stateful by design: eye and mouth classification, PERCLOS,
        blink rate and yawn counting are all judgements about time rather than
        about a single frame, so the rolling buffers created here are what make
        those metrics possible. The head pose tracker is constructed alongside
        them so posture is calibrated in step with the rest of perception.

        Parameters
        ----------
        debug : bool
            Whether to draw the telemetry overlay and EAR timeline on the
            frame, which is what the operator sees during a live session.

        Returns
        -------
        None
            The constructor only prepares perception state.
        """
        self.debug = debug
        self.perception_v2 = os.getenv("AURADRIVE_PERCEPTION_V2", "0") == "1"

        # ── Eye state ──
        self.eye_state        = "OPEN"
        self.eye_closed_start = None   # ms timestamp when closure began

        # Instance thresholds: the validated defaults (0.20/0.23), overridable
        # per-setup via env (AURADRIVE_EAR_CLOSE / AURADRIVE_EAR_OPEN) — e.g. a
        # low laptop camera reads EAR shallow and benefits from 0.18/0.21.
        # Per-driver calibration (v2) replaces them at runtime when enabled.
        self.ear_close_thresh = float(os.getenv("AURADRIVE_EAR_CLOSE", self.EAR_CLOSE_THRESH))
        self.ear_open_thresh  = float(os.getenv("AURADRIVE_EAR_OPEN",  self.EAR_OPEN_THRESH))
        self._ear_calib_samples: deque = deque(maxlen=self.EAR_CALIB_FRAMES)
        self.ear_calibrated   = False

        # ── Additive metrics (always on — never feed the deterministic score) ──
        # Lid reopening velocity (Johns' AVR family): slowed lid rise is a
        # validated early drowsiness marker. Computed per closure from the
        # minimum-EAR point to the reopen crossing.
        self._closure_ear_min:   float | None = None
        self._closure_ear_min_t: float | None = None
        self.reopen_velocities:  deque = deque(maxlen=30)
        # "Covered yawn": sustained eye-narrowing without closure while the
        # mouth stays shut — the hand-over-mouth yawn signature. Context-only.
        self._narrow_start: float | None = None
        self.covered_yawn_times: deque = deque(maxlen=60)

        # Frame dimensions for 3D distance computation (set per frame).
        self._fw, self._fh = 1, 1

        # ── Blink history (bounded deque — auto-evicts oldest) ──
        self.blink_times: deque = deque(maxlen=300)

        # ── EAR smoothing for sleep confirmation only ──
        self.ear_window: deque = deque(maxlen=15)

        # ── PERCLOS: completed closure intervals ──
        self.closed_intervals: deque = deque()  # (start_ms, end_ms) pairs

        # ── EAR history for the debug overlay plot ──
        self.ear_history:     deque = deque(maxlen=300)
        self.ear_timestamps:  deque = deque(maxlen=300)

        # ── Mouth state ──
        self.mouth_open_start    = None   # ms timestamp when mouth opened
        self.mouth_open_ear_min  = None   # min EAR observed during current open period
        self.current_mouth_state = "NORMAL"

        # ── Yawn history ──
        self.yawn_times: deque = deque(maxlen=60)

        # ── MAR history for talking detection ──
        self.mar_history: deque = deque(maxlen=self.TALKING_WINDOW_FRAMES)

        # ── Controller interface ──
        # Set to a full payload dict when risk is detected; None otherwise.
        # The controller polls this after each frame.
        self.last_risk_payload = None

        # ── MediaPipe ──
        self.mp_face   = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawer = mp.solutions.drawing_utils
        # Added posture tracker. The original mesh and overlay below remain unchanged.
        self.head_pose_tracker = HeadPoseTracker()

    # ══════════════════════════════════════════
    #  GEOMETRY HELPERS
    # ══════════════════════════════════════════

    def _dist(self, lm, a: int, b: int) -> float:
        """Landmark distance. v1: 2D normalized coords (legacy — calibrated).
        v2: 3D pixel-space distance (x·w, y·h, z·w) — MediaPipe scales z like
        x, so this yields a pose-invariant ratio: pitching the head down no
        longer foreshortens vertical distances into a fake eye closure."""
        if self.perception_v2:
            return math.dist(
                [lm[a].x * self._fw, lm[a].y * self._fh, lm[a].z * self._fw],
                [lm[b].x * self._fw, lm[b].y * self._fh, lm[b].z * self._fw],
            )
        return math.dist(
            [lm[a].x, lm[a].y],
            [lm[b].x, lm[b].y],
        )

    def _compute_EAR(self, lm, indices) -> float:
        """Eye Aspect Ratio — standard Soukupová & Čech (2016) formula."""
        p1, p2, p3, p4, p5, p6 = indices
        vertical   = self._dist(lm, p2, p6) + self._dist(lm, p3, p5)
        horizontal = 2.0 * self._dist(lm, p1, p4)
        return vertical / horizontal

    def _compute_MAR(self, lm) -> float:
        """
        Mouth Aspect Ratio — inner-lip vertical opening / outer mouth width.
        Uses only landmarks that are demonstrably on the lips themselves:
          numerator   = vertical distance from upper-lip inner edge to lower-lip inner edge
          denominator = horizontal distance between outer mouth corners
        Output range for a normal face: ~0.05 (closed) to ~0.7 (wide yawn).
        Values above 1.0 indicate a landmark detection failure, not a real mouth.
        """
        vertical   = self._dist(lm, self.MAR_LIP_TOP,  self.MAR_LIP_BOTTOM)
        horizontal = self._dist(lm, self.MAR_CORNER_L, self.MAR_CORNER_R)
        if horizontal < 1e-6:
            return 0.0
        return vertical / horizontal

    # ══════════════════════════════════════════
    #  PERCEPTION V2 + ADDITIVE-METRIC HELPERS
    # ══════════════════════════════════════════

    def _update_ear_calibration(self, ear_fast: float) -> None:
        """(v2) Learn the driver's open-eye EAR baseline, then derive the
        hysteresis thresholds as fractions of it. Runs once per session."""
        if self.ear_calibrated or not self.perception_v2:
            return
        if ear_fast >= self.EAR_CALIB_MIN_OPEN:
            self._ear_calib_samples.append(ear_fast)
        if len(self._ear_calib_samples) >= self.EAR_CALIB_FRAMES:
            baseline = float(np.median(self._ear_calib_samples))
            close = min(max(baseline * self.EAR_CLOSE_FRACTION, 0.14), 0.26)
            self.ear_close_thresh = close
            self.ear_open_thresh  = min(max(baseline * self.EAR_OPEN_FRACTION, close + 0.02), 0.30)
            self.ear_calibrated   = True

    def _track_reopen_velocity(self, ear_fast: float, now_ms: float, reopened: bool) -> None:
        """Additive metric: eyelid reopening speed (EAR units / second) from
        the minimum-EAR point of a closure to its reopen crossing. Slowed lid
        rise is a validated early drowsiness marker (Johns' AVR family)."""
        if self.eye_state == "CLOSED" and not reopened:
            if self._closure_ear_min is None or ear_fast < self._closure_ear_min:
                self._closure_ear_min, self._closure_ear_min_t = ear_fast, now_ms
        elif reopened:
            if self._closure_ear_min is not None and now_ms > self._closure_ear_min_t:
                dt_s = (now_ms - self._closure_ear_min_t) / 1000.0
                self.reopen_velocities.append((ear_fast - self._closure_ear_min) / dt_s)
            self._closure_ear_min = self._closure_ear_min_t = None

    def _update_covered_yawn(self, ear_fast: float, mar: float, now_ms: float) -> None:
        """Additive metric: sustained eye-narrowing WITHOUT closure while the
        mouth stays shut — the hand-over-mouth yawn signature that the
        MAR-gated yawn detector cannot see. Feeds LLM context only."""
        narrowing = (
            self.eye_state == "OPEN"
            and self.ear_close_thresh <= ear_fast < self.YAWN_EYE_NARROW_THRESH
            and mar < self.MAR_OPEN_THRESH
        )
        if narrowing:
            if self._narrow_start is None:
                self._narrow_start = now_ms
        else:
            if self._narrow_start is not None:
                duration = now_ms - self._narrow_start
                if self.YAWN_MIN_MS <= duration <= self.YAWN_MAX_MS:
                    self.covered_yawn_times.append(now_ms)
                self._narrow_start = None
        cutoff = now_ms - self.YAWN_WINDOW_MS
        while self.covered_yawn_times and self.covered_yawn_times[0] < cutoff:
            self.covered_yawn_times.popleft()

    def _compute_head_pose(self, lm, width: int, height: int) -> tuple[float, float, bool]:
        """Estimate Euler pitch and roll from stable FaceMesh landmarks."""
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
        ], dtype=np.float64)
        indices = [1, 152, 33, 263, 61, 291]
        image_points = np.array([(lm[i].x * width, lm[i].y * height) for i in indices], dtype=np.float64)
        focal = float(width)
        camera_matrix = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        try:
            ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return 0.0, 0.0, False
            matrix, _ = cv2.Rodrigues(rotation)
            angles, *_ = cv2.RQDecomp3x3(matrix)
            return float(angles[0]), float(angles[2]), True
        except cv2.error:
            return 0.0, 0.0, False

    # ══════════════════════════════════════════
    #  MAIN PER-FRAME PROCESSING
    # ══════════════════════════════════════════

    def process_frame(self, frame) -> dict:
        """
        Process one camera frame and return a metrics dict.
        Also updates self.last_risk_payload when risk is detected.
        """
        self.last_risk_payload = None
        now_ms = time.time() * 1000
        h, w, _ = frame.shape

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            if self.debug:
                cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return self._build_output(
                ear=None, driver_state="EYES_OPEN",
                closed_ms=0, blinks_per_min=0, perclos=0.0,
                mar=0.0, mouth_state="NORMAL", yawns_per_min=0,
                is_talking=False, no_face=True,
            )

        lm = results.multi_face_landmarks[0].landmark

        # ── EAR ──────────────────────────────────────────────────────────
        self._fw, self._fh = w, h   # 3D distances (v2) need pixel scaling
        ear_left  = self._compute_EAR(lm, self.LEFT_EYE)
        ear_right = self._compute_EAR(lm, self.RIGHT_EYE)
        ear_fast  = (ear_left + ear_right) / 2.0

        self.ear_window.append(ear_fast)
        # v2 uses the median: one landmark glitch cannot drag the estimate.
        ear_slow = float(np.median(self.ear_window) if self.perception_v2
                         else np.mean(self.ear_window))

        self.ear_history.append(ear_fast)
        self.ear_timestamps.append(now_ms)
        self._update_ear_calibration(ear_fast)

        # ── EYE STATE MACHINE (ear_fast only — responsive) ───────────────
        closed_ms = 0.0
        reopened  = False

        if self.eye_state == "OPEN":
            if ear_fast < self.ear_close_thresh:
                self.eye_state        = "CLOSED"
                self.eye_closed_start = now_ms

        elif self.eye_state == "CLOSED":
            closed_ms = now_ms - self.eye_closed_start

            if ear_fast > self.ear_open_thresh:
                # Closure ended — record it for PERCLOS
                self.closed_intervals.append((self.eye_closed_start, now_ms))

                # Count as a blink only if duration is in the voluntary range
                if self.MIN_BLINK_MS <= closed_ms <= self.MAX_BLINK_MS:
                    self.blink_times.append(now_ms)

                self.eye_state        = "OPEN"
                self.eye_closed_start = None
                closed_ms             = 0.0
                reopened              = True

        self._track_reopen_velocity(ear_fast, now_ms, reopened)

        # ── DRIVER STATE ─────────────────────────────────────────────────
        # "SLEEPING" is a useful internal label but the agent only accepts
        # EYES_OPEN / EYES_CLOSED. We map SLEEPING → EYES_CLOSED so the
        # agent's hard_safety_check fires correctly on long closures.
        sleeping = (
            self.eye_state == "CLOSED"
            and closed_ms >= self.SLEEP_CONFIRM_MS
            and ear_slow < self.ear_close_thresh
        )
        driver_state = "EYES_CLOSED" if self.eye_state == "CLOSED" else "EYES_OPEN"
        # Keep internal label for debug display
        display_state = "SLEEPING" if sleeping else driver_state

        # ── BLINK RATE ───────────────────────────────────────────────────
        self._evict_old_blinks(now_ms)
        blinks_per_min = len(self.blink_times)

        # ── PERCLOS ──────────────────────────────────────────────────────
        perclos = self._compute_perclos(now_ms)

        # ── MAR + MOUTH STATE ────────────────────────────────────────────
        mar = self._compute_MAR(lm)
        self.mar_history.append(mar)

        mouth_state    = self._classify_mouth_state(mar, ear_fast, now_ms)
        yawns_per_min  = self._compute_yawns_per_min(now_ms)
        is_talking     = self._detect_talking()
        self._update_covered_yawn(ear_fast, mar, now_ms)

        # Head pose is added without changing the original eye/mouth calculations
        # or the face-mesh drawing below. Calibration samples only stable, open-eye frames.
        head_pitch, head_roll, pose_ok = self._compute_head_pose(lm, w, h)
        pose_state = self.head_pose_tracker.update(
            head_pitch, head_roll, now_ms,
            usable=pose_ok and driver_state == "EYES_OPEN" and not is_talking,
        )

        # ── DEBUG OVERLAY ────────────────────────────────────────────────
        if self.debug:
            self._draw_debug_overlay(
                frame, w, h,
                ear_fast, ear_slow, display_state,
                closed_ms, blinks_per_min, perclos,
                mar, mouth_state, yawns_per_min, is_talking,
                results,
            )

        # ── BUILD OUTPUT ─────────────────────────────────────────────────
        out = self._build_output(
            ear=ear_fast,
            driver_state=driver_state,
            closed_ms=closed_ms,
            blinks_per_min=blinks_per_min,
            perclos=perclos,
            mar=mar,
            mouth_state=mouth_state,
            yawns_per_min=yawns_per_min,
            is_talking=is_talking,
            pose_state=pose_state,
            head_pitch=head_pitch,
            head_roll=head_roll,
        )

        # Expose to controller when any risk signal is active.
        # Thresholds are intentionally LOW so the LLM agent is engaged EARLY —
        # in the subtle / gradual regime where it adds value over the
        # deterministic reference. The agent returns NO_ACTION on benign
        # snapshots, so a wide gate engages it without raising alerts.
        #   PERCLOS  > 0.08   : Grace (2001) moderate floor               (was 0.15)
        #   yawns   >= 1      : a single yawn is worth a contextual look  (was >= 2)
        #   blinks  <10 / >30 : leading-edge blink anomalies              (was <8 / >45)
        #   head-down         : sustained nod-off posture (head-pose signal)
        if (
            driver_state == "EYES_CLOSED"
            or perclos > 0.08
            or yawns_per_min >= 1
            or blinks_per_min < 10
            or blinks_per_min > 30
            or bool(out.get("Head_Pitch_Down_Active", False))
        ):
            self.last_risk_payload = out

        return out

    # ══════════════════════════════════════════
    #  MOUTH STATE MACHINE
    # ══════════════════════════════════════════

    def _classify_mouth_state(self, mar: float, ear_fast: float, now_ms: float) -> str:
        """
        Classifies mouth state using BOTH the mouth signal (MAR) and the eye
        co-signal (EAR). Physiological insight: yawning produces partial eye
        closure; talking does not. We track the MINIMUM EAR observed during
        each mouth-open period and use it to disambiguate.

        NORMAL     — mouth closed or minimally open
        MOUTH_OPEN — mouth is open but we don't yet have yawn evidence
        YAWNING    — wide mouth opening for ≥ YAWN_MIN_MS AND eyes narrowed
                     to below YAWN_EYE_NARROW_THRESH at some point during it
        """
        # ── Mouth closed: close out any open period, possibly count a yawn ──
        if mar < self.MAR_OPEN_THRESH:
            if self.mouth_open_start is not None:
                duration       = now_ms - self.mouth_open_start
                duration_ok    = self.YAWN_MIN_MS <= duration <= self.YAWN_MAX_MS
                eyes_narrowed  = (
                    self.mouth_open_ear_min is not None
                    and self.mouth_open_ear_min < self.YAWN_EYE_NARROW_THRESH
                )
                # Only count as a yawn if BOTH duration AND eye-narrowing fired.
                # Long mouth-open with stable eyes is likely talking — not a yawn.
                if duration_ok and eyes_narrowed:
                    self.yawn_times.append(now_ms)
            self.mouth_open_start    = None
            self.mouth_open_ear_min  = None
            self.current_mouth_state = "NORMAL"
            return "NORMAL"

        # ── Mouth open: start a new period or extend the existing one ──
        if self.mouth_open_start is None:
            self.mouth_open_start   = now_ms
            self.mouth_open_ear_min = ear_fast
        else:
            # Track the minimum EAR observed during this open period
            self.mouth_open_ear_min = min(self.mouth_open_ear_min, ear_fast)

        open_duration = now_ms - self.mouth_open_start

        # YAWNING in-progress requires wide MAR, sustained duration, AND
        # eye narrowing — all three must agree before we label it a yawn.
        if (
            mar >= self.MAR_YAWN_THRESH
            and open_duration >= self.YAWN_MIN_MS
            and self.mouth_open_ear_min is not None
            and self.mouth_open_ear_min < self.YAWN_EYE_NARROW_THRESH
        ):
            self.current_mouth_state = "YAWNING"
            return "YAWNING"

        self.current_mouth_state = "MOUTH_OPEN"
        return "MOUTH_OPEN"

    def _compute_yawns_per_min(self, now_ms: float) -> int:
        """Counts confirmed yawns in the last 60 seconds."""
        cutoff = now_ms - self.YAWN_WINDOW_MS
        while self.yawn_times and self.yawn_times[0] < cutoff:
            self.yawn_times.popleft()
        return len(self.yawn_times)

    # ══════════════════════════════════════════
    #  TALKING DETECTION
    # ══════════════════════════════════════════

    def _detect_talking(self) -> bool:
        """
        Talking: rapid MAR oscillation AND eyes stayed near normal openness.

        Both yawning and talking elevate MAR, so MAR alone is ambiguous. Eye
        stability is the disambiguator: talking does not narrow the eyes,
        yawning does. We require min EAR over the same window to be at or
        above YAWN_EYE_NARROW_THRESH — i.e. no eye narrowing during the
        MAR-active period.
        """
        if len(self.mar_history) < self.TALKING_WINDOW_FRAMES // 2:
            return False
        if len(self.ear_history) < self.TALKING_WINDOW_FRAMES // 2:
            return False

        # MAR signal: high variance + elevated mean = mouth is moving
        mar_arr  = np.array(self.mar_history)
        mean_mar = float(np.mean(mar_arr))
        var_mar  = float(np.var(mar_arr))

        mar_active = (
            mean_mar >= self.TALKING_MAR_MEAN_MIN
            and var_mar >= self.TALKING_MAR_VAR_MIN
        )
        if not mar_active:
            return False

        # EAR co-signal: eyes stayed open throughout the same window.
        # If min EAR dropped below the yawn threshold during MAR activity,
        # that's likely yawning, not talking.
        recent_ear      = list(self.ear_history)[-self.TALKING_WINDOW_FRAMES:]
        min_ear_window  = float(np.min(recent_ear))
        eyes_stable     = min_ear_window >= self.YAWN_EYE_NARROW_THRESH

        return eyes_stable

    # ══════════════════════════════════════════
    #  PERCLOS
    # ══════════════════════════════════════════

    def _compute_perclos(self, now_ms: float) -> float:
        """
        PERCLOS = fraction of the last 60 seconds the eyes were closed.
        Includes the current ongoing closure so there is no lag.
        """
        cutoff     = now_ms - self.PERCLOS_WINDOW_MS
        closed_ms  = 0.0

        # Drop intervals that ended before the window
        while self.closed_intervals and self.closed_intervals[0][1] < cutoff:
            self.closed_intervals.popleft()

        # Sum completed intervals within the window
        for start, end in self.closed_intervals:
            closed_ms += max(0.0, min(end, now_ms) - max(start, cutoff))

        # Add current ongoing closure
        if self.eye_state == "CLOSED" and self.eye_closed_start is not None:
            closed_ms += max(0.0, now_ms - max(self.eye_closed_start, cutoff))

        return closed_ms / self.PERCLOS_WINDOW_MS

    def _evict_old_blinks(self, now_ms: float) -> None:
        """Drop blink timestamps that have aged out of the rolling minute.

        Blink rate is reported per minute, so the buffer must forget anything
        older than that or the rate would only ever climb across a session.
        Eviction happens at read time rather than on a timer, which keeps the
        metric correct without a second scheduled task.

        Parameters
        ----------
        now_ms : float
            Current frame time, defining the trailing edge of the window.

        Returns
        -------
        None
            The function performs an action without returning a value.
        """
        cutoff = now_ms - 60_000
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()

    # ══════════════════════════════════════════
    #  OUTPUT BUILDER
    # ══════════════════════════════════════════

    def _build_output(
        self,
        ear:            float | None,
        driver_state:   str,
        closed_ms:      float,
        blinks_per_min: int,
        perclos:        float,
        mar:            float,
        mouth_state:    str,
        yawns_per_min:  int,
        is_talking:     bool,
        no_face:        bool = False,
        pose_state:     dict | None = None,
        head_pitch:     float = 0.0,
        head_roll:      float = 0.0,
    ) -> dict:
        """
        Builds the canonical output dict.
        Field names match exactly what agent.py's REQUIRED_SENSOR_FIELDS expects.
        """
        return {
            # ── Eye metrics ──
            "Driver_State":         driver_state,
            "EAR":                  round(ear, 4) if ear is not None else None,
            "Eyes Closed Duration": int(closed_ms),
            "Blinks/min":           int(blinks_per_min),
            "PERCLOS":              round(perclos, 4),
            # ── Mouth metrics ──
            # MAR is clamped to [0.0, 1.0] before output.
            # Wide yawns can produce MAR > 1.0 via valid MediaPipe geometry
            # (vertical lip gap exceeds horizontal mouth width), but the agent
            # schema rejects anything above 1.0 as out-of-range.
            # Clamping preserves the "mouth very wide open" signal at the
            # maximum representable value instead of discarding the frame.
            "MAR":                  round(min(mar, 1.0), 4),
            "Mouth_State":          mouth_state,
            "Yawns/min":            int(yawns_per_min),
            "Is_Talking":           bool(is_talking),
            # ── Meta ──
            "no_face":              no_face,
            # ── Additive perception metrics (context only, never scored) ──
            "Lid_Reopen_Velocity_Avg": round(float(np.mean(self.reopen_velocities)), 3)
                                       if self.reopen_velocities else None,
            "Possible_Covered_Yawns/min": len(self.covered_yawn_times),
            "EAR_Thresholds": (round(self.ear_close_thresh, 3), round(self.ear_open_thresh, 3)),
            # ── Head pose additions; original fields above remain unchanged ──
            "Head_Pitch":           round(float(head_pitch), 2),
            "Head_Roll":            round(float(head_roll), 2),
            **(pose_state or {
                "Head_Pitch_Baseline": 0.0, "Head_Roll_Baseline": 0.0,
                "Head_Pitch_Delta": 0.0, "Head_Roll_Delta": 0.0,
                "Head_Pitch_Down_Delta": 0.0, "Head_Pitch_Down_Active": False,
                "Head_Roll_Active": False, "Head_Pose_Calibrated": False,
                "Head_Pitch_Down_Sign": 1,
            }),
        }

    # ══════════════════════════════════════════
    #  DEBUG OVERLAY
    # ══════════════════════════════════════════

    def _draw_debug_overlay(
        self, frame, w, h,
        ear_fast, ear_slow, display_state,
        closed_ms, blinks_per_min, perclos,
        mar, mouth_state, yawns_per_min, is_talking,
        results,
    ) -> None:
        """Draws all telemetry and the EAR plot onto the frame."""

        # Face mesh contours
        self.drawer.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            self.mp_face.FACEMESH_CONTOURS,
        )

        # ── Left column: eye metrics ──
        state_color = (0, 0, 255) if display_state in ("EYES_CLOSED", "SLEEPING") else (0, 255, 0)
        rows = [
            (f"EAR fast: {ear_fast:.3f}",           (0, 255, 255)),
            (f"EAR slow: {ear_slow:.3f}",            (0, 200, 200)),
            (f"State: {display_state}",              state_color),
            (f"Closed ms: {int(closed_ms)}",         (255, 100, 100)),
            (f"Blinks/min: {blinks_per_min}",        (255, 255, 0)),
            (f"PERCLOS(60s): {perclos:.3f}",         (255, 100, 100)),
        ]
        for i, (text, color) in enumerate(rows):
            cv2.putText(frame, text, (20, 40 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ── Left column continued: mouth metrics ──
        mouth_color = (0, 165, 255) if mouth_state == "YAWNING" else (200, 200, 200)
        talk_color  = (0, 255, 0)   if is_talking               else (100, 100, 100)
        mouth_rows = [
            (f"MAR: {mar:.3f}",                     (200, 180, 255)),
            (f"Mouth: {mouth_state}",               mouth_color),
            (f"Yawns/min: {yawns_per_min}",         (200, 180, 255)),
            (f"Talking: {'YES' if is_talking else 'no'}", talk_color),
        ]
        for i, (text, color) in enumerate(mouth_rows):
            cv2.putText(frame, text, (20, 260 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # ── SLEEPING alert border ──
        if display_state == "SLEEPING":
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 15)
            cv2.putText(frame, "SLEEPING!", (w - 360, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 6)

        # ── EAR timeline plot (top-right) ──
        self._draw_ear_plot(frame, w, h)

    def _draw_ear_plot(self, frame, w: int, h: int) -> None:
        """Draws a real-time EAR timeline with threshold reference lines."""
        if len(self.ear_history) < 2:
            return

        pw, ph = 400, 150
        px, py = w - pw - 20, 20

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (100, 100, 100), 2)
        cv2.putText(frame, "EAR Timeline", (px + 10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ear_min, ear_max = 0.1, 0.4
        ear_range = ear_max - ear_min

        def _to_y(val):
            """Map an EAR value to a pixel row on the telemetry plot.

            Rendering helper for the on-screen EAR timeline. Values are clamped
            to the plotted range so an outlier cannot draw outside the panel.

            Parameters
            ----------
            val : float
                The EAR value to place on the vertical axis.

            Returns
            -------
            int
                The pixel row for that value inside the plot area.
            """
            clamped = max(ear_min, min(ear_max, val))
            return py + ph - int(((clamped - ear_min) / ear_range) * (ph - 30)) - 15

        # Threshold reference lines
        cy = _to_y(self.EAR_CLOSE_THRESH)
        oy = _to_y(self.EAR_OPEN_THRESH)
        cv2.line(frame, (px, cy), (px + pw, cy), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (px, oy), (px + pw, oy), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Close:{self.EAR_CLOSE_THRESH:.2f}",
                    (px + 5, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
        cv2.putText(frame, f"Open:{self.EAR_OPEN_THRESH:.2f}",
                    (px + 5, oy + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1)

        # EAR signal line
        ear_list = list(self.ear_history)
        pts = [
            (px + int((i / len(ear_list)) * pw), _to_y(v))
            for i, v in enumerate(ear_list)
        ]
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
        if pts:
            cv2.circle(frame, pts[-1], 4, (0, 255, 255), -1)