"""Calibrated head posture tracking, the third fatigue channel.

Eye and mouth metrics miss the classic nodding off, where a driver's head
drops while the eyes are still open. This module supplies that signal by
recovering pitch and roll and judging them against a baseline learned for the
current driver, because raw angles carry no meaning when drivers sit
differently and cameras mount at different heights.

Two properties keep the channel resistant to false alarms. Deviations are
measured from a per-driver baseline rather than from absolute zero, and a
deviation must persist before it counts, which is what separates a genuine
head drop from a mirror check or a glance at the dashboard. The output is a
posture signal that corroborates other evidence, never a diagnosis on its own.
"""
from __future__ import annotations
import os
from statistics import median
from typing import Any, Dict, Optional

class HeadPoseTracker:
    CALIBRATION_FRAMES = 45
    # Head-pose fatigue thresholds from Wei, Chi & Chen (2023), "A Multi-Feature
    # Fusion and Situation Awareness-Based Method for Fatigue Driving Level
    # Determination," Electronics 12(13):2884, sec. 2.4: head down at |Pitch| >= 20deg,
    # head tilt at |Roll| >= 15.4deg (yaw is ignored — drowsy drivers show minimal yaw).
    # Measured here as deviation from the calibrated neutral baseline.
    PITCH_DOWN_DEVIATION_DEG = 20.0
    ROLL_DEVIATION_DEG = 15.4
    # 500 ms flagged every keyboard/dashboard glance as a "sustained" nod and
    # stacked a +1 tier onto mild scores. A genuine nod-off posture persists:
    # 1.5 s ignores glances yet still fires well before the 2 s microsleep
    # reflex. [ENGINEERING], env-tunable per setup.
    PERSIST_MS = float(os.getenv("AURADRIVE_PITCH_PERSIST_MS", "1500"))

    def __init__(self, pitch_down_sign: Optional[int] = None) -> None:
        """Initialise the calibration buffers and the pitch direction convention.

        Which sign of pitch means "head down" depends on how the camera is
        mounted, so it is configurable rather than assumed. The tracker starts
        uncalibrated and stays inert until it has seen enough stable frames to
        learn the driver's neutral posture.

        Parameters
        ----------
        pitch_down_sign : int or None
            Sign convention for a downward nod on this installation; taken from
            the environment when not supplied.

        Returns
        -------
        None
            The constructor only prepares internal state.
        """
        raw = pitch_down_sign if pitch_down_sign is not None else int(os.getenv("AURADRIVE_PITCH_DOWN_SIGN", "1"))
        self.pitch_down_sign = 1 if raw >= 0 else -1
        self._pitch_samples: list[float] = []
        self._roll_samples: list[float] = []
        self._pitch_baseline: Optional[float] = None
        self._roll_baseline: Optional[float] = None
        self._pitch_candidate_since: Optional[float] = None
        self._roll_candidate_since: Optional[float] = None

    @property
    def calibrated(self) -> bool:
        """Report whether a neutral posture baseline has been established.

        Until both baselines exist the head channel reports nothing active, so
        this flag is what prevents an uncalibrated session from contributing
        head-pose evidence to a fatigue decision.

        Returns
        -------
        bool
            True once both the pitch and roll baselines have been learned.
        """
        return self._pitch_baseline is not None and self._roll_baseline is not None

    @staticmethod
    def angular_delta(current: float, baseline: float) -> float:
        """Measure the signed angular difference, wrapped to the shortest arc.

        Euler angles recovered from the face mesh can sit either side of the
        plus or minus 180 degree discontinuity, where a naive subtraction would
        report a large deviation for a head that barely moved. Wrapping the
        result keeps a small physical movement small, which is what stops the
        discontinuity from manufacturing a spurious nod-off.

        Parameters
        ----------
        current : float
            The angle measured on the current frame, in degrees.
        baseline : float
            The driver's calibrated neutral angle, in degrees.

        Returns
        -------
        float
            The deviation from neutral along the shortest arc, in degrees.
        """
        return ((float(current) - float(baseline) + 180.0) % 360.0) - 180.0

    def update(self, pitch: float, roll: float, timestamp_ms: float, *, usable: bool) -> Dict[str, Any]:
        """Advance the tracker by one frame and report the head posture state.

        While uncalibrated the method collects samples and learns the neutral
        baseline from their median, which is robust to the occasional bad
        landmark fit in a way a mean would not be. Once calibrated it measures
        the deviation, and a deviation past the threshold only becomes active
        after it has persisted, so momentary glances never register as a nod.

        Parameters
        ----------
        pitch : float
            Head pitch on this frame, in degrees.
        roll : float
            Head roll on this frame, in degrees.
        timestamp_ms : float
            Frame time, used to measure how long a deviation has persisted.
        usable : bool
            Whether the pose estimate is reliable enough to calibrate from,
            which keeps a poor landmark fit out of the learned baseline.

        Returns
        -------
        dict
            The baselines, the current deviations, the two persistence-gated
            active flags and the calibration state, ready to join the frame's
            metrics payload.
        """
        if usable and not self.calibrated:
            self._pitch_samples.append(float(pitch))
            self._roll_samples.append(float(roll))
            if len(self._pitch_samples) >= self.CALIBRATION_FRAMES:
                self._pitch_baseline = float(median(self._pitch_samples))
                self._roll_baseline = float(median(self._roll_samples))
        pb = self._pitch_baseline
        rb = self._roll_baseline
        pitch_delta = 0.0 if pb is None else self.angular_delta(pitch, pb)
        roll_delta = 0.0 if rb is None else self.angular_delta(roll, rb)
        down_delta = self.pitch_down_sign * pitch_delta
        pitch_candidate = self.calibrated and down_delta >= self.PITCH_DOWN_DEVIATION_DEG
        roll_candidate = self.calibrated and abs(roll_delta) >= self.ROLL_DEVIATION_DEG
        if pitch_candidate and self._pitch_candidate_since is None:
            self._pitch_candidate_since = timestamp_ms
        if not pitch_candidate:
            self._pitch_candidate_since = None
        if roll_candidate and self._roll_candidate_since is None:
            self._roll_candidate_since = timestamp_ms
        if not roll_candidate:
            self._roll_candidate_since = None
        return {
            "Head_Pitch_Baseline": round(0.0 if pb is None else pb, 2),
            "Head_Roll_Baseline": round(0.0 if rb is None else rb, 2),
            "Head_Pitch_Delta": round(pitch_delta, 2),
            "Head_Roll_Delta": round(roll_delta, 2),
            "Head_Pitch_Down_Delta": round(down_delta, 2),
            "Head_Pitch_Down_Active": bool(pitch_candidate and self._pitch_candidate_since is not None and timestamp_ms - self._pitch_candidate_since >= self.PERSIST_MS),
            "Head_Roll_Active": bool(roll_candidate and self._roll_candidate_since is not None and timestamp_ms - self._roll_candidate_since >= self.PERSIST_MS),
            "Head_Pose_Calibrated": self.calibrated,
            "Head_Pitch_Down_Sign": self.pitch_down_sign,
        }
