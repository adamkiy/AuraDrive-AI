"""Calibrated directional head-pose tracker (posture signal, not a diagnosis)."""
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
        return self._pitch_baseline is not None and self._roll_baseline is not None

    @staticmethod
    def angular_delta(current: float, baseline: float) -> float:
        return ((float(current) - float(baseline) + 180.0) % 360.0) - 180.0

    def update(self, pitch: float, roll: float, timestamp_ms: float, *, usable: bool) -> Dict[str, Any]:
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
