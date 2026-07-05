# sensor.py
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import json
from collections import deque


class EyeBlinkSensor:
    """
     PERCLOS includes the current ongoing closure (no lag).
     ear_fast -> ONLY for state transitions & blink timing.
       ear_slow -> ONLY for sleep confirmation.
    """

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, debug=True):
        self.debug = debug

        # ---------- HYSTERESIS ----------
        self.ear_close_thresh = 0.20  # OPEN -> CLOSED
        self.ear_open_thresh = 0.23  # CLOSED -> OPEN

        # ---------- TIME THRESHOLDS (ms) ----------
        self.min_blink_ms = 60
        self.max_blink_ms = 350
        self.sleep_ms = 1000  # in speed of 100 Km/h we cover 27.8 meters :|

        # ---------- STATE ----------
        self.eye_state = "OPEN"
        self.eye_closed_start = None

        # ---------- BLINK HISTORY (bounded) ----------
        self.blink_times = deque(maxlen=300)

        # ---------- EAR smoothing (sleep only) ----------
        self.ear_window = deque(maxlen=15)

        # ---------- PERCLOS ----------
        self.perclos_window_ms = 60_000  # 60 seconds window
        self.closed_intervals = deque()  # list of (start_ms, end_ms) completed closures

        # ---------- EAR HISTORY (for plotting) ----------
        self.ear_history = deque(maxlen=300)  # ~10 seconds at 30 FPS
        self.ear_timestamps = deque(maxlen=300)

        # ---------- MediaPipe ----------
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawer = mp.solutions.drawing_utils

    # ==================================================
    # EAR math
    # ==================================================

    def compute_EAR(self, landmarks, eye_indices):
        def dist(a, b):
            return math.dist(
                [landmarks[a].x, landmarks[a].y],
                [landmarks[b].x, landmarks[b].y]
            )

        p1, p2, p3, p4, p5, p6 = eye_indices
        vertical = dist(p2, p6) + dist(p3, p5)
        horizontal = 2.0 * dist(p1, p4)
        return vertical / horizontal

    # ==================================================
    # Main processing
    # ==================================================

    def process_frame(self, frame):
        # NEW: controller can poll this each frame
    # NEW: controller can poll this each frame
        self.last_risk_payload = None

        now_ms = time.time() * 1000
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            out = self._output(
                ear=None,
                state="NO_FACE",
                closed_ms=0,
                blinks_per_min=0,
                perclos=0.0
            )
            # no risk payload when no face
            return out

        landmarks = results.multi_face_landmarks[0].landmark

        # ---------- EAR ----------
        ear_left = self.compute_EAR(landmarks, self.LEFT_EYE)
        ear_right = self.compute_EAR(landmarks, self.RIGHT_EYE)
        ear_fast = (ear_left + ear_right) / 2.0

        # ear_slow for sleep only
        self.ear_window.append(ear_fast)
        ear_slow = float(np.mean(self.ear_window))

        # ---------- EAR HISTORY ----------
        self.ear_history.append(ear_fast)
        self.ear_timestamps.append(now_ms)

        # ---------- STATE MACHINE (ear_fast ONLY) ----------
        closed_duration = 0

        if self.eye_state == "OPEN":
            if ear_fast < self.ear_close_thresh:
                self.eye_state = "CLOSED"
                self.eye_closed_start = now_ms

        elif self.eye_state == "CLOSED":
            closed_duration = now_ms - self.eye_closed_start

            if ear_fast > self.ear_open_thresh:
                # store completed closure interval for PERCLOS
                self.closed_intervals.append((self.eye_closed_start, now_ms))

                # blink detection
                if self.min_blink_ms <= closed_duration <= self.max_blink_ms:
                    self.blink_times.append(now_ms)

                self.eye_state = "OPEN"
                self.eye_closed_start = None
                closed_duration = 0

        # ---------- SLEEP (ear_slow ONLY) ----------
        sleeping = (
            self.eye_state == "CLOSED"
            and closed_duration >= self.sleep_ms
            and ear_slow < self.ear_close_thresh
        )

        if sleeping:
            state = "SLEEPING"
        elif self.eye_state == "CLOSED":
            state = "EYES_CLOSED"
        else:
            state = "EYES_OPEN"

        # ---------- BLINK RATE ----------
        self._cleanup_old_blinks(now_ms)
        blinks_per_min = len(self.blink_times)

        # ---------- PERCLOS (includes current closure) ----------
        perclos = self._compute_perclos(now_ms)

        # ---------- DEBUG UI ----------
        if self.debug:
            self.drawer.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                self.mp_face.FACEMESH_CONTOURS
            )

            cv2.putText(frame, f"EAR fast: {ear_fast:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            cv2.putText(frame, f"EAR slow: {ear_slow:.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.putText(frame, f"State: {state}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.putText(frame, f"Closed ms: {int(closed_duration)}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 100), 2)

            cv2.putText(frame, f"Blinks/min: {blinks_per_min}", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            cv2.putText(frame, f"PERCLOS(60s): {perclos:.2f}", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 100), 2)

            # Visual alert: RED BORDER when sleeping
            if state == "SLEEPING":
                border_thickness = 15
                cv2.rectangle(frame,
                            (0, 0),
                            (w - 1, h - 1),
                            (0, 0, 255),
                            border_thickness)

                cv2.putText(frame, "SLEEPING!", (w - 360, int(h / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 6)

            # Draw EAR plot
            self._draw_ear_plot(frame, w, h)

        # build the standard output (unchanged behavior)
        out = self._output(
            ear=ear_fast,
            state=state,
            closed_ms=closed_duration,
            blinks_per_min=blinks_per_min,
            perclos=perclos
        )

        # NEW: only when risk is detected, expose payload for controller -> agent
        if state in ["EYES_CLOSED", "SLEEPING"] or perclos > 0.15:
            self.last_risk_payload = out

        return out


    # ==================================================
    # Helpers
    # ==================================================

    def _cleanup_old_blinks(self, now_ms):
        cutoff = now_ms - 60_000
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()

    def _compute_perclos(self, now_ms):
        cutoff = now_ms - self.perclos_window_ms
        closed_time = 0

        # remove intervals that end before cutoff
        while self.closed_intervals and self.closed_intervals[0][1] < cutoff:
            self.closed_intervals.popleft()

        # sum completed intervals overlapping the window
        for start, end in self.closed_intervals:
            closed_time += max(0, min(end, now_ms) - max(start, cutoff))

        # include current ongoing closure (FIX)
        if self.eye_state == "CLOSED" and self.eye_closed_start is not None:
            closed_time += max(0, now_ms - max(self.eye_closed_start, cutoff))

        return closed_time / self.perclos_window_ms

    @staticmethod
    def _avg(dq):
        if not dq:
            return 0.0
        return float(sum(dq) / len(dq))

    def _output(self, ear, state, closed_ms, blinks_per_min, perclos):
        output = {
            "EAR": ear,
            "state": state,
            "closed_duration_ms": int(closed_ms),
            "blinks_per_min": int(blinks_per_min),
            "perclos": float(perclos)
        }

        # Stream to agent when there's a risk of sleeping
        if state in ["EYES_CLOSED", "SLEEPING"] or perclos > 0.15:
            self._stream_to_agent(output)

        return output

    def _stream_to_agent(self, data):
        """Stream parameters to agent console when sleeping risk detected"""
        output = {
            "case": "SLEEPING_RISK_DETECTED",
            "Driver_State": data['state'],
            "EAR": round(data['EAR'], 3) if data['EAR'] is not None else None,
            "Eyes Closed Duration": data['closed_duration_ms'],
            "Blinks/min": data['blinks_per_min'],
            "PERCLOS": round(data['perclos'], 3)
        }
        #print(json.dumps(output))
        return output

    def _draw_ear_plot(self, frame, w, h):
        """Draw real-time EAR plot with thresholds"""
        if len(self.ear_history) < 2:
            return

        # Plot dimensions and position
        plot_width = 400
        plot_height = 150
        plot_x = w - plot_width - 20
        plot_y = 20

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (plot_x, plot_y),
                      (plot_x + plot_width, plot_y + plot_height),
                      (0, 0, 0),
                      -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Border
        cv2.rectangle(frame,
                      (plot_x, plot_y),
                      (plot_x + plot_width, plot_y + plot_height),
                      (100, 100, 100),
                      2)

        # Title
        cv2.putText(frame, "EAR Timeline",
                    (plot_x + 10, plot_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # EAR range for plotting (0.1 to 0.4)
        ear_min = 0.1
        ear_max = 0.4
        ear_range = ear_max - ear_min

        # Convert EAR values to plot coordinates
        ear_list = list(self.ear_history)
        points = []
        for i, ear_val in enumerate(ear_list):
            x = plot_x + int((i / len(ear_list)) * plot_width)
            # Clamp EAR value to range
            ear_clamped = max(ear_min, min(ear_max, ear_val))
            y = plot_y + plot_height - int(((ear_clamped - ear_min) / ear_range) * (plot_height - 30)) - 15
            points.append((x, y))

        # Draw threshold lines
        # Close threshold (red)
        close_y = plot_y + plot_height - int(((self.ear_close_thresh - ear_min) / ear_range) * (plot_height - 30)) - 15
        cv2.line(frame,
                 (plot_x, close_y),
                 (plot_x + plot_width, close_y),
                 (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Close: {self.ear_close_thresh:.2f}",
                    (plot_x + 5, close_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Open threshold (green)
        open_y = plot_y + plot_height - int(((self.ear_open_thresh - ear_min) / ear_range) * (plot_height - 30)) - 15
        cv2.line(frame,
                 (plot_x, open_y),
                 (plot_x + plot_width, open_y),
                 (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Open: {self.ear_open_thresh:.2f}",
                    (plot_x + 5, open_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Draw EAR line
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 255, 255), 2, cv2.LINE_AA)

        # Draw current EAR value
        if points:
            last_point = points[-1]
            cv2.circle(frame, last_point, 4, (0, 255, 255), -1)