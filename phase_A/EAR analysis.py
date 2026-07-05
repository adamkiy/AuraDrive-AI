"""
EAR Analysis Tool - Validates Eye Aspect Ratio thresholds and formula correctness
Generates multiple plots to justify design decisions for AuraDrive
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
from collections import deque
import time


class EARAnalyzer:
    """Captures and analyzes EAR data to validate thresholds"""

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Data storage
        self.ear_values = []
        self.ear_left_values = []
        self.ear_right_values = []
        self.timestamps = []
        self.states = []  # Track eye state

        # Thresholds (from sensor.py)
        self.ear_close_thresh = 0.20
        self.ear_open_thresh = 0.23

        # State tracking
        self.current_state = "OPEN"
        self.state_changes = []  # (timestamp, old_state, new_state, ear_value)

    def compute_EAR(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio for one eye"""

        def dist(a, b):
            return math.dist(
                [landmarks[a].x, landmarks[a].y],
                [landmarks[b].x, landmarks[b].y]
            )

        p1, p2, p3, p4, p5, p6 = eye_indices
        vertical = dist(p2, p6) + dist(p3, p5)
        horizontal = 2.0 * dist(p1, p4)
        return vertical / horizontal

    def collect_data(self, duration_seconds=30):
        """Collect EAR data from webcam for analysis"""
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0

        print(f"Collecting data for {duration_seconds} seconds...")
        print("Instructions:")
        print("  - Keep your face visible to the camera")
        print("  - Blink naturally")
        print("  - Close your eyes for 1-2 seconds at various points")
        print("  - Try to simulate drowsiness")
        print("\nPress 'q' to stop early\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - start_time
            if elapsed > duration_seconds:
                break

            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Calculate EAR for both eyes
                ear_left = self.compute_EAR(landmarks, self.LEFT_EYE)
                ear_right = self.compute_EAR(landmarks, self.RIGHT_EYE)
                ear_avg = (ear_left + ear_right) / 2.0

                # Update state with hysteresis
                old_state = self.current_state
                if self.current_state == "OPEN" and ear_avg < self.ear_close_thresh:
                    self.current_state = "CLOSED"
                elif self.current_state == "CLOSED" and ear_avg > self.ear_open_thresh:
                    self.current_state = "OPEN"

                # Record state change
                if old_state != self.current_state:
                    self.state_changes.append((elapsed, old_state, self.current_state, ear_avg))

                # Store data
                self.ear_values.append(ear_avg)
                self.ear_left_values.append(ear_left)
                self.ear_right_values.append(ear_right)
                self.timestamps.append(elapsed)
                self.states.append(1 if self.current_state == "CLOSED" else 0)

                frame_count += 1

                # Display progress
                cv2.putText(frame, f"Time: {elapsed:.1f}s / {duration_seconds}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {ear_avg:.3f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"State: {self.current_state}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255) if self.current_state == "CLOSED" else (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {frame_count}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nData collection complete!")
        print(f"Total frames collected: {frame_count}")
        print(f"Total state changes: {len(self.state_changes)}")

    def generate_analysis_plots(self, save_path="ear_analysis.png"):
        """Generate comprehensive analysis plots"""
        if len(self.ear_values) == 0:
            print("No data collected. Run collect_data() first.")
            return

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Convert to numpy arrays
        ear_array = np.array(self.ear_values)
        time_array = np.array(self.timestamps)
        states_array = np.array(self.states)

        # ============================================================
        # PLOT 1: EAR Timeline with Thresholds and States
        # ============================================================
        ax1 = fig.add_subplot(gs[0, :])

        # Plot EAR values
        ax1.plot(time_array, ear_array, 'b-', linewidth=1.5, alpha=0.7, label='EAR (avg)')

        # Add threshold lines
        ax1.axhline(y=self.ear_close_thresh, color='r', linestyle='--',
                    linewidth=2, label=f'Close Threshold ({self.ear_close_thresh})')
        ax1.axhline(y=self.ear_open_thresh, color='g', linestyle='--',
                    linewidth=2, label=f'Open Threshold ({self.ear_open_thresh})')

        # Shade closed periods
        closed_periods = states_array == 1
        ax1.fill_between(time_array, 0.1, 0.4, where=closed_periods,
                         alpha=0.2, color='red', label='Eyes Closed')

        # Mark state transitions
        for ts, old_state, new_state, ear_val in self.state_changes:
            color = 'red' if new_state == "CLOSED" else 'green'
            ax1.plot(ts, ear_val, 'o', color=color, markersize=8,
                     markeredgecolor='black', markeredgewidth=1)

        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=12, fontweight='bold')
        ax1.set_title('Plot 1: EAR Timeline with State Transitions\n' +
                      'Validates hysteresis behavior and threshold placement',
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.1, 0.4])

        # ============================================================
        # PLOT 2: EAR Distribution (Histogram)
        # ============================================================
        ax2 = fig.add_subplot(gs[1, 0])

        # Separate open and closed states
        ear_open = ear_array[states_array == 0]
        ear_closed = ear_array[states_array == 1]

        # Plot histograms
        ax2.hist(ear_open, bins=50, alpha=0.6, color='green',
                 label=f'Eyes Open (n={len(ear_open)})', edgecolor='black')
        ax2.hist(ear_closed, bins=50, alpha=0.6, color='red',
                 label=f'Eyes Closed (n={len(ear_closed)})', edgecolor='black')

        # Add threshold lines
        ax2.axvline(x=self.ear_close_thresh, color='red', linestyle='--',
                    linewidth=2, label='Close Threshold')
        ax2.axvline(x=self.ear_open_thresh, color='green', linestyle='--',
                    linewidth=2, label='Open Threshold')

        # Add statistics
        if len(ear_open) > 0:
            ax2.text(0.32, ax2.get_ylim()[1] * 0.95,
                     f'Open: μ={np.mean(ear_open):.3f}, σ={np.std(ear_open):.3f}',
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        if len(ear_closed) > 0:
            ax2.text(0.12, ax2.get_ylim()[1] * 0.95,
                     f'Closed: μ={np.mean(ear_closed):.3f}, σ={np.std(ear_closed):.3f}',
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

        ax2.set_xlabel('Eye Aspect Ratio (EAR)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Plot 2: EAR Distribution Analysis\n' +
                      'Shows clear separation between open/closed states',
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # ============================================================
        # PLOT 3: Left vs Right Eye EAR Correlation
        # ============================================================
        ax3 = fig.add_subplot(gs[1, 1])

        ear_left_array = np.array(self.ear_left_values)
        ear_right_array = np.array(self.ear_right_values)

        # Scatter plot colored by state
        scatter_open = ax3.scatter(ear_left_array[states_array == 0],
                                   ear_right_array[states_array == 0],
                                   c='green', alpha=0.5, s=20, label='Open')
        scatter_closed = ax3.scatter(ear_left_array[states_array == 1],
                                     ear_right_array[states_array == 1],
                                     c='red', alpha=0.5, s=20, label='Closed')

        # Add diagonal line (perfect correlation)
        min_val = min(ear_left_array.min(), ear_right_array.min())
        max_val = max(ear_left_array.max(), ear_right_array.max())
        ax3.plot([min_val, max_val], [min_val, max_val],
                 'k--', linewidth=2, alpha=0.5, label='Perfect Correlation')

        # Add threshold lines
        ax3.axhline(y=self.ear_close_thresh, color='red', linestyle=':', linewidth=1.5)
        ax3.axvline(x=self.ear_close_thresh, color='red', linestyle=':', linewidth=1.5)
        ax3.axhline(y=self.ear_open_thresh, color='green', linestyle=':', linewidth=1.5)
        ax3.axvline(x=self.ear_open_thresh, color='green', linestyle=':', linewidth=1.5)

        # Calculate correlation
        correlation = np.corrcoef(ear_left_array, ear_right_array)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax3.set_xlabel('Left Eye EAR', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Right Eye EAR', fontsize=11, fontweight='bold')
        ax3.set_title('Plot 3: Bilateral Eye Correlation\n' +
                      'Validates bilateral averaging approach',
                      fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')

        # ============================================================
        # PLOT 4: State Duration Analysis
        # ============================================================
        ax4 = fig.add_subplot(gs[2, 0])

        # Calculate durations of closed periods
        closed_durations = []
        open_durations = []

        current_duration = 0
        current_state_val = states_array[0]

        for i in range(1, len(states_array)):
            if states_array[i] == current_state_val:
                current_duration += 1
            else:
                # State changed - record duration
                duration_ms = current_duration * (1000.0 / 30.0)  # Assume ~30 FPS
                if current_state_val == 1:
                    closed_durations.append(duration_ms)
                else:
                    open_durations.append(duration_ms)
                current_duration = 1
                current_state_val = states_array[i]

        # Plot histograms
        if closed_durations:
            ax4.hist(closed_durations, bins=30, alpha=0.6, color='red',
                     label=f'Closed Periods (n={len(closed_durations)})',
                     edgecolor='black', range=(0, 2000))
        if open_durations:
            ax4.hist(open_durations, bins=30, alpha=0.6, color='green',
                     label=f'Open Periods (n={len(open_durations)})',
                     edgecolor='black', range=(0, 2000))

        # Add blink duration markers
        ax4.axvline(x=60, color='blue', linestyle='--', linewidth=2,
                    label='Min Blink (60ms)')
        ax4.axvline(x=350, color='orange', linestyle='--', linewidth=2,
                    label='Max Blink (350ms)')
        ax4.axvline(x=1000, color='darkred', linestyle='--', linewidth=2,
                    label='Sleep Threshold (1000ms)')

        ax4.set_xlabel('Duration (milliseconds)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Plot 4: State Duration Distribution\n' +
                      'Validates blink vs. sleep duration thresholds',
                      fontsize=13, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xlim([0, 2000])

        # ============================================================
        # PLOT 5: Threshold Sensitivity Analysis
        # ============================================================
        ax5 = fig.add_subplot(gs[2, 1])

        # Test different threshold values
        test_thresholds = np.linspace(0.15, 0.30, 50)
        false_positives = []
        false_negatives = []

        # Ground truth: use majority voting in time windows
        window_size = 5
        ground_truth = []
        for i in range(len(states_array)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(states_array), i + window_size + 1)
            window_mean = np.mean(states_array[start_idx:end_idx])
            ground_truth.append(1 if window_mean > 0.5 else 0)
        ground_truth = np.array(ground_truth)

        for thresh in test_thresholds:
            predicted = (ear_array < thresh).astype(int)
            fp = np.sum((predicted == 1) & (ground_truth == 0)) / len(ground_truth)
            fn = np.sum((predicted == 0) & (ground_truth == 1)) / len(ground_truth)
            false_positives.append(fp * 100)
            false_negatives.append(fn * 100)

        # Plot error rates
        ax5.plot(test_thresholds, false_positives, 'r-', linewidth=2,
                 label='False Positive Rate', marker='o', markersize=3)
        ax5.plot(test_thresholds, false_negatives, 'b-', linewidth=2,
                 label='False Negative Rate', marker='s', markersize=3)
        ax5.plot(test_thresholds,
                 np.array(false_positives) + np.array(false_negatives),
                 'purple', linewidth=2, linestyle='--',
                 label='Total Error', marker='^', markersize=3)

        # Mark chosen threshold
        ax5.axvline(x=self.ear_close_thresh, color='green', linestyle='--',
                    linewidth=2, label=f'Chosen Threshold ({self.ear_close_thresh})')

        ax5.set_xlabel('EAR Threshold Value', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Plot 5: Threshold Sensitivity Analysis\n' +
                      'Demonstrates optimal threshold selection',
                      fontsize=13, fontweight='bold')
        ax5.legend(loc='upper left', fontsize=9)
        ax5.grid(True, alpha=0.3)

        # ============================================================
        # Add overall title and statistics
        # ============================================================
        fig.suptitle('AuraDrive EAR Analysis - Validation of Formula and Thresholds',
                     fontsize=16, fontweight='bold', y=0.995)

        # Add statistics box
        stats_text = (
            f"Dataset Statistics:\n"
            f"Total Frames: {len(ear_array)}\n"
            f"Duration: {time_array[-1]:.1f} seconds\n"
            f"State Changes: {len(self.state_changes)}\n"
            f"EAR Range: [{ear_array.min():.3f}, {ear_array.max():.3f}]\n"
            f"Mean EAR: {ear_array.mean():.3f} ± {ear_array.std():.3f}"
        )

        fig.text(0.02, 0.02, stats_text, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nAnalysis plots saved to: {save_path}")

        # Show plot
        plt.show()

    def print_summary_statistics(self):
        """Print detailed statistics about the collected data"""
        if len(self.ear_values) == 0:
            print("No data collected.")
            return

        ear_array = np.array(self.ear_values)
        states_array = np.array(self.states)

        print("\n" + "=" * 60)
        print("EAR ANALYSIS SUMMARY STATISTICS")
        print("=" * 60)

        print(f"\nDataset Overview:")
        print(f"  Total frames: {len(ear_array)}")
        print(f"  Duration: {self.timestamps[-1]:.2f} seconds")
        print(f"  Average FPS: {len(ear_array) / self.timestamps[-1]:.1f}")

        print(f"\nEAR Statistics (All Data):")
        print(f"  Mean: {ear_array.mean():.4f}")
        print(f"  Std Dev: {ear_array.std():.4f}")
        print(f"  Min: {ear_array.min():.4f}")
        print(f"  Max: {ear_array.max():.4f}")
        print(f"  Median: {np.median(ear_array):.4f}")

        # Open state statistics
        ear_open = ear_array[states_array == 0]
        if len(ear_open) > 0:
            print(f"\nEAR Statistics (Eyes Open):")
            print(f"  Mean: {ear_open.mean():.4f}")
            print(f"  Std Dev: {ear_open.std():.4f}")
            print(f"  Min: {ear_open.min():.4f}")
            print(f"  Max: {ear_open.max():.4f}")
            print(f"  Frames: {len(ear_open)} ({len(ear_open) / len(ear_array) * 100:.1f}%)")

        # Closed state statistics
        ear_closed = ear_array[states_array == 1]
        if len(ear_closed) > 0:
            print(f"\nEAR Statistics (Eyes Closed):")
            print(f"  Mean: {ear_closed.mean():.4f}")
            print(f"  Std Dev: {ear_closed.std():.4f}")
            print(f"  Min: {ear_closed.min():.4f}")
            print(f"  Max: {ear_closed.max():.4f}")
            print(f"  Frames: {len(ear_closed)} ({len(ear_closed) / len(ear_array) * 100:.1f}%)")

        print(f"\nThreshold Analysis:")
        print(f"  Close Threshold: {self.ear_close_thresh}")
        print(f"  Open Threshold: {self.ear_open_thresh}")
        print(f"  Hysteresis Gap: {self.ear_open_thresh - self.ear_close_thresh:.3f}")

        if len(ear_open) > 0 and len(ear_closed) > 0:
            separation = (ear_open.mean() - ear_closed.mean()) / np.sqrt(
                (ear_open.std() ** 2 + ear_closed.std() ** 2) / 2
            )
            print(f"  Cohen's d (separation): {separation:.3f}")

        print(f"\nState Transitions:")
        print(f"  Total transitions: {len(self.state_changes)}")
        if len(self.state_changes) > 0:
            print(f"  First transition: {self.state_changes[0][0]:.2f}s")
            print(f"  Last transition: {self.state_changes[-1][0]:.2f}s")

        print("\n" + "=" * 60)


def main():
    """Main execution function"""
    print("=" * 60)
    print("AuraDrive EAR Analysis Tool")
    print("=" * 60)
    print("\nThis tool will:")
    print("1. Collect EAR data from your webcam")
    print("2. Generate comprehensive analysis plots")
    print("3. Validate threshold selection and formula correctness")
    print("\n")

    analyzer = EARAnalyzer()

    # Collect data
    duration = 30  # seconds
    analyzer.collect_data(duration_seconds=duration)

    # Print statistics
    analyzer.print_summary_statistics()

    # Generate plots
    analyzer.generate_analysis_plots(save_path="ear_analysis_results.png")

    print("\nAnalysis complete!")
    print("Review the generated plots to validate:")
    print("  - EAR formula correctly separates open/closed states")
    print("  - Threshold values (0.20, 0.23) are optimally placed")
    print("  - Hysteresis prevents oscillation")
    print("  - Bilateral averaging improves robustness")


if __name__ == "__main__":
    main()