"""
live_ecg_stream.py
==================
Live ECG Streaming + Real-Time Spike Visualization
Replays 210.csv through the Delta Modulation encoder
and animates the ECG + spike train in real time.

Usage:
    python live_ecg_stream.py
    python live_ecg_stream.py --speed 2.0   # 2x playback speed
    python live_ecg_stream.py --window 5    # 5-second scrolling window
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import argparse
import time
import sys

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
SAMPLE_RATE     = 360       # Hz (MIT-BIH standard)
WINDOW_SEC      = 8         # seconds visible in scrolling window
UPDATE_INTERVAL = 50        # ms between animation frames (20 FPS)
SAMPLES_PER_FRAME = 3       # ECG samples added per frame
DELTA_THRESHOLD = 0.03      # Delta modulation step size

# ─────────────────────────────────────────────
#  DELTA MODULATION ENCODER (real-time)
# ─────────────────────────────────────────────
class DeltaEncoder:
    def __init__(self, threshold=DELTA_THRESHOLD):
        self.threshold = threshold
        self.prev = 0.0

    def encode(self, sample):
        diff = sample - self.prev
        if diff > self.threshold:
            self.prev += self.threshold
            return 1
        elif diff < -self.threshold:
            self.prev -= self.threshold
            return -1
        return 0

# ─────────────────────────────────────────────
#  LIVE STREAM CLASS
# ─────────────────────────────────────────────
class LiveECGStream:
    def __init__(self, csv_path, window_sec=WINDOW_SEC, speed=1.0):
        print(f"\n{'='*55}")
        print(f"  🫀 LIVE ECG SPIKE ENCODER — Delta Modulation")
        print(f"{'='*55}")
        print(f"  📂 Source   : {csv_path}")
        print(f"  ⏱  Window   : {window_sec} seconds")
        print(f"  ⚡ Speed    : {speed}x")
        print(f"  📡 Rate     : {SAMPLE_RATE} Hz")
        print(f"{'='*55}\n")

        # Load data
        print("Loading ECG data...")
        df = pd.read_csv(csv_path)
        self.ecg_raw = df['MLII'].values.astype(float)

        # Normalize
        self.ecg = (self.ecg_raw - self.ecg_raw.mean()) / self.ecg_raw.std()
        self.total_samples = len(self.ecg)
        print(f"✅ Loaded {self.total_samples:,} samples ({self.total_samples/SAMPLE_RATE:.1f} seconds)\n")

        self.window_samples = int(window_sec * SAMPLE_RATE)
        self.speed = speed
        self.encoder = DeltaEncoder()

        # Streaming buffers
        self.ecg_buffer   = np.zeros(self.window_samples)
        self.spike_buffer  = np.zeros(self.window_samples)
        self.time_buffer   = np.linspace(-window_sec, 0, self.window_samples)

        # Counters
        self.sample_idx   = 0
        self.total_spikes = 0
        self.start_time   = None
        self.spike_rate_history = []

        # Samples to advance per frame (affected by speed)
        self.samples_per_frame = max(1, int(SAMPLES_PER_FRAME * speed))

        self._build_figure()

    # ── UI SETUP ──────────────────────────────
    def _build_figure(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 9), facecolor='#0d1117')
        self.fig.canvas.manager.set_window_title('🫀 Live ECG Spike Encoder')

        gs = self.fig.add_gridspec(
            3, 2,
            height_ratios=[2, 1.2, 0.6],
            width_ratios=[3, 1],
            hspace=0.45, wspace=0.35,
            left=0.08, right=0.96, top=0.92, bottom=0.08
        )

        # ── ECG panel
        self.ax_ecg = self.fig.add_subplot(gs[0, 0])
        self.ax_ecg.set_facecolor('#0d1117')
        self.ax_ecg.set_title('Normalized ECG Signal  (live replay)',
                               color='#a0aec0', fontsize=11, pad=8)
        self.ax_ecg.set_ylabel('Amplitude', color='#a0aec0', fontsize=9)
        self.ax_ecg.tick_params(colors='#4a5568', labelsize=8)
        for sp in self.ax_ecg.spines.values():
            sp.set_color('#2d3748')
        self.ax_ecg.grid(True, alpha=0.15, color='#4a5568')
        self.ax_ecg.set_xlim(-WINDOW_SEC, 0)
        self.ax_ecg.set_ylim(-4, 6)
        self.line_ecg, = self.ax_ecg.plot([], [], color='#4299e1',
                                            linewidth=0.9, alpha=0.95)
        # R-peak markers
        self.scatter_peaks = self.ax_ecg.scatter([], [], color='#fc8181',
                                                   s=25, zorder=5, alpha=0.8)

        # ── Spike panel
        self.ax_spk = self.fig.add_subplot(gs[1, 0])
        self.ax_spk.set_facecolor('#0d1117')
        self.ax_spk.set_title('Delta Encoder — Spike Train  (+1 up  /  −1 down)',
                               color='#a0aec0', fontsize=11, pad=8)
        self.ax_spk.set_ylabel('Spike', color='#a0aec0', fontsize=9)
        self.ax_spk.set_xlabel('Time (seconds)', color='#a0aec0', fontsize=9)
        self.ax_spk.tick_params(colors='#4a5568', labelsize=8)
        for sp in self.ax_spk.spines.values():
            sp.set_color('#2d3748')
        self.ax_spk.set_xlim(-WINDOW_SEC, 0)
        self.ax_spk.set_ylim(-1.5, 1.5)
        self.ax_spk.axhline(0, color='#4a5568', linewidth=0.5, alpha=0.5)
        self.ax_spk.grid(True, alpha=0.15, color='#4a5568')
        self.line_spk, = self.ax_spk.plot([], [], color='#fc4d4d',
                                            linewidth=0.7, alpha=0.9)

        # ── Spike rate panel
        self.ax_rate = self.fig.add_subplot(gs[2, 0])
        self.ax_rate.set_facecolor('#0d1117')
        self.ax_rate.set_title('Spike Rate (spikes/sec)',
                                color='#a0aec0', fontsize=9, pad=5)
        self.ax_rate.set_ylabel('Rate', color='#a0aec0', fontsize=8)
        self.ax_rate.tick_params(colors='#4a5568', labelsize=7)
        for sp in self.ax_rate.spines.values():
            sp.set_color('#2d3748')
        self.ax_rate.set_xlim(0, 60)
        self.ax_rate.set_ylim(0, 150)
        self.ax_rate.grid(True, alpha=0.1, color='#4a5568')
        self.line_rate, = self.ax_rate.plot([], [], color='#68d391',
                                              linewidth=1.2)
        self.rate_times  = []
        self.rate_values = []

        # ── Stats panel
        self.ax_stats = self.fig.add_subplot(gs[:, 1])
        self.ax_stats.set_facecolor('#161b22')
        self.ax_stats.axis('off')
        for sp in self.ax_stats.spines.values():
            sp.set_color('#30363d')

        self.ax_stats.set_title('Live Stats', color='#e2e8f0',
                                 fontsize=11, fontweight='bold', pad=10)

        labels = ['⏱  Elapsed', '📍 Sample', '⚡ Spikes', '📈 Rate',
                  '💓 HR Est.', '📊 Progress']
        self.stat_texts = {}
        for i, lbl in enumerate(labels):
            y = 0.88 - i * 0.14
            self.ax_stats.text(0.05, y + 0.04, lbl,
                               transform=self.ax_stats.transAxes,
                               color='#718096', fontsize=8.5)
            self.stat_texts[lbl] = self.ax_stats.text(
                0.05, y - 0.01, '—',
                transform=self.ax_stats.transAxes,
                color='#e2e8f0', fontsize=11, fontweight='bold'
            )

        # Encoder label
        self.ax_stats.text(0.5, 0.06, 'DELTA\nMODULATION',
                           transform=self.ax_stats.transAxes,
                           color='#e94560', fontsize=13, fontweight='bold',
                           ha='center', va='center', alpha=0.6)

        self.fig.suptitle('🫀  ECG Spike Encoder  —  Live Stream',
                          color='#e2e8f0', fontsize=14, fontweight='bold', y=0.97)

    # ── DETECT R-PEAKS (simple threshold) ─────
    def _detect_peaks(self, ecg_window, time_window):
        threshold = 2.5
        peaks_x, peaks_y = [], []
        in_peak = False
        for i in range(1, len(ecg_window) - 1):
            if ecg_window[i] > threshold:
                if not in_peak:
                    peaks_x.append(time_window[i])
                    peaks_y.append(ecg_window[i])
                    in_peak = True
            else:
                in_peak = False
        return peaks_x, peaks_y

    # ── ANIMATION FRAME ───────────────────────
    def update(self, frame):
        if self.start_time is None:
            self.start_time = time.time()

        if self.sample_idx >= self.total_samples:
            self.sample_idx = 0   # loop replay
            self.encoder = DeltaEncoder()

        # Process N samples this frame
        n = self.samples_per_frame
        end_idx = min(self.sample_idx + n, self.total_samples)
        new_ecg    = self.ecg[self.sample_idx:end_idx]
        new_spikes = np.array([self.encoder.encode(s) for s in new_ecg])

        # Roll buffers
        self.ecg_buffer   = np.roll(self.ecg_buffer,   -len(new_ecg))
        self.spike_buffer  = np.roll(self.spike_buffer,  -len(new_spikes))
        self.ecg_buffer[-len(new_ecg):]    = new_ecg
        self.spike_buffer[-len(new_spikes):] = new_spikes

        self.total_spikes += int(np.sum(np.abs(new_spikes)))
        self.sample_idx   += n

        # Time axis
        elapsed = self.sample_idx / SAMPLE_RATE
        t_window = np.linspace(elapsed - WINDOW_SEC, elapsed, self.window_samples) - elapsed

        # Update ECG line
        self.line_ecg.set_data(t_window, self.ecg_buffer)

        # Update R-peak markers
        px, py = self._detect_peaks(self.ecg_buffer, t_window)
        if px:
            self.scatter_peaks.set_offsets(np.c_[px, py])
        else:
            self.scatter_peaks.set_offsets(np.empty((0, 2)))

        # Update spike line
        self.line_spk.set_data(t_window, self.spike_buffer)

        # Spike rate (per second)
        window_spikes = int(np.sum(np.abs(self.spike_buffer)))
        spike_rate    = window_spikes / WINDOW_SEC
        self.rate_times.append(elapsed)
        self.rate_values.append(spike_rate)
        if len(self.rate_times) > 200:
            self.rate_times.pop(0)
            self.rate_values.pop(0)
        self.line_rate.set_data(self.rate_times, self.rate_values)
        if elapsed > 60:
            self.ax_rate.set_xlim(elapsed - 60, elapsed)

        # Estimate heart rate from peak spacing
        peak_times = [t_window[i] for i in range(1, len(t_window)-1)
                      if self.ecg_buffer[i] > 2.5
                      and self.ecg_buffer[i] > self.ecg_buffer[i-1]
                      and self.ecg_buffer[i] > self.ecg_buffer[i+1]]
        if len(peak_times) >= 2:
            rr_intervals = np.diff(peak_times)
            hr_est = int(60.0 / np.mean(np.abs(rr_intervals))) if len(rr_intervals) > 0 else 0
        else:
            hr_est = 0

        # Update stats
        wall_elapsed = time.time() - self.start_time
        progress = self.sample_idx / self.total_samples * 100

        self.stat_texts['⏱  Elapsed'].set_text(
            f"{int(wall_elapsed//60):02d}:{int(wall_elapsed%60):02d}")
        self.stat_texts['📍 Sample'].set_text(
            f"{self.sample_idx:,} / {self.total_samples:,}")
        self.stat_texts['⚡ Spikes'].set_text(f"{self.total_spikes:,}")
        self.stat_texts['📈 Rate'].set_text(f"{spike_rate:.1f} / sec")
        self.stat_texts['💓 HR Est.'].set_text(
            f"{hr_est} BPM" if hr_est > 30 else "calculating...")
        self.stat_texts['📊 Progress'].set_text(f"{progress:.1f}%")

        return (self.line_ecg, self.line_spk, self.scatter_peaks,
                self.line_rate)

    # ── START ─────────────────────────────────
    def run(self):
        print("🚀 Starting live stream... Close the window to stop.\n")
        self.anim = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False
        )
        plt.show()
        print("\n✅ Stream ended.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Live ECG Spike Encoder — Delta Modulation Stream'
    )
    parser.add_argument('--csv',    default='210.csv',
                        help='Path to ECG CSV file (default: 210.csv)')
    parser.add_argument('--speed',  type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--window', type=int,   default=WINDOW_SEC,
                        help='Scrolling window in seconds (default: 8)')
    args = parser.parse_args()

    try:
        stream = LiveECGStream(
            csv_path=args.csv,
            window_sec=args.window,
            speed=args.speed
        )
        stream.run()
    except FileNotFoundError:
        print(f"\n❌ Error: '{args.csv}' not found.")
        print("   Make sure you're in the ecg-spike-encoding folder.")
        print("   Run: python convert_to_csv.py  to generate the CSV first.\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⛔ Stream stopped by user.")


if __name__ == '__main__':
    main()
