import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('210_delta_spikes.csv')
ecg_df = pd.read_csv('210.csv')

t = df['time'].values
spikes = df['spikes'].values
ecg = ecg_df['MLII'].values

mask_10s = t <= 10
mask_8s = t <= 8

# Plot 1 - 10 seconds
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
ax1.plot(t[mask_10s], ecg[mask_10s], color='blue', linewidth=0.8)
ax1.set_title('Normalized ECG Signal (First 10 seconds)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)
ax2.plot(t[mask_10s], spikes[mask_10s], color='red', linewidth=0.5)
ax2.set_title('Spike Train - DELTA Encoder')
ax2.set_ylabel('Spikes')
ax2.set_xlabel('Time (seconds)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/delta_spike_10s.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2 - 8 seconds
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
ax1.plot(t[mask_8s], ecg[mask_8s], color='blue', linewidth=0.8)
ax1.set_title('ECG Signal to Spike Train (Delta Encoder)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)
ax2.plot(t[mask_8s], spikes[mask_8s], color='red', linewidth=0.5)
ax2.set_title('Spike Train - DELTA Encoder')
ax2.set_ylabel('Spikes')
ax2.set_xlabel('Time (seconds)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/delta_spike_8s.png', dpi=150, bbox_inches='tight')
plt.close()

print('Both images saved to assets/ folder!')