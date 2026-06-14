import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================== LOAD WFDB DATA ======================
print("Loading ECG data from WFDB...")
df = pd.read_csv('210.csv')
print(f"Loaded {len(df):,} samples | Channels: {list(df.columns)}")

# Use MLII lead (main ECG signal)
signal = df['MLII'].values.astype(np.float32)

# Normalize signal
signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

print(f"Signal length: {len(signal)} samples (~{len(signal)/360:.1f} seconds)")

# ====================== ENCODERS ======================

def bsa_encoder(ecg_signal, filter_kernel=None, threshold=0.2):
    """Bens Spiker Algorithm"""
    if filter_kernel is None:
        # Default Gaussian kernel
        kernel_len = 20
        t = np.linspace(-2, 2, kernel_len)
        filter_kernel = np.exp(-t**2) / np.sqrt(2*np.pi)
    
    num_steps = len(ecg_signal)
    spikes = np.zeros(num_steps)
    kernel_len = len(filter_kernel)
    
    signal_copy = ecg_signal.copy()
    
    for t in range(num_steps - kernel_len):
        error1 = 0
        error2 = 0
        for k in range(kernel_len):
            error1 += abs(signal_copy[t + k] - filter_kernel[k])
            error2 += abs(signal_copy[t + k])
        
        if error1 <= (error2 - threshold):
            spikes[t] = 1
            for k in range(kernel_len):
                signal_copy[t + k] -= filter_kernel[k]
    
    return spikes


def delta_encoder(ecg_signal, threshold=0.15):
    """Temporal Contrast / Delta Modulation"""
    num_steps = len(ecg_signal)
    up_spikes = np.zeros(num_steps)
    down_spikes = np.zeros(num_steps)
    
    ref_level = ecg_signal[0]
    
    for t in range(1, num_steps):
        diff = ecg_signal[t] - ref_level
        if diff >= threshold:
            up_spikes[t] = 1
            ref_level = ecg_signal[t]
        elif diff <= -threshold:
            down_spikes[t] = 1
            ref_level = ecg_signal[t]
    
    return up_spikes, down_spikes


def level_crossing_encoder(ecg_signal, num_levels=20):
    """Level Crossing Encoder"""
    num_steps = len(ecg_signal)
    spikes = np.zeros(num_steps)
    
    sig_min, sig_max = np.min(ecg_signal), np.max(ecg_signal)
    thresholds = np.linspace(sig_min, sig_max, num_levels)
    
    last_zone = np.digitize(ecg_signal[0], thresholds)
    
    for t in range(1, num_steps):
        current_zone = np.digitize(ecg_signal[t], thresholds)
        if current_zone != last_zone:
            spikes[t] = 1
            last_zone = current_zone
    return spikes


# ====================== RUN ENCODER ======================

# Choose encoder here:
encoder_type = "delta"   # Options: "delta", "bsa", "level"

if encoder_type == "delta":
    up, down = delta_encoder(signal, threshold=0.15)
    spikes = up + down
    print("Using Delta Modulation Encoder")
    
elif encoder_type == "bsa":
    spikes = bsa_encoder(signal, threshold=0.25)
    print("Using Bens Spiker Algorithm (BSA)")
    
elif encoder_type == "level":
    spikes = level_crossing_encoder(signal, num_levels=30)
    print("Using Level Crossing Encoder")

print(f"Total spikes generated: {int(spikes.sum())}")

# Save spike train
spike_df = pd.DataFrame({
    'time': df['time'],
    'spikes': spikes
})
spike_df.to_csv(f'210_{encoder_type}_spikes.csv', index=False)
print(f"✅ Spike train saved to: 210_{encoder_type}_spikes.csv")

# ====================== PLOT ======================
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(df['time'][:3600], signal[:3600], 'b-', linewidth=1)
plt.title('Normalized ECG Signal (First 10 seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df['time'][:3600], spikes[:3600], 'r|', markersize=4)
plt.title(f'Spike Train - {encoder_type.upper()} Encoder')
plt.xlabel('Time (seconds)')
plt.ylabel('Spikes')
plt.grid(True)

plt.tight_layout()
plt.savefig(f'210_{encoder_type}_spikes.png')
plt.show()