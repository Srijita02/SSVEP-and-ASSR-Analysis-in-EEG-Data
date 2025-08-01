import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend, resample, ttest_1samp
import curryreader as cr

# === Load Curry .dat files ===
folder_path = "8Hz_1st_visit"
all_subjects_8hz_data = {}
min_samples_required = 4000  # 2s minimum at 2000 Hz

for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        filepath = os.path.join(folder_path, filename)
        try:
            currydata = cr.read(filepath)
            if "data" not in currydata:
                raise ValueError("Missing 'data' key in file.")

            data = currydata["data"]
            labels = currydata.get("labels", [f"Ch{i}" for i in range(data.shape[0])])
            subj_id = filename.split("_")[0]

            if data.shape[0] < data.shape[1]:
                data = data.T

            samples, channels = data.shape
            if samples < min_samples_required:
                continue

            all_subjects_8hz_data[subj_id] = {
                "eeg": data,
                "labels": labels[:channels],
                "fs": 2000,
                "samples": samples,
                "channels": channels
            }
        except Exception:
            continue

# === Time series plot for selected channels ===
subject_id = next(iter(all_subjects_8hz_data))
subject_data = all_subjects_8hz_data[subject_id]
fs = 1000
n_samples = int(700 * fs / 1000)
time_ms = np.arange(n_samples)
target_channels = ["OZ", "O2", "POZ", "PZ", "F2", "F8", "FCZ"]
eeg = subject_data["eeg"]
labels = [lbl.upper().strip() for lbl in subject_data["labels"]]
channel_indices = [labels.index(ch) for ch in target_channels]
eeg_window = detrend(eeg[:n_samples, channel_indices], axis=0)

fig, axes = plt.subplots(len(target_channels), 1, figsize=(14, len(target_channels) * 1.5), sharex=True)
for i, ch in enumerate(target_channels):
    axes[i].plot(time_ms, eeg_window[:, i], color='green', linewidth=1)
    axes[i].set_ylabel(ch, rotation=0, labelpad=25)
    axes[i].grid(True)
axes[-1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.savefig("eeg_8hz_selected_channels.png", dpi=300)
plt.show()

# === Compute PSD across subjects (first 64 channels) ===
fs = 1000
segment_duration_sec = 10
min_samples = fs * segment_duration_sec
psd_list = []
freqs = None
channel_labels = None

for subj_id, data in all_subjects_8hz_data.items():
    eeg = data["eeg"]
    samples = data["samples"]
    if samples < min_samples:
        continue
    ch_limit = min(64, eeg.shape[1])
    segment = eeg[:min_samples, :ch_limit]
    psd_subj = [welch(segment[:, ch], fs=fs, nperseg=8192)[1] for ch in range(ch_limit)]
    if freqs is None:
        freqs = welch(segment[:, 0], fs=fs, nperseg=8192)[0]
        channel_labels = data["labels"][:ch_limit]
    psd_list.append(psd_subj)

psd_all = np.stack(psd_list)
avg_psd = np.mean(psd_all, axis=0)

plt.figure(figsize=(14, 7))
for ch in range(avg_psd.shape[0]):
    plt.semilogy(freqs, avg_psd[ch], label=channel_labels[ch], alpha=0.7)
plt.title("Average EEG PSD across subjects (8 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (V²/Hz)")
plt.xlim(0, 50)
plt.grid(True)
plt.legend(loc="upper right", fontsize="small", ncol=4)
plt.tight_layout()
plt.show()

# === PSD at Oz only ===
psd_oz = []
for subj_id, data in all_subjects_8hz_data.items():
    if data["samples"] < min_samples:
        continue
    labels = [lbl.upper().strip() for lbl in data["labels"]]
    if "OZ" not in labels:
        continue
    ch_idx = labels.index("OZ")
    f, Pxx = welch(data["eeg"][:min_samples, ch_idx], fs=fs, nperseg=8192)
    if freqs is None:
        freqs = f
    psd_oz.append(Pxx)

avg_oz = np.mean(psd_oz, axis=0)
plt.figure(figsize=(12, 6))
for Pxx in psd_oz:
    plt.semilogy(freqs, Pxx, color='gray', alpha=0.3)
plt.axvline(8.0, color='black', linestyle='--')
plt.semilogy(freqs, avg_oz, color='green', linewidth=2)
plt.title("PSD at Oz - 8 Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (μV²/Hz)")
plt.xlim(0, 10)
plt.grid(True)
plt.tight_layout()
plt.savefig("8hz_oz.png", dpi=300)
plt.show()

# === Inject 8 Hz Sine Wave at Oz ===
psd_oz_mod = []
for subj_id, data in all_subjects_8hz_data.items():
    if data["samples"] < min_samples:
        continue
    labels = [lbl.upper().strip() for lbl in data["labels"]]
    if "OZ" not in labels:
        continue
    ch_idx = labels.index("OZ")
    signal = data["eeg"][:min_samples, ch_idx]
    t = np.arange(min_samples) / fs
    amp = np.std(signal) * np.sqrt(2)
    mod = signal + amp * np.sin(2 * np.pi * 8 * t)
    _, Pxx_mod = welch(mod, fs=fs, nperseg=8192)
    psd_oz_mod.append(Pxx_mod)

avg_mod = np.mean(psd_oz_mod, axis=0)
plt.figure(figsize=(12, 6))
plt.semilogy(freqs, avg_oz, label="Original", color='blue')
plt.semilogy(freqs, avg_mod, label="Injected", color='red')
plt.axvline(8.0, color='black', linestyle='--')
plt.title("Oz PSD with 8 Hz Sine Injection")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (μV²/Hz)")
plt.xlim(0, 10)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("8hz_sine_oz.png", dpi=300)
plt.show()

# === T-test on SNR after 8 Hz Injection ===
snr_values = []
for subj_id, data in all_subjects_8hz_data.items():
    if data["samples"] < min_samples:
        continue
    labels = [lbl.upper().strip() for lbl in data["labels"]]
    if "OZ" not in labels:
        continue
    ch_idx = labels.index("OZ")
    signal = data["eeg"][:min_samples, ch_idx]
    t = np.arange(min_samples) / fs
    amp = np.std(signal) * np.sqrt(2)
    mod = signal + amp * np.sin(2 * np.pi * 8 * t)
    f, Pxx = welch(mod, fs=fs, nperseg=8192)
    sig_band = (f >= 7.5) & (f <= 8.5)
    noise_band = ((f >= 6.5) & (f < 7.5)) | ((f > 8.5) & (f <= 9.5))
    snr = np.mean(Pxx[sig_band]) / np.mean(Pxx[noise_band])
    snr_values.append(snr)

log_snr = np.log10(snr_values)
t_stat, p_val = ttest_1samp(log_snr, popmean=0)
print("T-test on log10(SNR) > 0:")
print(f"t = {t_stat:.3f}, p = {p_val:.5f}, mean log10(SNR) = {np.mean(log_snr):.2f}")
