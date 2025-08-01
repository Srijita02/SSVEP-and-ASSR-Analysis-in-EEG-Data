# Auditory Steady-State Response (ASSR) - 40 Hz EEG Analysis Pipeline
# Author: [Your Name or Lab]
# Description: This script performs time series visualization, PSD estimation,
#              sine wave injection, and SNR evaluation for 40 Hz ASSR EEG data.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend, resample
from scipy.stats import ttest_1samp
import curryreader as cr

# === Step 1: Load Curry EEG data ===
folder_path = "40Hz"
all_subjects_data = {}
fs = 2000
min_samples_required = 2 * fs

for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        filepath = os.path.join(folder_path, filename)
        try:
            currydata = cr.read(filepath)
            if "data" not in currydata:
                continue
            data = currydata["data"]
            labels = currydata.get("labels", [f"Ch{i}" for i in range(data.shape[0])])
            subj_id = filename.split("_")[0]

            if data.shape[0] < data.shape[1]:
                data = data.T

            samples, channels = data.shape
            if samples < min_samples_required:
                continue

            all_subjects_data[subj_id] = {
                "eeg": data,
                "labels": labels[:channels],
                "fs": fs,
                "samples": samples,
                "channels": channels
            }
        except Exception:
            continue

# === Step 2: Time Series Plot (Example Subject) ===
subject_id = list(all_subjects_data.keys())[0]
subject_data = all_subjects_data[subject_id]
n_samples = int(fs * 0.7)
time_ms = np.arange(n_samples)
target_channels = ["CZ", "PO8", "OZ"]
labels = [lbl.upper().strip() for lbl in subject_data["labels"]]
channel_indices = [labels.index(ch) for ch in target_channels if ch in labels]

eeg = subject_data["eeg"]
eeg_window = detrend(eeg[:n_samples, channel_indices], axis=0)

fig, axes = plt.subplots(len(channel_indices), 1, figsize=(14, 8), sharex=True)
for i, ch in enumerate(target_channels):
    axes[i].plot(time_ms, eeg_window[:, i], color='navy')
    axes[i].set_ylabel(ch)
    axes[i].grid(True)
axes[-1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.savefig("eeg_40hz_time_series.png", dpi=300)
plt.show()

# === Step 3: PSD Estimation Across Subjects (First 64 Channels) ===
segment_duration_sec = 10
min_samples = segment_duration_sec * fs
psd_list = []
freqs = None
channel_labels = None

for subj_id, data in all_subjects_data.items():
    eeg = data["eeg"]
    samples = data["samples"]
    if samples < min_samples:
        continue
    ch_limit = min(64, eeg.shape[1])
    segment = eeg[:min_samples, :ch_limit]
    psd_subj = [welch(segment[:, ch], fs=fs, nperseg=16384)[1] for ch in range(ch_limit)]
    if freqs is None:
        freqs = welch(segment[:, 0], fs=fs, nperseg=16384)[0]
        channel_labels = data["labels"][:ch_limit]
    psd_list.append(psd_subj)

psd_all = np.stack(psd_list)
avg_psd = np.mean(psd_all, axis=0)

plt.figure(figsize=(14, 7))
for ch in range(avg_psd.shape[0]):
    plt.semilogy(freqs, avg_psd[ch], alpha=0.7)
plt.title("Average PSD across Subjects (40 Hz ASSR)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (\u03bcV\u00b2/Hz)")
plt.xlim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_psd_40hz.png", dpi=300)
plt.show()

# === Step 4: PSD at Cz (Original and Injected 40 Hz) ===
psd_cz = []
psd_cz_mod = []

for subj_id, data in all_subjects_data.items():
    eeg = data["eeg"].copy()
    samples = data["samples"]
    labels = [lbl.upper().strip() for lbl in data["labels"]]
    if samples < min_samples or "CZ" not in labels:
        continue
    cz_idx = labels.index("CZ")
    signal = eeg[:min_samples, cz_idx]
    f, Pxx_orig = welch(signal, fs=fs, nperseg=16384)
    psd_cz.append(Pxx_orig)
    t = np.arange(min_samples) / fs
    sine = np.std(signal) * np.sqrt(2) * np.sin(2 * np.pi * 40 * t)
    Pxx_mod = welch(signal + sine, fs=fs, nperseg=16384)[1]
    psd_cz_mod.append(Pxx_mod)

psd_cz = np.stack(psd_cz)
psd_cz_mod = np.stack(psd_cz_mod)
avg_cz = np.mean(psd_cz, axis=0)
avg_cz_mod = np.mean(psd_cz_mod, axis=0)

plt.figure(figsize=(12, 6))
plt.semilogy(freqs, avg_cz, label="Original", color='blue')
plt.semilogy(freqs, avg_cz_mod, label="Injected", color='red')
plt.axvline(40, color='black', linestyle='--')
plt.title("Cz PSD with and without 40 Hz Injection")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (\u03bcV\u00b2/Hz)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("40hz_injection_cz.png", dpi=300)
plt.show()

# === Step 5: SNR Analysis with One-Sample T-Test ===
snr_values = []
for subj_id, data in all_subjects_data.items():
    eeg = data["eeg"].copy()
    samples = data["samples"]
    labels = [lbl.upper().strip() for lbl in data["labels"]]
    if samples < min_samples or "CZ" not in labels:
        continue
    cz_idx = labels.index("CZ")
    signal = eeg[:min_samples, cz_idx]
    t = np.arange(min_samples) / fs
    sine = np.std(signal) * np.sqrt(2) * np.sin(2 * np.pi * 40 * t)
    modified = signal + sine
    f, Pxx = welch(modified, fs=fs, nperseg=16384)
    sig_band = (f >= 39.5) & (f <= 40.5)
    noise_band = ((f >= 38.5) & (f < 39.5)) | ((f > 40.5) & (f <= 41.5))
    snr = np.mean(Pxx[sig_band]) / np.mean(Pxx[noise_band])
    snr_values.append(snr)

log_snr = np.log10(np.array(snr_values))
t_stat, p_val = ttest_1samp(log_snr, popmean=0)

print("SNR Log10 T-test at 40 Hz (Cz):")
print(f"t = {t_stat:.3f}, p = {p_val:.5f}, mean log10(SNR) = {np.mean(log_snr):.2f}")
