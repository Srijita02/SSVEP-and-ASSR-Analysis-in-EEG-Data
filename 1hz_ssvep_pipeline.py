# SSVEP EEG Analysis Pipeline

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, resample, detrend
import curryreader as cr

# --- Load Curry Data ---
folder_path = "1hz_2nd"
fs = 1000
min_samples_required = 2 * fs
all_subjects_data = {}

for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        filepath = os.path.join(folder_path, filename)
        subj_id = filename.split("_")[0]
        try:
            currydata = cr.read(filepath)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                filtered = []
                skip = False
                for line in lines:
                    if "IMPEDANCE_VALUES START" in line or "TRIGGER_FLAGS_OTHERS START" in line:
                        skip = True
                    if skip and "END" in line:
                        skip = False
                        continue
                    if not skip:
                        filtered.append(line)
                temp_file = filepath + ".tmp"
                with open(temp_file, "w") as f:
                    f.writelines(filtered)
                currydata = cr.read(temp_file)
                os.remove(temp_file)
            else:
                continue

        if "data" not in currydata:
            continue
        data = currydata["data"]
        labels = currydata.get("labels", [f"Ch{i}" for i in range(data.shape[0])])
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
print(f"Total valid subjects: {len(all_subjects_data)}")

# --- Time Series Plot ---
labels = [lbl.strip().upper() for lbl in currydata['labels']]
data = currydata['data']
fs = 2000
n_trials = 97
n_samples_per_trial = 1401
time_ms = np.arange(n_samples_per_trial) / fs * 1000

data_reshaped = data.reshape((n_trials, n_samples_per_trial, -1))
trial_data = data_reshaped[0]
target_channels = ["OZ", "O2", "POZ", "PZ", "F2", "F8", "FCZ"]
channel_indices = [labels.index(ch) for ch in target_channels if ch in labels]
channel_labels = [labels[i] for i in channel_indices]

fig, axes = plt.subplots(len(channel_indices), 1, figsize=(12, len(channel_indices)*2), sharex=True)
for i, ch_idx in enumerate(channel_indices):
    ax = axes[i]
    ax.plot(time_ms, trial_data[:, ch_idx], color='darkred', linewidth=0.8)
    ax.set_ylabel(channel_labels[i], rotation=0, labelpad=30, fontsize=9)
    ax.grid(True)
    if i != len(channel_indices) - 1:
        ax.set_xticks([])
axes[-1].set_xlabel("Time (ms)", fontsize=9)
plt.tight_layout()
plt.savefig("1hz_time_series.png", dpi=300)
plt.show()

# --- Downsampled PSD ---
fs_down = 250
psd_list = []
for subj_id, data in all_subjects_data.items():
    eeg = data["eeg"]
    if data["samples"] < 10000:
        continue
    segment = eeg[:10000, :min(64, eeg.shape[1])]
    segment_ds = resample(segment, int(2500), axis=0)
    psd_subj = [welch(segment_ds[:, ch], fs=fs_down, nperseg=1024)[1] for ch in range(segment_ds.shape[1])]
    psd_list.append(psd_subj)

psd_all = np.stack(psd_list, axis=0)
avg_psd = np.mean(psd_all, axis=0)
freqs = welch(segment_ds[:, 0], fs=fs_down, nperseg=1024)[0]

plt.figure(figsize=(14, 7))
for ch in range(avg_psd.shape[0]):
    plt.semilogy(freqs, avg_psd[ch], alpha=0.6)
plt.axvline(1.0, linestyle='--', color='black')
plt.title("1 Hz SSVEP – PSD (Downsampled to 250 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (µV²/Hz)")
plt.xlim(0, 10)
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_psd_1hz.png", dpi=300)
plt.show()

# --- Original vs. Injected PSD at Oz ---
psd_oz_list, psd_oz_modified_list = [], []
for data in all_subjects_data.values():
    if data["samples"] < 10000:
        continue
    labels = [lbl.strip().upper() for lbl in data["labels"]]
    if "OZ" not in labels:
        continue
    idx = labels.index("OZ")
    signal = data["eeg"][:10000, idx]
    t = np.arange(10000) / fs
    amp = np.std(signal) * np.sqrt(2)
    injected = signal + amp * np.sin(2 * np.pi * 1 * t)
    f, Pxx = welch(signal, fs=fs, nperseg=8192)
    _, Pxx_mod = welch(injected, fs=fs, nperseg=8192)
    psd_oz_list.append(Pxx)
    psd_oz_modified_list.append(Pxx_mod)

avg_orig = np.mean(np.stack(psd_oz_list), axis=0)
avg_mod = np.mean(np.stack(psd_oz_modified_list), axis=0)

plt.figure(figsize=(12, 6))
plt.semilogy(f, avg_orig, label="Original", color='blue')
plt.semilogy(f, avg_mod, label="Injected 1 Hz", color='red')
plt.axvline(1.0, linestyle='--', color='black')
plt.title("PSD at Oz – Original vs Injected")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (µV²/Hz)")
plt.xlim(0, 10)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("1hz_sine_oz.png", dpi=300)
plt.show()

# --- PSD Subplots at Occipital Channels ---
occipital_channels = ["OZ", "O1", "O2", "POZ", "PO3", "PO4"]
psd_dict = {ch: [] for ch in occipital_channels}
freqs = None
for data in all_subjects_data.values():
    if data["samples"] < 10000:
        continue
    labels = [lbl.strip().upper() for lbl in data["labels"]]
    label_map = {lbl: i for i, lbl in enumerate(labels)}
    for ch in occipital_channels:
        if ch in label_map:
            seg = data["eeg"][:10000, label_map[ch]]
            f, Pxx = welch(seg, fs=fs, nperseg=8192)
            psd_dict[ch].append(Pxx)

fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey=True)
axes = axes.flatten()
for i, ch in enumerate(occipital_channels):
    ax = axes[i]
    psds = psd_dict[ch]
    if psds:
        psd_all = np.stack(psds, axis=0)
        avg_psd = np.mean(psd_all, axis=0)
        for p in psd_all:
            ax.semilogy(f, p, color='gray', alpha=0.3)
        ax.semilogy(f, avg_psd, color='darkgreen', linewidth=2)
        ax.axvline(1.0, linestyle='--', color='black')
    ax.set_title(f"{ch}")
    ax.grid(True)
axes[0].legend(["Subjects", "Average"], fontsize=8)
plt.suptitle("1 Hz PSD at Occipital Channels")
plt.tight_layout()
plt.savefig("1hz_occipital_subplots.png", dpi=300)
plt.show()
