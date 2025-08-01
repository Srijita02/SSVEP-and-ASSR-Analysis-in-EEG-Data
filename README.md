# SSVEP-and-ASSR-Analysis-in-EEG-Data
Analysis of Visual and Auditory Steady-State Response in Electroencephalography

==============================================================

This repository includes three finalized EEG processing pipelines using Curry .dat files.
Each pipeline performs signal cleaning, PSD computation, and SNR-based significance analysis.

1. 1 Hz SSVEP Pipeline (file: 1hz_ssvep_pipeline.py)
   - Loads EEG recordings and targets the Oz channel.
   - Injects synthetic 1 Hz sine wave for evaluation.
   - Computes power spectral density (PSD) and evaluates log10(SNR).
   - Averages PSD and SNR across all valid subjects.
   - Performs t-test to assess significance of 1 Hz response.

2. 8 Hz SSVEP Pipeline (file: 8hz_ssvep_pipeline.py)
   - Loads data from 8 Hz SSVEP experiments.
   - Computes PSDs at Oz and other channels.
   - Injects 8 Hz sine wave to simulate response.
   - Computes log10(SNR) and tests for significance using a t-test.
   - Averages PSD and SNR values across subjects.

3. 40 Hz ASSR Pipeline (file: 40hz_assr_pipeline.py)
   - Processes EEG data at 2000 Hz sampling rate.
   - Injects 40 Hz synthetic signal at Cz and other central channels.
   - Computes PSD and evaluates signal-to-noise ratio (SNR).
   - Performs one-sample t-test on log10(SNR) to assess statistical significance.
   - All plots and metrics are averaged across valid subjects.

All three scripts are intended for all subjects' analysis of EEG Data.

Dependencies:
-------------
- numpy
- scipy
- matplotlib
- curryreader


