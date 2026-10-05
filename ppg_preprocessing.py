"""
ppg_preprocessing.py
Very simple live cleaning for wrist PPG - the 4 steps from the paper.

1. Bandpass filter 0.5-5 Hz (4th order Butterworth) - keeps only heart beats
2. Find peaks with 250 ms gap - so two beats can't be too close
3. Calculate RR intervals - time between beats
4. Check accelerometer - if moving too much (RMS > 0.05), throw window away
"""

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt, find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def clean_ppg(raw_ppg, fs=64):
    """
    Step 1: Clean the raw PPG signal.
    raw_ppg: list of numbers from the light sensor
    fs: how many times per second we measure (64 Hz is common)
    Returns clean signal.
    """
    if not SCIPY_AVAILABLE:
        return np.array(raw_ppg)  # no scipy, return as is
    # 4th order = sharp cut, not too slow. 0.5 Hz = 30 BPM, 5 Hz = 300 BPM
    b, a = butter(4, [0.5, 5.0], btype='band', fs=fs)
    clean = filtfilt(b, a, raw_ppg)
    return clean


def find_heartbeats(clean_ppg, fs=64):
    """
    Step 2: Find each heartbeat peak. 250 ms rule = heart needs time to fill.
    """
    if not SCIPY_AVAILABLE:
        return np.array([]), np.array([])
    distance = int(0.25 * fs)  # 250 ms = 0.25 seconds
    peaks, _ = find_peaks(clean_ppg, distance=distance)
    # RR intervals = time between peaks in milliseconds
    if len(peaks) > 1:
        rr_ms = np.diff(peaks) / fs * 1000
    else:
        rr_ms = np.array([])
    return peaks, rr_ms


def is_noisy_window(acc_x, acc_y, acc_z):
    """
    Step 3 & 4: Check if person is moving too much.
    If RMS > 0.05 g^2, signal is unreliable -> skip this window.
    Returns True if noisy (should be removed).
    """
    acc_x = np.array(acc_x)
    acc_y = np.array(acc_y)
    acc_z = np.array(acc_z)
    # RMS = shake amount
    rms = np.sqrt(np.mean(acc_x**2 + acc_y**2 + acc_z**2))
    return rms > 0.05


def preprocess_dataframe(df, ppg_col="ppg", acc_cols=("acc_x", "acc_y", "acc_z"), fs=64, window_s=10):
    """
    Simple helper: takes a DataFrame with raw ppg + accelerometer,
    cleans it window by window, removes noisy windows,
    and returns a clean DataFrame with bpm, rr intervals.

    df: DataFrame with columns ppg, acc_x, acc_y, acc_z
    fs: sampling rate
    window_s: window length in seconds (10s = standard ECG)
    """
    if ppg_col not in df.columns:
        return df  # not raw data, return as is

    if not SCIPY_AVAILABLE:
        # without scipy we can't filter, just return
        return df

    # If no accelerometer columns, just clean whole signal
    has_acc = all(c in df.columns for c in acc_cols)

    clean_rows = []
    window_len = int(fs * window_s)

    for start in range(0, len(df), window_len):
        window = df.iloc[start:start+window_len]
        if len(window) < window_len * 0.5:
            continue  # window too short, skip

        # Check movement first
        if has_acc:
            if is_noisy_window(window[acc_cols[0]], window[acc_cols[1]], window[acc_cols[2]]):
                continue  # noisy -> throw away, don't add to baseline

        raw = window[ppg_col].values
        clean = clean_ppg(raw, fs=fs)
        peaks, rr_ms = find_heartbeats(clean, fs=fs)

        if len(rr_ms) == 0:
            continue

        # Simple features from RR intervals
        bpm = 60000 / np.mean(rr_ms) if np.mean(rr_ms) > 0 else None
        hrv_sdnn = np.std(rr_ms) if len(rr_ms) > 1 else None

        # Keep one row per window with clean features
        clean_rows.append({
            "timestamp": window.iloc[0].get("timestamp", start),
            "bpm": bpm,
            "hrv_sdnn": hrv_sdnn,
            "rr_mean_ms": np.mean(rr_ms),
            "rr_std_ms": np.std(rr_ms),
            "n_beats": len(peaks),
            "window_start": start
        })

    if not clean_rows:
        # All windows were noisy -> return empty clean table (0 rows), not the noisy raw data
        return pd.DataFrame(columns=["timestamp", "bpm", "hrv_sdnn", "rr_mean_ms", "rr_std_ms", "n_beats", "window_start"])

    return pd.DataFrame(clean_rows)
