"""
generate_dummy_data.py

Bundled synthetic data for the Streamlit app when no real feature CSVs are
around. Mirrors the generators in web-app/src/App.jsx so both front-ends
show the same shape of sample data:

  make_history()       - 14 days of resting + exercise readings
  make_new_readings()  - 10 fresh readings, 2 of them deliberately odd
  make_ecg_snapshots() - 4 hand-picked ECG-style feature rows
"""

import numpy as np
import pandas as pd


def make_history(days: int = 14, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-08-24 00:00:00")
    rows = []
    for day in range(days):
        day_start = start + pd.Timedelta(days=day)
        for i in range(40):
            rows.append({
                "timestamp": day_start + pd.Timedelta(minutes=30 * i),
                "bpm": rng.normal(66, 5),
                "hrv_rmssd": rng.normal(45, 8),
                "hrv_sdnn": rng.normal(50, 9),
                "activity_state": "resting",
            })
        for i in range(6):
            rows.append({
                "timestamp": day_start + pd.Timedelta(hours=7 + i),
                "bpm": rng.normal(128, 12),
                "hrv_rmssd": rng.normal(18, 5),
                "hrv_sdnn": rng.normal(22, 6),
                "activity_state": "exercise",
            })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def make_new_readings() -> pd.DataFrame:
    base = pd.Timestamp("2026-09-07 08:00:00")
    raw = [
        {"bpm": 65, "hrv_rmssd": 44, "hrv_sdnn": 49, "activity_state": "resting"},
        {"bpm": 68, "hrv_rmssd": 43, "hrv_sdnn": 51, "activity_state": "resting"},
        {"bpm": 63, "hrv_rmssd": 47, "hrv_sdnn": 52, "activity_state": "resting"},
        {"bpm": 128, "hrv_rmssd": 11, "hrv_sdnn": 14, "activity_state": "resting"},   # planted anomaly
        {"bpm": 66, "hrv_rmssd": 45, "hrv_sdnn": 50, "activity_state": "resting"},
        {"bpm": 125, "hrv_rmssd": 19, "hrv_sdnn": 21, "activity_state": "exercise"},
        {"bpm": 132, "hrv_rmssd": 17, "hrv_sdnn": 20, "activity_state": "exercise"},
        {"bpm": 58, "hrv_rmssd": 46, "hrv_sdnn": 48, "activity_state": "exercise"},   # planted anomaly
        {"bpm": 130, "hrv_rmssd": 18, "hrv_sdnn": 22, "activity_state": "exercise"},
        {"bpm": 64, "hrv_rmssd": 46, "hrv_sdnn": 50, "activity_state": "resting"},
    ]
    for i, r in enumerate(raw):
        r["timestamp"] = base + pd.Timedelta(minutes=15 * i)
    return pd.DataFrame(raw)[["timestamp", "bpm", "hrv_rmssd", "hrv_sdnn", "activity_state"]]


def make_ecg_snapshots() -> pd.DataFrame:
    return pd.DataFrame([
        {"label_hint": "Typical reading", "mean_rr_ms": 860, "std_rr_ms": 32, "pnn50": 17, "qrs_width_ms": 88},
        {"label_hint": "Irregular beat sample", "mean_rr_ms": 800, "std_rr_ms": 95, "pnn50": 11, "qrs_width_ms": 145},
        {"label_hint": "Highly irregular sample", "mean_rr_ms": 690, "std_rr_ms": 150, "pnn50": 38, "qrs_width_ms": 93},
        {"label_hint": "Slow, steady sample", "mean_rr_ms": 1210, "std_rr_ms": 28, "pnn50": 14, "qrs_width_ms": 90},
    ])


if __name__ == "__main__":
    h = make_history()
    n = make_new_readings()
    e = make_ecg_snapshots()
    print(f"history: {len(h)} rows, new: {len(n)} rows, ecg snapshots: {len(e)} rows")
