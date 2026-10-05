"""
baseline_threshold.py

TIER 1 BASELINE COMPARISON — for the paper's "why ML over simple rules" argument.
"""

import numpy as np
import pandas as pd

from anomaly_model import FEATURE_COLS, MIN_BASELINE_WINDOWS, PersonalBaselineModel, find_data_file, load_real_wearable_features


class StdDevBoundsBaseline:
    """Classic static-threshold baseline: flag if any feature is outside personal mean ± k*std."""

    def __init__(self, k_sigma=3.0):
        self.k_sigma = k_sigma
        self.baseline_stats = {}

    def fit(self, history_df: pd.DataFrame):
        for state, group in history_df.groupby("activity_state"):
            group = group.dropna(subset=FEATURE_COLS)
            if len(group) < MIN_BASELINE_WINDOWS:
                continue
            self.baseline_stats[state] = {
                f: {"mean": group[f].mean(), "std": max(group[f].std(), 1e-6)} for f in FEATURE_COLS
            }
        return self

    def score(self, new_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for state, group in new_df.groupby("activity_state"):
            group = group.copy()
            stats = self.baseline_stats.get(state)
            if stats is None:
                group["risk_bucket"] = "insufficient_baseline"
                results.append(group)
                continue
            flags = []
            for _, row in group.iterrows():
                out_of_bounds = any(
                    pd.notna(row[f]) and abs(row[f] - stats[f]["mean"]) > self.k_sigma * stats[f]["std"]
                    for f in FEATURE_COLS
                )
                flags.append("flag_for_review" if out_of_bounds else "normal")
            group["risk_bucket"] = flags
            results.append(group)
        return pd.concat(results, ignore_index=True) if results else new_df


class MovingAverageBaseline:
    """Flags a reading if it deviates from the mean of the last `window` readings by more than `threshold_pct`."""

    def __init__(self, window=10, threshold_pct=0.25):
        self.window = window
        self.threshold_pct = threshold_pct

    def score_sequence(self, ordered_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for state, group in ordered_df.groupby("activity_state"):
            group = group.sort_values("timestamp").reset_index(drop=True).copy()
            flags = []
            for i in range(len(group)):
                recent = group["bpm"].iloc[max(0, i - self.window):i]
                if len(recent) < max(3, self.window // 2):
                    flags.append("insufficient_baseline")
                    continue
                recent_avg = recent.mean()
                deviation = abs(group["bpm"].iloc[i] - recent_avg) / max(recent_avg, 1e-6)
                flags.append("flag_for_review" if deviation > self.threshold_pct else "normal")
            group["risk_bucket"] = flags
            results.append(group)
        return pd.concat(results, ignore_index=True) if results else ordered_df


def compare_flagged_rates(history_df: pd.DataFrame, new_df: pd.DataFrame, k_sigma=3.0) -> dict:
    iso = PersonalBaselineModel(contamination=0.05).fit(history_df)
    iso_scored = iso.score(new_df)

    std_baseline = StdDevBoundsBaseline(k_sigma=k_sigma).fit(history_df)
    std_scored = std_baseline.score(new_df)

    def flagged_rate(scored):
        valid = scored[scored["risk_bucket"] != "insufficient_baseline"]
        return (valid["risk_bucket"] != "normal").mean() if len(valid) else float("nan")

    iso_flags = (iso_scored["risk_bucket"] != "normal") & (iso_scored["risk_bucket"] != "insufficient_baseline")
    std_flags = (std_scored["risk_bucket"] != "normal") & (std_scored["risk_bucket"] != "insufficient_baseline")
    both_valid = (iso_scored["risk_bucket"] != "insufficient_baseline") & (std_scored["risk_bucket"] != "insufficient_baseline")

    agreement = (iso_flags == std_flags)[both_valid].mean() if both_valid.any() else float("nan")
    iso_only = (iso_flags & ~std_flags & both_valid).sum()
    std_only = (std_flags & ~iso_flags & both_valid).sum()

    return {
        "isolation_forest_flagged_rate": round(float(flagged_rate(iso_scored)), 4),
        "std_dev_bounds_flagged_rate": round(float(flagged_rate(std_scored)), 4),
        "agreement_rate": round(float(agreement), 4),
        "flagged_by_isolation_forest_only": int(iso_only),
        "flagged_by_std_dev_bounds_only": int(std_only),
        "n_compared": int(both_valid.sum()),
    }


def demonstrate_multivariate_advantage():
    rng = np.random.default_rng(0)
    n = 300
    bpm = rng.normal(65, 4, n)
    hrv_rmssd = 90 - 0.7 * bpm + rng.normal(0, 3, n)
    hrv_sdnn = hrv_rmssd + rng.normal(5, 2, n)
    history = pd.DataFrame({"bpm": bpm, "hrv_rmssd": hrv_rmssd, "hrv_sdnn": hrv_sdnn, "activity_state": "resting"})

    hrv_mean, hrv_std = history["hrv_rmssd"].mean(), history["hrv_rmssd"].std()
    test_bpm = float(history["bpm"].mean())
    test_hrv = hrv_mean - 2.5 * hrv_std
    test_sdnn = test_hrv + 5
    test_row = pd.DataFrame([{"bpm": test_bpm, "hrv_rmssd": test_hrv, "hrv_sdnn": test_sdnn, "activity_state": "resting"}])

    std_baseline = StdDevBoundsBaseline(k_sigma=3.0).fit(history)
    std_result = std_baseline.score(test_row)

    iso = PersonalBaselineModel(contamination=0.05).fit(history)
    iso_result = iso.score(test_row)

    return {
        "history_bpm_mean_std": (round(float(history["bpm"].mean()), 1), round(float(history["bpm"].std()), 1)),
        "history_hrv_rmssd_mean_std": (round(float(hrv_mean), 1), round(float(hrv_std), 1)),
        "hrv_rmssd_marginal_3sigma_bound": (round(float(hrv_mean - 3 * hrv_std), 1), round(float(hrv_mean + 3 * hrv_std), 1)),
        "test_reading": {"bpm": round(test_bpm, 1), "hrv_rmssd": round(float(test_hrv), 1), "hrv_sdnn": round(float(test_sdnn), 1)},
        "std_dev_bounds_verdict": std_result.iloc[0]["risk_bucket"],
        "isolation_forest_verdict": iso_result.iloc[0]["risk_bucket"],
        "note": "test_reading's hrv_rmssd is INSIDE its own 3-sigma marginal bound but is a large residual relative to bpm.",
    }


if __name__ == "__main__":
    print("=== Constructed example: multivariate vs univariate detection ===")
    demo = demonstrate_multivariate_advantage()
    for k, v in demo.items():
        print(f"  {k}: {v}")

    print("\n=== Real-data comparison (requires ppg_wearable_features.csv) ===")
    real_df, rejection_reason = load_real_wearable_features()
    if real_df is None:
        print(f"No usable ppg_wearable_features.csv found. Reason: {rejection_reason}")
        print("Run data_prep/download_preprocess_ppg_features.py first.")
    else:
        from anomaly_model import split_by_subject_time
        train_df, test_df = split_by_subject_time(real_df)
        result = compare_flagged_rates(train_df, test_df)
        for k, v in result.items():
            print(f"  {k}: {v}")
