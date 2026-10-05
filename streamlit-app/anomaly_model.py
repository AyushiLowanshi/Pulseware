import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["bpm", "hrv_rmssd", "hrv_sdnn"]
MIN_BASELINE_WINDOWS = 30  # minimum windows needed before baseline is considered reliable


def find_data_file(filename: str) -> str:
    """
    Search for filename in both the current directory and the data_prep/ subfolder.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "data_prep", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "data_prep", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


DEFAULT_REAL_DATA_PATH = find_data_file("ppg_wearable_features.csv")


class DataLoadResult(tuple):
    """
    A backward-compatible result object:
      1. Unpacks cleanly as a 2-tuple: (df, error_reason)
      2. Transparently proxies DataFrame attributes (.columns, len(), .shape, [col], etc.)
         so if callers do `real_ppg_df = load_real_wearable_features()`, it won't crash
         with AttributeError: 'tuple' object has no attribute 'columns'.
    """

    def __new__(cls, df, err):
        return super().__new__(cls, (df, err))

    @property
    def df(self):
        return tuple.__getitem__(self, 0)

    @property
    def err(self):
        return tuple.__getitem__(self, 1)

    def __bool__(self):
        return self.df is not None

    def __len__(self):
        return len(self.df) if self.df is not None else 0

    @property
    def columns(self):
        if self.df is not None:
            return self.df.columns
        return pd.Index([])

    def __getattr__(self, name):
        if self.df is not None:
            return getattr(self.df, name)
        raise AttributeError(f"DataLoadResult: failed to load ({self.err}). No attribute {name!r}.")

    def __getitem__(self, key):
        if self.df is not None and isinstance(key, (str, list)):
            return self.df[key]
        return super().__getitem__(key)


def load_real_wearable_features(path=None):
    """
    Loads real WESAD/PPG-DaLiA-derived features if the CSV exists and is
    usable. Returns DataLoadResult(df, None) on success, or
    DataLoadResult(None, reason_string) on failure.
    """
    if path is None:
        path = find_data_file("ppg_wearable_features.csv")

    if not os.path.exists(path):
        return DataLoadResult(None, f"No file found at '{path}'. Check data_prep/ppg_wearable_features.csv.")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return DataLoadResult(None, f"{path} exists but couldn't be read as a CSV ({type(e).__name__}: {e}).")
    if len(df) == 0:
        return DataLoadResult(None, f"{path} exists but is completely empty (0 rows).")
    missing = [c for c in FEATURE_COLS + ["activity_state"] if c not in df.columns]
    if missing:
        return DataLoadResult(
            None,
            f"{path} exists ({len(df)} rows) but is missing required column(s): {missing}. Found columns: {list(df.columns)}.",
        )
    return DataLoadResult(df, None)


RESTING_GUARDRAILS = {
    "monitor":         {"high": 100, "low": 50},   # >= 100 bpm at rest = tachycardia, <= 50 = bradycardia
    "flag_for_review": {"high": 120, "low": 40},
}
_BUCKET_RANK = {"insufficient_baseline": 0, "normal": 0, "expected_recovery": 0, "during_exercise": 0,
                "monitor": 1, "flag_for_review": 2}
BUCKET_LABELS = {"normal": "Normal", "expected_recovery": "Expected recovery", "during_exercise": "During exercise",
                 "monitor": "Monitor", "flag_for_review": "Flag for review", "insufficient_baseline": "Still learning"}


RECOVERY_WINDOW_MIN = 120          
LATE_RECOVERY_MIN = 180           
LATE_RECOVERY_MONITOR_BPM = 110    
RECOVERY_LIMITS = {"monitor": 120, "flag_for_review": 140}  
RECOVERY_RISE_BPM = 10           


SYMPTOM_OPTIONS = [
    "Chest pain or pressure",
    "Fainting or nearly fainting",
    "Palpitations or fluttering",
    "Dizziness or light-headedness",
    "Breathlessness at rest",
    "Unusual fatigue",
]
SYMPTOM_RULES = {
    "Chest pain or pressure": "flag_for_review",
    "Fainting or nearly fainting": "flag_for_review",
    "Palpitations or fluttering": "monitor",
    "Dizziness or light-headedness": "monitor",
    "Breathlessness at rest": "monitor",
    "Unusual fatigue": "monitor",
}
URGENT_SYMPTOMS = {"Chest pain or pressure", "Fainting or nearly fainting"}


def symptom_escalation(symptoms):
    """(bucket, sentence) for the worst reported symptom, or (None, "")."""
    chosen = [s for s in (symptoms or []) if s in SYMPTOM_RULES]
    if not chosen:
        return None, ""
    worst = max(chosen, key=lambda s: _BUCKET_RANK[SYMPTOM_RULES[s]])
    bucket = SYMPTOM_RULES[worst]
    listed = ", ".join(c.lower() for c in chosen)
    if any(c in URGENT_SYMPTOMS for c in chosen):
        text = (f"You reported {listed}. With symptoms like these, get medical advice promptly "
                f"(emergency services if the chest pain is ongoing) - whatever the numbers say.")
    else:
        text = (f"You reported {listed}. Symptoms count for more than numbers: this is rated at least "
                f"{BUCKET_LABELS[bucket]} because of them, and worth mentioning to a doctor if it keeps happening.")
    return bucket, text


MAX_ASSUMED_WORKOUT_MIN = 180      # when only the end is known, readings up to 3 h before it count as "during"


def apply_recovery_context(df: pd.DataFrame, workout_end=None, workout_start=None):
    """
    Second pass after score(). Given when a workout ended (pandas Timestamp /
    datetime, or None) and optionally when it started:
      * resting rows inside the workout are rated "during_exercise";
      * elevated resting rows in the 2 h after it are rated "expected_recovery"
        (unless high even for recovery, or rising instead of settling);
      * rows still >=100 bpm 2-3 h after it are rated at least "monitor".
    Readings that were already normal are left alone. The reason is recorded
    in a 'context' column. Returns (df, summary) where summary is
    {"n_recovery", "n_exercise", "trend", "first_bpm", "last_bpm", "rising"}.
    """
    df = df.copy()
    if "context" not in df.columns:
        df["context"] = None
    summary = {"n_recovery": 0, "n_exercise": 0, "trend": "", "first_bpm": None, "last_bpm": None, "rising": False}
    if workout_end is None or len(df) == 0 or "timestamp" not in df.columns:
        return df, summary
    workout_end = pd.Timestamp(workout_end)
    workout_start = pd.Timestamp(workout_start) if workout_start is not None else workout_end - pd.Timedelta(minutes=MAX_ASSUMED_WORKOUT_MIN)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    bpm = pd.to_numeric(df["bpm"], errors="coerce")
    resting = df["activity_state"].astype(str).str.lower() == "resting"
    mins = (ts - workout_end).dt.total_seconds() / 60.0
    rank = df["risk_bucket"].map(_BUCKET_RANK).fillna(0)
    elevated = (rank >= _BUCKET_RANK["monitor"]) | (bpm >= RESTING_GUARDRAILS["monitor"]["high"])

    during = resting & (ts >= workout_start) & (ts < workout_end) & bpm.notna()
    for idx in df.index[during]:
        df.at[idx, "risk_bucket"] = "during_exercise"
        df.at[idx, "context"] = (f"{bpm[idx]:.0f} bpm during your workout - an exercise heart rate, "
                                 f"not a resting reading, so it is not judged against resting limits.")
        summary["n_exercise"] += 1

    in_window = resting & elevated & (mins >= 0) & (mins <= RECOVERY_WINDOW_MIN) & bpm.notna()
    late = resting & (mins > RECOVERY_WINDOW_MIN) & (mins <= LATE_RECOVERY_MIN) & bpm.notna()

    for idx in df.index[in_window]:
        m, b = mins[idx], bpm[idx]
        when = f"{m:.0f} min after your workout ended"
        if b >= RECOVERY_LIMITS["flag_for_review"]:
            df.at[idx, "risk_bucket"] = "flag_for_review"
            df.at[idx, "context"] = (f"{b:.0f} bpm {when} is above {RECOVERY_LIMITS['flag_for_review']} bpm - "
                                     f"higher than normal recovery, so it is flagged for review.")
        elif b >= RECOVERY_LIMITS["monitor"]:
            if _BUCKET_RANK.get(df.at[idx, "risk_bucket"], 0) < _BUCKET_RANK["monitor"]:
                df.at[idx, "risk_bucket"] = "monitor"
            elif df.at[idx, "risk_bucket"] == "flag_for_review":
                df.at[idx, "risk_bucket"] = "monitor"  # recovery softens the resting guardrail one step
            df.at[idx, "context"] = (f"{b:.0f} bpm {when} is on the high side even for recovery "
                                     f"(above {RECOVERY_LIMITS['monitor']} bpm), so it is worth monitoring.")
        else:
            df.at[idx, "risk_bucket"] = "expected_recovery"
            df.at[idx, "context"] = (f"{b:.0f} bpm {when}. Heart rate normally stays above resting for one to "
                                     f"two hours after exercise, so this is expected recovery, not a resting reading.")
            summary["n_recovery"] += 1

    for idx in df.index[late]:
        m, b = mins[idx], bpm[idx]
        if b >= LATE_RECOVERY_MONITOR_BPM:
            if _BUCKET_RANK.get(df.at[idx, "risk_bucket"], 0) < _BUCKET_RANK["monitor"]:
                df.at[idx, "risk_bucket"] = "monitor"
            df.at[idx, "context"] = (f"Still {b:.0f} bpm {m / 60:.1f} hours after your workout ended. Recovery usually "
                                     f"settles within two hours, so this is worth monitoring.")
        elif elevated[idx]:
            df.at[idx, "risk_bucket"] = "expected_recovery"
            df.at[idx, "context"] = (f"{b:.0f} bpm {m / 60:.1f} hours after your workout ended - still mildly elevated, which is "
                                     f"common after long or hot sessions. It should settle within a few hours; if it doesn't, "
                                     f"that is worth monitoring.")
            summary["n_recovery"] += 1

    rec = df[(df["risk_bucket"] == "expected_recovery") & (in_window | late)].copy()
    if len(rec) >= 2:
        rec = rec.assign(_t=ts[rec.index]).sort_values("_t")
        first_b, last_b = float(rec["bpm"].iloc[0]), float(rec["bpm"].iloc[-1])
        summary.update(first_bpm=first_b, last_bpm=last_b)
        if last_b - first_b >= RECOVERY_RISE_BPM:
            summary["rising"] = True
            summary["trend"] = f"rising from {first_b:.0f} to {last_b:.0f} bpm instead of settling"
            last_idx = rec.index[-1]
            df.at[last_idx, "risk_bucket"] = "monitor"
            df.at[last_idx, "context"] = (f"{last_b:.0f} bpm {mins[last_idx]:.0f} min after your workout, and rising "
                                          f"(from {first_b:.0f} bpm earlier) instead of settling - worth monitoring.")
            summary["n_recovery"] -= 1
        elif last_b <= float(rec["bpm"].max()) - 3:
            summary["trend"] = f"settling from {rec['bpm'].max():.0f} to {last_b:.0f} bpm"
        else:
            summary["trend"] = f"holding around {rec['bpm'].mean():.0f} bpm"
    return df, summary


def apply_resting_guardrails(df: pd.DataFrame) -> pd.DataFrame:
    """Escalates risk_bucket for resting rows outside RESTING_GUARDRAILS; records why in a 'guardrail' column."""
    df = df.copy()
    df["guardrail"] = None
    if "activity_state" not in df.columns or "bpm" not in df.columns or len(df) == 0:
        return df
    resting = df["activity_state"].astype(str).str.lower() == "resting"
    bpm = pd.to_numeric(df["bpm"], errors="coerce")
    for bucket in ("monitor", "flag_for_review"):  # stricter level last so it wins
        lim = RESTING_GUARDRAILS[bucket]
        for mask, note in (
            (resting & (bpm >= lim["high"]), f"at or above the {lim['high']} bpm resting limit"),
            (resting & (bpm <= lim["low"]), f"at or below the {lim['low']} bpm resting limit"),
        ):
            if not mask.any():
                continue
            df.loc[mask, "guardrail"] = note
            upgrade = mask & (df["risk_bucket"].map(_BUCKET_RANK).fillna(0) < _BUCKET_RANK[bucket])
            df.loc[upgrade, "risk_bucket"] = bucket
    return df


class PersonalBaselineModel:
    """One IsolationForest per activity_state (e.g. 'resting', 'exercise')."""

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.models: dict = {}
        self.scalers: dict = {}
        self.baseline_stats: dict = {}

    def fit(self, history_df: pd.DataFrame):
        for state, group in history_df.groupby("activity_state"):
            group = group.dropna(subset=FEATURE_COLS)
            if len(group) < MIN_BASELINE_WINDOWS:
                continue

            X = group[FEATURE_COLS].values
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            model = IsolationForest(contamination=self.contamination, n_estimators=200, random_state=42)
            model.fit(X_scaled)

            self.models[state] = model
            self.scalers[state] = scaler
            self.baseline_stats[state] = {
                "bpm_mean": group["bpm"].mean(),
                "bpm_std": group["bpm"].std(),
                "hrv_rmssd_mean": group["hrv_rmssd"].mean(),
                "n_windows": len(group),
            }
        return self

    def score(self, new_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for state, group in new_df.groupby("activity_state"):
            group = group.copy()

            if state not in self.models:
                group["anomaly_score"] = np.nan
                group["risk_bucket"] = "insufficient_baseline"
                results.append(group)
                continue

            X = group[FEATURE_COLS].fillna(group[FEATURE_COLS].mean())
            # A screenshot or manual entry usually carries bpm only. If a whole
            # column is still empty after the batch-mean fill, use that feature's
            # baseline mean (neutral) instead of NaN - otherwise IsolationForest
            # treats the missing values as unremarkable and calls every row
            # "normal", even 125 bpm at rest.
            X = X.fillna(pd.Series(self.scalers[state].mean_, index=FEATURE_COLS)).values
            X_scaled = self.scalers[state].transform(X)

            raw_scores = self.models[state].decision_function(X_scaled)
            predictions = self.models[state].predict(X_scaled)  # -1 = anomaly, 1 = normal

            group["anomaly_score"] = raw_scores
            group["risk_bucket"] = [self._bucket(s, p) for s, p in zip(raw_scores, predictions)]
            results.append(group)

        scored = pd.concat(results, ignore_index=True) if results else new_df.copy()
        return apply_resting_guardrails(scored)

    @staticmethod
    def _bucket(score: float, prediction: int) -> str:
        if prediction == 1:
            return "normal"
        elif score > -0.15:
            return "monitor"
        return "flag_for_review"

    def explain(self, row: pd.Series) -> str:
        state = row["activity_state"]
        stats = self.baseline_stats.get(state)
        context = row.get("context") if hasattr(row, "get") else None
        if isinstance(context, str) and context:
            return context  # the recovery pass already said what matters
        guard = row.get("guardrail") if hasattr(row, "get") else None
        guard_txt = (
            f"{row['bpm']:.0f} bpm while resting is {guard} (the standard tachycardia/bradycardia cut-off), "
            f"so it is rated {BUCKET_LABELS.get(row['risk_bucket'], row['risk_bucket'])} regardless of the baseline."
        ) if guard else ""

        if stats is None:
            base = "Not enough historical data yet to establish a personal baseline for this activity."
            return f"{base} {guard_txt}".strip()

        bpm_diff = row["bpm"] - stats["bpm_mean"]
        direction = "higher" if bpm_diff > 0 else "lower"

        if row["risk_bucket"] == "normal":
            return f"Within the typical {state} range."
        if guard and abs(bpm_diff) < 3 * (stats.get("bpm_std") or 0):
            # Inside the (wide) history, but past an absolute limit: say that plainly.
            return f"Inside the range of the baseline history, but {guard_txt}"
        return (
            f"{abs(bpm_diff):.0f} bpm {direction} than the usual {state} average "
            f"({stats['bpm_mean']:.0f} bpm). Rated {BUCKET_LABELS.get(row['risk_bucket'], row['risk_bucket'])}. "
            f"Not a diagnosis - a persistent pattern like this is worth discussing with a doctor. {guard_txt}"
        ).strip()


def evaluate_qualitatively(model: "PersonalBaselineModel", test_df: pd.DataFrame) -> dict:
    scored = model.score(test_df)
    report = {}
    for state, group in scored.groupby("activity_state"):
        valid = group[group["risk_bucket"] != "insufficient_baseline"]
        if len(valid) == 0:
            continue
        flagged_rate = (valid["risk_bucket"] != "normal").mean()
        report[state] = {
            "n_test_windows": len(valid),
            "flagged_rate": round(float(flagged_rate), 4),
            "expected_contamination": model.contamination,
            "sane": abs(flagged_rate - model.contamination) < 0.15,
        }
    return report


def split_by_subject_time(df: pd.DataFrame, test_frac: float = 0.3):
    train_parts, test_parts = [], []
    subject_col = "subject_id" if "subject_id" in df.columns else None
    groups = df.groupby(subject_col) if subject_col else [(None, df)]
    for _, g in groups:
        g = g.sort_values("window_start_s") if "window_start_s" in g.columns else g
        split_idx = int(len(g) * (1 - test_frac))
        train_parts.append(g.iloc[:split_idx])
        test_parts.append(g.iloc[split_idx:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


if __name__ == "__main__":
    real_df, rejection_reason = load_real_wearable_features()
    if real_df is None:
        print(f"No usable real feature file found. Reason: {rejection_reason}")
        print("Run: python data_prep/download_preprocess_ppg_features.py")
    else:
        path_used = find_data_file("ppg_wearable_features.csv")
        print(f"Loaded {len(real_df)} real feature rows from {path_used}")
        train_df, test_df = split_by_subject_time(real_df)
        print(f"Per-subject time-based split: {len(train_df)} train windows, {len(test_df)} test windows")

        model = PersonalBaselineModel(contamination=0.05).fit(train_df)
        report = evaluate_qualitatively(model, test_df)
        print("\nQualitative sanity check:")
        for state, stats in report.items():
            flag = "✅" if stats["sane"] else "⚠️ "
            print(
                f"  {flag} {state}: flagged {stats['flagged_rate']:.1%} of {stats['n_test_windows']} "
                f"held-out windows (target ~{stats['expected_contamination']:.0%})"
            )
