import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = ["mean_rr_ms", "std_rr_ms", "pnn50", "qrs_width_ms"]

ABNORMAL_TYPES = ["PVC-like", "AFib-like", "Bradycardia-like", "Tachycardia-like"]
ALL_CLASSES = ["Normal"] + ABNORMAL_TYPES

RULES = {
    "HR_TACHY": 100.0,
    "HR_BRADY": 50.0,
    "QRS_WIDE": 120.0,
    "QTC_LONG": 500.0,
}

def find_data_file(filename: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(base_dir, filename),
                 os.path.join(base_dir, "..", "data", filename),
                 os.path.join(base_dir, "data_prep", filename),
                 os.path.join(os.getcwd(), filename),
                 os.path.join(os.getcwd(), "data", filename)):
        if os.path.exists(cand):
            return cand
    return os.path.join(base_dir, filename)

DEFAULT_REAL_DATA_PATH = find_data_file("mitbih_features.csv")

def _generate_synthetic_training_data(n_per_class: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spec = {
        "Normal": [(850, 60), (35, 8), (18, 5), (90, 8)],
        "PVC-like": [(820, 120), (90, 20), (12, 6), (140, 15)],
        "AFib-like": [(700, 150), (140, 30), (35, 10), (95, 10)],
        "Bradycardia-like": [(1300, 60), (30, 8), (15, 5), (92, 8)],
        "Tachycardia-like": [(480, 30), (30, 8), (15, 5), (92, 8)],
    }
    rows = []
    for label, params in spec.items():
        for _ in range(n_per_class):
            rows.append({c: float(rng.normal(m, s)) for c, (m, s) in
                         zip(FEATURE_COLS, params)} | {"label": label})
    return pd.DataFrame(rows)

def patient_groups(df: pd.DataFrame) -> pd.Series:
    if "patient_id" in df.columns:
        return df["patient_id"].astype(str)
    if "record_id" in df.columns:
        warnings.warn("patient_id column missing; grouping by record_id (201/202 cannot be merged). "
                      "Rebuild the CSV with data_prep/build_mitbih_features.py.")
        return df["record_id"].astype(str)
    raise ValueError("CSV must contain record_id and patient_id columns "
                     "(build it with data_prep/build_mitbih_features.py).")

class ECGSnapshotClassifier:

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.binary_model = RandomForestClassifier(
            n_estimators=150, random_state=seed, class_weight="balanced")
        self.subtype_model = RandomForestClassifier(
            n_estimators=150, random_state=seed, class_weight="balanced")
        self.multi_model = RandomForestClassifier(
            n_estimators=150, random_state=seed, class_weight="balanced")
        self.is_fitted = False
        self.trained_on = None
        self.train_records_ = None
        self.test_records_ = None

    def fit_on_real_data(self, path=None, test_size=0.25):
        path = path or find_data_file("mitbih_features.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Build it first:\n"
                "  python3 data_prep/build_mitbih_features.py --download "
                "--dl-dir data/mitdb --out data/mitbih_features.csv")
        data = pd.read_csv(path)
        data = data.dropna(subset=FEATURE_COLS + ["label"])
        data = data[data["label"].isin(ALL_CLASSES)]
        groups = patient_groups(data)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                        random_state=self.seed).split(data, groups=groups))
        train_df, test_df = data.iloc[tr], data.iloc[te]
        assert not (set(train_df.patient_id) & set(test_df.patient_id)), "patient leakage!"
        self._fit_from_dataframe(train_df)
        self.train_records_ = sorted(train_df.record_id.unique())
        self.test_records_ = sorted(test_df.record_id.unique())
        return test_df

    def _fit_from_dataframe(self, train_df: pd.DataFrame):
        X = train_df[FEATURE_COLS].values
        y_bin = (train_df["label"] != "Normal").astype(int)
        self.binary_model.fit(X, y_bin)
        abn = train_df[train_df["label"] != "Normal"]
        if abn["label"].nunique() >= 2:
            self.subtype_model.fit(abn[FEATURE_COLS].values, abn["label"].values)
        self.multi_model.fit(X, train_df["label"].values)
        self.is_fitted = True
        self.trained_on = "real"

    def fit_auto(self, path=None, allow_demo_fallback=False):
        try:
            self.fit_on_real_data(path)
            print(f"[ecg_classifier] trained on REAL MIT-BIH windows "
                  f"(patient-grouped split; {len(self.train_records_)} train / "
                  f"{len(self.test_records_)} test recordings)")
        except FileNotFoundError:
            if allow_demo_fallback:
                warnings.warn("Real data missing — training on SYNTHETIC demo data. "
                              "Results from this model must never be reported.")
                self.fit_on_demo_data()
            else:
                raise
        return self

    def fit_on_demo_data(self):
        self._fit_from_dataframe(_generate_synthetic_training_data())
        self.trained_on = "synthetic"
        return self

    def predict_ml(self, X) -> dict:
        X = np.asarray(X, dtype=float)
        p_abn = self.binary_model.predict_proba(X)[:, list(self.binary_model.classes_).index(1)]
        bin_hat = (p_abn >= 0.5).astype(int)
        sub_hat = np.array([None] * len(X), dtype=object)
        sub_p = np.zeros(len(X))
        if hasattr(self.subtype_model, "classes_") and len(getattr(self.subtype_model, "classes_", [])) >= 2:
            idx_abn = np.where(bin_hat == 1)[0]
            if len(idx_abn):
                sp = self.subtype_model.predict_proba(X[idx_abn])
                best = sp.argmax(axis=1)
                sub_hat[idx_abn] = self.subtype_model.classes_[best]
                sub_p[idx_abn] = sp[np.arange(len(best)), best]
        conf = np.where(bin_hat == 1, p_abn * sub_p, 1.0 - p_abn)
        return {"p_abn": p_abn, "bin_hat": bin_hat, "sub_hat": sub_hat,
                "sub_p": sub_p, "conf": conf}

    @staticmethod
    def evaluate_rules(row) -> list:
        hr, qrs, qtc = row.get("mean_hr_bpm"), row.get("qrs_width_ms"), row.get("qtc_ms")
        flags = []
        if pd.notna(hr):
            if hr > RULES["HR_TACHY"]:
                flags.append({"rule": "HR>tachy", "value": float(hr),
                              "threshold": RULES["HR_TACHY"],
                              "label": "Tachycardia-like",
                              "msg": f"Too fast - heart rate {hr:.0f} BPM is above the "
                                     f"{RULES['HR_TACHY']:.0f} BPM limit (Tachycardia-like)"})
            elif hr < RULES["HR_BRADY"]:
                flags.append({"rule": "HR<brady", "value": float(hr),
                              "threshold": RULES["HR_BRADY"],
                              "label": "Bradycardia-like",
                              "msg": f"Too slow - heart rate {hr:.0f} BPM is below the "
                                     f"{RULES['HR_BRADY']:.0f} BPM limit (Bradycardia-like)"})
        if pd.notna(qrs) and qrs >= RULES["QRS_WIDE"]:
            flags.append({"rule": "QRS wide", "value": float(qrs),
                          "threshold": RULES["QRS_WIDE"], "label": "PVC-like",
                          "msg": f"Wide beat - QRS {qrs:.0f} ms is above the "
                                 f"{RULES['QRS_WIDE']:.0f} ms limit (PVC-like)"})
        if qtc is not None and pd.notna(qtc) and qtc >= RULES["QTC_LONG"]:
            flags.append({"rule": "QTc long", "value": float(qtc),
                          "threshold": RULES["QTC_LONG"], "label": "Abnormal",
                          "msg": f"Long QT - QTc {qtc:.0f} ms is above the "
                                 f"{RULES['QTC_LONG']:.0f} ms limit"})
        return flags

    def classify(self, features: dict, apply_rules: bool = True) -> dict:
        if not self.is_fitted:
            self.fit_auto()
        row = {**features, "mean_hr_bpm": features.get("heart_rate_bpm")}
        if not row.get("mean_hr_bpm") and features.get("mean_rr_ms"):
            try:
                row["mean_hr_bpm"] = 60000.0 / float(features["mean_rr_ms"])
            except Exception:
                pass
        X = np.array([[float(features[c]) for c in FEATURE_COLS]])
        ml = self.predict_ml(X)
        ml_abn = bool(ml["bin_hat"][0])
        subtype = None if not ml_abn else (ml["sub_hat"][0] or "Abnormal")
        conf = float(ml["conf"][0])
        flags = self.evaluate_rules(row) if apply_rules else []
        rule_label = flags[0]["label"] if flags else None
        is_abnormal = ml_abn or bool(flags)
        final_subtype = subtype if ml_abn else (rule_label if flags else None)
        return {
            "is_abnormal": bool(is_abnormal),
            "subtype": final_subtype,
            "confidence": conf,
            "confidence_source": "model",
            "ml": {"is_abnormal": ml_abn, "subtype": subtype, "p_abnormal": float(ml["p_abn"][0])},
            "rule_flags": flags,
            "rule_label": rule_label,
            "escalated_by_rules": bool(flags) and not ml_abn,
            "trained_on": self.trained_on,
            "guardrail": ("Fixed safety limits: " + "; ".join(f["msg"] for f in flags)
                          + ". This flag comes from the rule layer, not the AI model.")
                         if flags else None,
        }
