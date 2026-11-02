import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, "ecg_classifier.py")):
    sys.path.insert(0, HERE)
from ecg_classifier import ECGSnapshotClassifier, FEATURE_COLS, RULES

rng_global = np.random.default_rng(42)

def rule_only_label(row) -> str:
    if pd.notna(row["mean_hr_bpm"]) and row["mean_hr_bpm"] < RULES["HR_BRADY"]:
        return "Bradycardia-like"
    if pd.notna(row["mean_hr_bpm"]) and row["mean_hr_bpm"] > RULES["HR_TACHY"]:
        return "Tachycardia-like"
    if pd.notna(row["qrs_width_ms"]) and row["qrs_width_ms"] >= RULES["QRS_WIDE"]:
        return "PVC-like"
    return "Normal"

def binary_metrics(y_true, y_pred, p_abn):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    out = {"accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
           "sensitivity": tp / max(1, tp + fn), "specificity": tn / max(1, tn + fp),
           "precision": tp / max(1, tp + fp),
           "f1_abnormal": 2 * tp / max(1, 2 * tp + fp + fn),
           "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}
    try:
        out["auc"] = float(roc_auc_score(y_true, p_abn)) if len(set(y_true)) > 1 else None
    except Exception:
        out["auc"] = None
    return out

def cluster_bootstrap_ci(df, B=2000, seed=42):
    patients = df.patient_id.unique()
    rng = np.random.default_rng(seed)
    keys = {"accuracy": [], "sensitivity": [], "specificity": [], "auc": []}
    for _ in range(B):
        pick = rng.choice(patients, size=len(patients), replace=True)
        parts = [df[df.patient_id == p] for p in pick]
        b = pd.concat(parts)
        y, yh, p = b.y_true.values, b.y_pred.values, b.p_abn.values
        tp = ((y == 1) & (yh == 1)).sum(); tn = ((y == 0) & (yh == 0)).sum()
        fp = ((y == 0) & (yh == 1)).sum(); fn = ((y == 1) & (yh == 0)).sum()
        keys["accuracy"].append((tp + tn) / len(b))
        keys["sensitivity"].append(tp / max(1, tp + fn))
        keys["specificity"].append(tn / max(1, tn + fp))
        if len(set(y)) > 1:
            try:
                keys["auc"].append(roc_auc_score(y, p))
            except Exception:
                pass
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in keys.items() if v}

def run(csv_path, k=5):
    df = pd.read_csv(csv_path).dropna(subset=FEATURE_COLS + ["label"])
    groups = df.patient_id.astype(str)
    print(f"{len(df)} windows | {df.record_id.nunique()} records | "
          f"{groups.nunique()} patients | classes: {df.label.value_counts().to_dict()}")

    oof = []
    gkf = GroupKFold(n_splits=k)
    for fold, (tr, te) in enumerate(gkf.split(df, groups=groups), 1):
        clf = ECGSnapshotClassifier(seed=42 + fold)
        clf._fit_from_dataframe(df.iloc[tr])
        test = df.iloc[te]
        ml = clf.predict_ml(test[FEATURE_COLS].values)
        rules = test.apply(rule_only_label, axis=1)
        for i, (_, row) in enumerate(test.iterrows()):
            ml_abn = bool(ml["bin_hat"][i])
            r_lab = rules.iloc[i]
            oof.append({
                "patient_id": row.patient_id, "record_id": row.record_id,
                "y": row.label, "y_bin": int(row.label != "Normal"),
                "ml_bin": ml_abn, "ml_sub": ml["sub_hat"][i], "p_abn": float(ml["p_abn"][i]),
                "rule_label": r_lab,
                "comb_bin": ml_abn or (r_lab != "Normal"),
                "comb_sub": (ml["sub_hat"][i] if ml_abn else (r_lab if r_lab != "Normal" else None)),
                "single_bin": int(clf.multi_model.predict(
                    test[FEATURE_COLS].values[[i]])[0] != "Normal"),
            })
        print(f"fold {fold}/{k} done (train patients "
              f"{groups.iloc[tr].nunique()}, test patients {groups.iloc[te].nunique()})")

    oof = pd.DataFrame(oof)
    res = {"n_windows": len(oof), "n_patients": int(oof.patient_id.nunique())}

    for name, ycol, pcol in [("ml_only", "ml_bin", "p_abn"),
                             ("ml_plus_rules", "comb_bin", "p_abn")]:
        res[name] = binary_metrics(oof.y_bin, oof[ycol], oof[pcol])
    res["rules_only"] = binary_metrics(oof.y_bin, (oof.rule_label != "Normal").astype(int),
                                       oof.rule_label.eq("Normal").map({True: 0, False: 1}))
    res["single_multiclass_rf"] = binary_metrics(oof.y_bin, oof.single_bin, oof.p_abn)

    abn = oof[oof.y_bin == 1].copy()
    abn["ml_sub"] = abn.ml_subfillna if False else abn.ml_sub
    mask = abn.ml_sub.notna()
    from sklearn.metrics import f1_score, classification_report
    res["subtype_macro_f1"] = float(f1_score(abn.y[mask], abn.ml_sub[mask], average="macro"))
    res["subtype_report"] = classification_report(abn.y[mask], abn.ml_sub[mask], zero_division=0)

    boot = oof.rename(columns={"y_bin": "y_true", "ml_bin": "y_pred"})
    res["ml_only_ci95_patients"] = cluster_bootstrap_ci(boot)
    boot2 = oof.rename(columns={"y_bin": "y_true", "comb_bin": "y_pred"})
    res["ml_plus_rules_ci95_patients"] = cluster_bootstrap_ci(boot2)

    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("subtype_report",)}, indent=2, default=str))
    print("\nSubtype report (true-abnormal windows):\n", res["subtype_report"])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tier2_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"saved -> results/tier2_results.json")
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../data/mitbih_features.csv")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    run(a.csv, a.k)
