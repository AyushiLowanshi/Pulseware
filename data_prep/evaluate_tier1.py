#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

FEATURE_SETS = {"hr_only": ["bpm"], "hr_rmssd": ["bpm", "hrv_rmssd"],
                "full": ["bpm", "hrv_rmssd", "hrv_sdnn"]}
MODELS = ["iforest", "ksd", "ocsvm", "lof"]
WINDOWS_PER_DAY = 2 * 24 * 60
MIN_TRAIN, MIN_TEST = 20, 10
FLAG_RATE = 0.05

def subject_key(df):
    return df["dataset"].astype(str) + ":" + df["subject_id"].astype(str)

def fit_scores(name, X_train, X_test, contamination=0.05, seed=42):
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Ztr, Zte = (X_train - mu) / sd, (X_test - mu) / sd
    if name == "iforest":
        m = IsolationForest(contamination=contamination, n_estimators=200,
                            random_state=seed).fit(Ztr)
        return -m.score_samples(Ztr), -m.score_samples(Zte)
    if name == "ocsvm":
        m = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale").fit(Ztr)
        return (-m.decision_function(Ztr).ravel(),
                -m.decision_function(Zte).ravel())
    if name == "lof":
        m = LocalOutlierFactor(n_neighbors=20, contamination=contamination,
                               novelty=True).fit(Ztr)
        return -m.score_samples(Ztr), -m.score_samples(Zte)
    if name == "ksd":

        return np.abs(Ztr).max(axis=1), np.abs(Zte).max(axis=1)
    raise ValueError(name)

def metrics_of(y_true, pred):
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    n = max(1, tp + fp + fn + tn)
    return {"n": n, "positives": tp + fn,
            "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "false_alerts_per_day": round((tp + fp) / n * WINDOWS_PER_DAY, 1)}

def sens_stat(df):
    df = df[(df.y == 1)]
    if len(df) == 0:
        return None
    return (df.pred == 1).mean()

def subject_bootstrap_ci(rows, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rows)
    subs = frame.subject.unique()
    vals = []
    for _ in range(B):
        pick = rng.choice(subs, size=len(subs), replace=True)
        parts = pd.concat([frame[frame.subject == p] for p in pick])
        v = sens_stat(parts)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]

def run(csv_path, out_dir="results", motion_gate=0.05):
    df = pd.read_csv(csv_path)
    df["subject"] = subject_key(df)
    if motion_gate > 0:
        gate = df.acc_rms_g.fillna(0) <= motion_gate
    else:
        gate = pd.Series(True, index=df.index)
    kept = df[gate]
    print(f"loaded {len(df)} windows | {df.subject.nunique()} subjects "
          f"(dataset:subject) | motion gate {motion_gate} g excludes "
          f"{int((~gate).sum())} | scoring on {len(kept)} kept windows")

    ok = df.dropna(subset=["bpm", "ecg_hr_ref"]).copy()
    ok["abs_err"] = (ok.bpm - ok.ecg_hr_ref).abs()
    per_subject = ok.groupby("subject").apply(
        lambda g: pd.Series({"n": len(g),
                             "median_err": round(g.abs_err.median(), 2),
                             "mean_err": round(g.abs_err.mean(), 2),
                             "corr": round(g.bpm.corr(g.ecg_hr_ref), 3)
                             if len(g) > 2 else np.nan}))
    per_condition = ok.groupby(["dataset", "condition"]).apply(
        lambda g: pd.Series({"n": len(g),
                             "median_err": round(g.abs_err.median(), 2),
                             "p90_err": round(g.abs_err.quantile(0.9), 2)}))

    thr_grid = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    rej_curve = df.groupby("dataset").apply(
        lambda g: pd.Series({f"rej>{t}g": round(float((g.acc_rms_g > t).mean()), 3)
                             for t in thr_grid}))
    rej_condition = df.groupby(["dataset", "condition"]).apply(
        lambda g: pd.Series({f"rej>{t}g": round(float((g.acc_rms_g > t).mean()), 3)
                             for t in thr_grid}))

    records = []
    skipped = []
    for s, g in kept.groupby("subject"):
        g = g.sort_values("t_start_s").reset_index(drop=True)
        cut = int(len(g) * 0.7)
        tr, te = g.iloc[:cut], g.iloc[cut:]
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            skipped.append((s, len(tr), len(te)))
            continue
        hr_m, hr_s = tr.ecg_hr_ref.mean(), tr.ecg_hr_ref.std()
        rm_m, rm_s = tr.ecg_rmssd_ref.mean(), tr.ecg_rmssd_ref.std()
        y = (((te.ecg_hr_ref - hr_m).abs() > 2 * max(hr_s, 1e-6)) |
             ((te.ecg_rmssd_ref - rm_m).abs() > 2 * max(rm_s, 1e-6))).astype(int).values
        for fs_name, cols in FEATURE_SETS.items():
            Xtr = tr[cols].values.astype(float)
            Xte = te[cols].values.astype(float)
            m_tr, m_te = np.isfinite(Xtr).all(axis=1), np.isfinite(Xte).all(axis=1)
            Xtr, Xte, y_f = Xtr[m_tr], Xte[m_te], y[m_te]
            if len(Xtr) < MIN_TRAIN or len(Xte) < MIN_TEST or len(set(y_f)) < 2:
                continue
            models = MODELS if fs_name == "full" else ["iforest"]
            for mname in models:
                _, ste = fit_scores(mname, Xtr, Xte)
                thr = np.quantile(ste, 1 - FLAG_RATE)
                pred = (ste >= thr).astype(int)
                index_te = te.index[m_te]
                def emit(mname_, pred_):
                    for i, wi in enumerate(index_te):
                        records.append({"subject": s, "dataset": te.at[wi, "dataset"],
                                        "condition": te.at[wi, "condition"],
                                        "y": int(y_f[i]), "model": mname_,
                                        "fs": fs_name, "pred": int(pred_[i])})
                emit(mname, pred)
                if mname == "ksd" and fs_name == "full":

                    emit("ksd2_native", (ste > 2.0).astype(int))
                    emit("ksd3_native", (ste > 3.0).astype(int))
    if skipped:
        print(f"skipped {len(skipped)} subjects (too few windows): "
              + ", ".join(f"{s}" for s, _, _ in skipped[:8]) + " ...")
    R = pd.DataFrame(records)
    print(f"protocol produced {len(R):,} scored window-model pairs")

    headline = {}
    for mname in MODELS + ["ksd2_native", "ksd3_native"]:
        sub = R[(R.model == mname) & (R.fs == "full")]
        if not len(sub):
            continue
        row = metrics_of(sub.y.values, sub.pred.values)
        row["n_subjects"] = int(sub.subject.nunique())
        row["sens_ci95_subject_boot"] = subject_bootstrap_ci(sub.to_dict("records"))
        headline[mname] = row
    print("\n=== HEADLINE (full features, matched 5% flag rate, ECG reference) ===")
    print(pd.DataFrame(headline).T.to_string())

    ifc = (R[(R.model == "iforest") & (R.fs == "full")]
           .groupby(["dataset", "condition"])
           .apply(lambda d: pd.Series({
               "n": len(d),
               "positives": int((d.y == 1).sum()),
               "sensitivity": round(float((d[(d.pred == 1) & (d.y == 1)].shape[0]) /
                                        (d.y == 1).sum()), 3) if (d.y == 1).sum() else None})))
    print("\n=== IF sensitivity by condition (matched 5%) ===")
    print(ifc.to_string())

    ablation = {}
    for fs_name in FEATURE_SETS:
        sub = R[(R.model == "iforest") & (R.fs == fs_name)]
        if len(sub):
            ablation[fs_name] = metrics_of(sub.y.values, sub.pred.values)
    print("\n=== Ablation (IF, matched 5%) ===")
    print(pd.DataFrame(ablation).T.to_string())

    os.makedirs(out_dir, exist_ok=True)
    tables = os.path.join(out_dir, "tier1_tables")
    os.makedirs(tables, exist_ok=True)
    with open(os.path.join(out_dir, "tier1_results.json"), "w") as f:
        json.dump({"protocol": ("per-subject 70/30 chronological split; reference = "
                                "ECG-channel deviation (2 SD from the subject's own ECG "
                                "baseline); models use PPG features only; matched 5% "
                                "flag rate"),
                   "headline_matched5": headline,
                   "if_by_condition": json.loads(ifc.to_json()),
                   "ablation": ablation,
                   "n_scored_pairs": int(len(R)),
                   "skipped_subjects": [s for s, _, _ in skipped]}, f, indent=2)
    per_subject.to_csv(os.path.join(tables, "ppg_vs_ecg_per_subject.csv"))
    per_condition.to_csv(os.path.join(tables, "ppg_vs_ecg_per_condition.csv"))
    rej_curve.to_csv(os.path.join(tables, "motion_rejection_vs_threshold.csv"))
    rej_condition.to_csv(os.path.join(tables, "motion_rejection_by_condition.csv"))
    ifc.to_csv(os.path.join(tables, "if_sensitivity_by_condition.csv"))
    R.to_csv(os.path.join(tables, "scored_windows.csv"), index=False)
    print(f"\nsaved -> {out_dir}/tier1_results.json + {out_dir}/tier1_tables/*.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/ppg_wearable_features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--motion-gate", type=float, default=0.05,
                    help="acc_rms_g gate in g for the SCORING set "
                         "(0 = keep everything; builder default 0.05)")
    a = ap.parse_args()
    run(a.csv, a.out, a.motion_gate)
