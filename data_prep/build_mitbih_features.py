import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import wfdb

PATIENT_OVERRIDE = {"201": "P201_202", "202": "P201_202"}
BRADY_RHYTHMS = {"B"}
AFIB_RHYTHMS = {"AFIB", "AFL"}
VENTRICULAR_BEATS = {"V", "F"}
PACED_SYMBOL = "/"
MIN_BEATS_PER_WINDOW = 3

def window_label(mean_hr, frac_afib, frac_brady, frac_pvc):
    if frac_afib >= 0.5:
        return "AFib-like"
    if frac_brady >= 0.5 or mean_hr < 50:
        return "Bradycardia-like"
    if frac_pvc >= 0.10:
        return "PVC-like"
    if mean_hr > 100:
        return "Tachycardia-like"
    return "Normal"

def qrs_width_fn_for_record(rec, dl_dir, sig, fs):
    for ext in ("ecgpuwa", "q1c"):
        try:
            qann = wfdb.rdann(os.path.join(dl_dir, rec), ext)
        except Exception:
            continue
        onset = None
        starts, ends = [], []
        for s, sym in zip(qann.sample, qann.symbol):
            if sym == "(":
                onset = s
            elif sym == ")" and onset is not None:
                starts.append(onset)
                ends.append(s)
                onset = None
        if not starts:
            continue
        starts = np.array(starts)
        ends = np.array(ends)

        def from_ann(beat_sample):
            i = int(np.searchsorted(starts, beat_sample, side="right")) - 1
            if i < 0:
                return None
            o, f = starts[i], ends[i]
            if o <= beat_sample <= f + int(0.25 * fs):
                return (f - o) / fs * 1000.0
            return None
        return from_ann, ext

    def from_signal(beat_sample):
        lo = max(0, beat_sample - int(0.08 * fs))
        hi = min(len(sig), beat_sample + int(0.12 * fs))
        seg = sig[lo:hi]
        if len(seg) < 10:
            return None
        base = float(np.median(seg))
        amp = float(np.max(np.abs(seg - base)))
        if amp <= 0:
            return None
        idx = np.where(np.abs(seg - base) >= 0.4 * amp)[0]
        return (idx[-1] - idx[0] + 1) / fs * 1000.0
    return from_signal, "signal-threshold-40pct"

def parse_annotations(ann, total_samples):
    beats, rhythms = [], []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        if sym == "+":
            name = (aux or "").replace("\x00", "").strip().lstrip("(").rstrip(")").strip()
            if name:
                rhythms.append((int(s), name))
        elif sym and sym not in {'"', "~"}:
            beats.append((int(s), sym))
    return beats, rhythms

def build_record(rec, dl_dir, win_s, stride_s):
    fs = 360
    rec_sig = wfdb.rdrecord(os.path.join(dl_dir, rec))
    sig = rec_sig.p_signal[:, 0].astype(float)
    ann = wfdb.rdann(os.path.join(dl_dir, rec), "atr")
    beats, rhythms = parse_annotations(ann, len(sig))
    qrs_width_fn, qrs_source = qrs_width_fn_for_record(rec, dl_dir, sig, fs)

    segs = []
    for k, (s0, name) in enumerate(rhythms):
        s1 = rhythms[k + 1][0] if k + 1 < len(rhythms) else len(sig)
        segs.append((s0, s1, name))

    rows, excl_paced, excl_short = [], 0, 0
    for st in range(0, len(sig) - int(win_s * fs) + 1, int(stride_s * fs)):
        en = st + int(win_s * fs)
        b = [(s, y) for s, y in beats if st <= s < en]
        if not b:
            continue
        if any(y == PACED_SYMBOL for _, y in b):
            excl_paced += 1
            continue
        if len(b) < MIN_BEATS_PER_WINDOW:
            excl_short += 1
            continue
        rr = np.diff([s for s, _ in b]) / fs * 1000.0
        mean_rr = float(np.mean(rr))
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else float("nan")
        std_rr = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
        pnn50 = float(np.mean(np.abs(np.diff(rr)) > 50) * 100.0) if len(rr) > 1 else 0.0
        widths = [w for w in (qrs_width_fn(s) for s, _ in b) if w is not None]
        qrs_med = float(np.median(widths)) if widths else float("nan")

        cov = Counter()
        for s0, s1, name in segs:
            ov = max(0, min(en, s1) - max(st, s0))
            if ov > 0:
                cov[name] += ov
        frac = {k: v / (en - st) for k, v in cov.items()}
        frac_afib = sum(v for k, v in frac.items() if k in AFIB_RHYTHMS)
        frac_brady = sum(v for k, v in frac.items() if k in BRADY_RHYTHMS)
        frac_pvc = sum(1 for _, y in b if y in VENTRICULAR_BEATS) / len(b)

        rows.append({
            "record_id": rec,
            "patient_id": PATIENT_OVERRIDE.get(rec, f"P{rec}"),
            "t_start_s": round(st / fs, 3),
            "window_s": win_s,
            "stride_s": stride_s,
            "mean_rr_ms": round(mean_rr, 2),
            "std_rr_ms": round(std_rr, 2),
            "pnn50": round(pnn50, 2),
            "qrs_width_ms": round(qrs_med, 2),
            "qrs_source": qrs_source,
            "mean_hr_bpm": round(mean_hr, 2),
            "n_beats": len(b),
            "frac_afib_rhythm": round(frac_afib, 3),
            "frac_brady_rhythm": round(frac_brady, 3),
            "frac_pvc_beats": round(frac_pvc, 3),
            "rhythms_present": "|".join(sorted(cov)),
            "label": window_label(mean_hr, frac_afib, frac_brady, frac_pvc),
        })
    stats = {"record": rec, "windows": len(rows), "excluded_paced": excl_paced,
             "excluded_short": excl_short}
    return rows, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--dl-dir", default="../data/mitdb")
    ap.add_argument("--out", default="../data/mitbih_features.csv")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    if args.download:
        print("Downloading MIT-BIH Arrhythmia DB (mitdb) from PhysioNet ...")
        wfdb.dl_database("mitdb", args.dl_dir)
        print("Download complete.")

    records = [l.strip() for l in open(os.path.join(args.dl_dir, "RECORDS")) if l.strip()]
    all_rows, per_record = [], []
    for rec in records:
        rows, stats = build_record(rec, args.dl_dir, args.window, args.stride)
        all_rows.extend(rows)
        per_record.append(stats)
        cls = Counter(r["label"] for r in rows)
        print(f"rec {rec}: {stats['windows']:4d} windows "
              f"(paced-exc {stats['excluded_paced']}, short-exc {stats['excluded_short']}) {dict(cls)}")

    df = pd.DataFrame(all_rows)
    out_abs = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    df.to_csv(out_abs, index=False)
    pd.DataFrame(per_record).to_csv(out_abs.replace(".csv", "_per_record.csv"), index=False)
    print(f"\nWrote {len(df)} windows from {df.record_id.nunique()} records "
          f"({df.patient_id.nunique()} patients) -> {out_abs}")
    print("Class counts:", df.label.value_counts().to_dict())
    print("\nPer-record class counts:")
    print(pd.crosstab(df.record_id, df.label).to_string())

if __name__ == "__main__":
    main()
