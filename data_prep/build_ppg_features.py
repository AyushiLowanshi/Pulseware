#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

WESAD_LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation",
                0: None, 5: None, 6: None, 7: None}
MIN_LABEL_PURITY = 0.80
LABEL_FS = 700
MOTION_THRESHOLD_G = 0.05

def _first_key(d, candidates, where=""):
    for c in candidates:
        if c in d:
            return c
    raise KeyError(f"{where}: none of {candidates} found (keys: {list(d)[:12]})")

def _sid(value, fallback_folder):
    if isinstance(value, bool) or value is None:
        return fallback_folder
    if isinstance(value, int):
        return f"S{value:02d}"
    s = str(value).strip().upper().lstrip("S")
    if s.isdigit():
        return f"S{int(s):02d}"
    return fallback_folder or str(value)

def bandpass(x, lo, hi, fs, order=4):
    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, np.asarray(x, dtype=float))

try:
    from wfdb.processing import xqrs_detect
    XQRS_AVAILABLE = True
except Exception:
    XQRS_AVAILABLE = False

def ecg_rpeaks(ecg, fs):
    if XQRS_AVAILABLE and len(ecg) > int(5 * fs):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pk = xqrs_detect(np.asarray(ecg, dtype=float), fs=fs, verbose=False)
            pk = np.asarray(pk, dtype=int)

            min_expected = int(len(ecg) / fs / 2.0)
            if len(pk) >= min_expected:
                return pk
        except Exception:
            pass
    return _ecg_rpeaks_fixed(ecg, fs)

def _ecg_rpeaks_fixed(ecg, fs):
    x = bandpass(ecg, 5.0, 15.0, fs)
    sq = x ** 2
    w = int(0.15 * fs)
    integr = np.convolve(sq, np.ones(w) / w, mode="same")
    if float(integr.max()) <= 0:
        return np.array([], dtype=int)

    thr = 0.4 * float(np.percentile(integr, 99.5))
    peaks, _ = find_peaks(integr, distance=int(0.25 * fs), prominence=thr)

    half = int(0.05 * fs)
    snapped = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(x), p + half + 1)
        snapped.append((lo + int(np.argmax(np.abs(x[lo:hi]))),
                        float(np.max(np.abs(x[lo:hi])))))

    snapped.sort()
    kept = []
    for s, amp in snapped:
        if kept and s - kept[-1][0] < int(0.25 * fs):
            if amp > kept[-1][1]:
                kept[-1] = (s, amp)
            continue
        kept.append((s, amp))
    return np.array([s for s, _ in kept], dtype=int)

def ppg_peaks(ppg, fs):
    x = bandpass(ppg, 0.5, 5.0, fs)
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(x, distance=int(0.30 * fs), prominence=0.5 * rms)
    return peaks

def hrv_from_peaks(peaks, fs):
    if len(peaks) < 3:
        return np.nan, np.nan, np.nan
    ibi = np.diff(peaks) / fs * 1000.0
    ibi = ibi[(ibi > 300) & (ibi < 2000)]
    if len(ibi) < 2:
        return np.nan, np.nan, np.nan
    bpm = 60000.0 / float(np.mean(ibi))
    sdnn = float(np.std(ibi, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 2 else np.nan
    return bpm, sdnn, rmssd

def acc_motion_g(acc_win, unit_hint=None):
    a = np.asarray(acc_win, dtype=float)
    if len(a) < 8:
        return np.nan, "no_data"
    norm = np.sqrt((a ** 2).sum(axis=1))
    med = float(np.median(norm))
    if unit_hint == "g":
        unit, scale = "g (forced)", 1.0
    elif unit_hint == "m/s2":
        unit, scale = "m/s2 (forced)", 1 / 9.81
    elif abs(med - 1.0) < 0.35:
        unit, scale = "g", 1.0
    elif 5.0 <= med < 60.0:
        unit, scale = "m/s2", 1 / 9.81
    elif 55.0 <= med < 75.0:

        unit, scale = "E4-raw (1g=64)", 1.0 / 64.0
    else:

        unit, scale = f"raw:med={med:.2f}", 1.0 / max(med, 1e-9)
    dyn = a - a.mean(axis=0, keepdims=True)
    rms_raw = float(np.sqrt(np.mean((dyn ** 2).sum(axis=1))))
    return rms_raw * scale, unit

def build_rows(sid, dataset, ecg, ecg_fs, ppg, ppg_fs, acc, acc_fs, lab,
               label_map, win_s, stride_s, unit_hint=None):

    r_all = ecg_rpeaks(np.asarray(ecg, dtype=float), ecg_fs) if len(ecg) else np.array([], dtype=int)
    p_all = ppg_peaks(np.asarray(ppg, dtype=float), ppg_fs) if len(ppg) else np.array([], dtype=int)
    rows = []
    w_lab, s_lab = int(win_s * LABEL_FS), int(stride_s * LABEL_FS)
    for st in range(0, len(lab) - w_lab + 1, s_lab):
        en = st + w_lab
        seg = lab[st:en]
        vals, counts = np.unique(seg, return_counts=True)
        lab_id = int(vals[np.argmax(counts)])
        purity = float(counts.max()) / len(seg)
        if purity < MIN_LABEL_PURITY:
            continue
        cond = label_map(lab_id)
        if cond is None:
            continue
        lo_e, hi_e = int(st * ecg_fs / LABEL_FS), int(en * ecg_fs / LABEL_FS)
        rp = r_all[(r_all >= lo_e) & (r_all < hi_e)] - lo_e
        hr_ref, sdnn_ref, rmssd_ref = hrv_from_peaks(rp, ecg_fs)
        lo_p, hi_p = int(st * ppg_fs / LABEL_FS), int(en * ppg_fs / LABEL_FS)
        pp = p_all[(p_all >= lo_p) & (p_all < hi_p)] - lo_p
        bpm, sdnn, rmssd = hrv_from_peaks(pp, ppg_fs)
        a = acc[int(st * acc_fs / LABEL_FS):int(en * acc_fs / LABEL_FS)]
        if len(a) == 0:
            continue
        rms_g, unit = acc_motion_g(a, unit_hint)
        rows.append({
            "subject_id": sid, "dataset": dataset, "condition": cond,
            "t_start_s": round(st / LABEL_FS, 1), "window_s": win_s,
            "label_purity": round(purity, 3),
            "bpm": round(bpm, 2) if bpm == bpm else np.nan,
            "hrv_rmssd": round(rmssd, 2) if rmssd == rmssd else np.nan,
            "hrv_sdnn": round(sdnn, 2) if sdnn == sdnn else np.nan,
            "n_ppg_peaks": int(len(pp)),
            "ecg_hr_ref": round(hr_ref, 2) if hr_ref == hr_ref else np.nan,
            "ecg_rmssd_ref": round(rmssd_ref, 2) if rmssd_ref == rmssd_ref else np.nan,
            "ecg_sdnn_ref": round(sdnn_ref, 2) if sdnn_ref == sdnn_ref else np.nan,
            "n_ecg_peaks": int(len(rp)),
            "acc_rms_g": round(rms_g, 4) if rms_g == rms_g else np.nan,
            "acc_unit": unit,
            "motion_rejected": bool(rms_g == rms_g and rms_g > MOTION_THRESHOLD_G),
        })
    return rows

def read_wesad(pkl_path, win_s, stride_s, unit_hint=None):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    sid = _sid(d.get("subject"), os.path.basename(os.path.dirname(pkl_path)))
    chest, wrist = d["signal"]["chest"], d["signal"]["wrist"]
    ecg = np.asarray(chest["ECG"], dtype=float).ravel()
    bvp = np.asarray(wrist[_first_key(wrist, ["BVP", "bvp", "PPG", "ppg"], "wrist")],
                     dtype=float).ravel()
    acc = np.asarray(wrist[_first_key(wrist, ["ACC", "acc"], "wrist")], dtype=float)
    lab = np.asarray(d["label"]).ravel().astype(int)

    def label_map(v):
        return WESAD_LABELS.get(v)
    return build_rows(sid, "WESAD", ecg, 700, bvp, 64, acc, 32, lab, label_map,
                      win_s, stride_s, unit_hint)

def _dalia_activity_names(d):
    raw = d.get("activity_name", d.get("activity"))
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], str):
        return [str(x).strip().lower() for x in raw]
    arr = np.asarray(raw).ravel()
    if arr.size <= 40:
        try:
            names = []
            for x in arr:
                inner = np.asarray(x).ravel()
                names.append(str(inner[0]).strip().lower() if inner.size else "")
            if any(names):
                return names
        except Exception:
            pass
        names = [str(x).strip().lower() for x in arr]
        return names if any(names) else None
    return None

def read_dalia(pkl_path, win_s, stride_s, unit_hint=None):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    sid = _sid(d.get("subject"), os.path.basename(os.path.dirname(pkl_path)))
    chest, wrist = d["signal"]["chest"], d["signal"]["wrist"]
    ecg = np.asarray(chest[_first_key(chest, ["ECG", "ecg"], "chest")], dtype=float).ravel()
    ppg = np.asarray(wrist[_first_key(wrist, ["PPG", "ppg", "BVP", "bvp"], "wrist")],
                     dtype=float).ravel()
    acc = np.asarray(wrist[_first_key(wrist, ["ACC", "acc"], "wrist")], dtype=float)
    act = np.asarray(d.get("activity", np.array([]))).ravel().astype(int)
    hr_truth = np.asarray(d.get("label", np.array([]))).ravel().astype(float)
    if "rpeaks" in d:
        r_all = np.asarray(d["rpeaks"]).ravel().astype(int)
        r_all = r_all[(r_all >= 0) & (r_all < len(ecg))]
    else:
        r_all = ecg_rpeaks(ecg, 700)
    p_all = ppg_peaks(ppg, 64) if len(ppg) else np.array([], dtype=int)

    rows = []
    w_e, s_e = int(win_s * 700), int(stride_s * 700)

    act_cadence = len(ecg) / len(act) if len(act) else np.inf
    truth_cadence = len(ecg) / len(hr_truth) if len(hr_truth) else np.inf
    for st in range(0, len(ecg) - w_e + 1, s_e):
        en = st + w_e
        i0, i1 = int(st / act_cadence), max(int(st / act_cadence) + 1, int(en / act_cadence))
        a = act[i0:min(i1, len(act))]
        if len(a) == 0:
            continue
        vals, counts = np.unique(a, return_counts=True)
        act_id = int(vals[np.argmax(counts)])
        purity = float(counts.max()) / len(a)
        if purity < MIN_LABEL_PURITY or act_id < 0:
            continue
        rp = r_all[(r_all >= st) & (r_all < en)] - st
        hr_ref, sdnn_ref, rmssd_ref = hrv_from_peaks(rp, 700)
        lo_p, hi_p = int(st * 64 / 700), int(en * 64 / 700)
        pp = p_all[(p_all >= lo_p) & (p_all < hi_p)] - lo_p
        bpm, sdnn, rmssd = hrv_from_peaks(pp, 64)
        a32 = acc[int(st * 32 / 700):int(en * 32 / 700)]
        if len(a32) == 0:
            continue
        rms_g, unit = acc_motion_g(a32, unit_hint)
        t0 = int(st / truth_cadence)
        t1 = max(t0 + 1, int(en / truth_cadence))
        truth = float(np.mean(hr_truth[t0:min(t1, len(hr_truth))])) if len(hr_truth) else np.nan
        rows.append({
            "subject_id": sid, "dataset": "PPG-DaLiA", "condition": f"act{act_id}",
            "t_start_s": round(st / 700, 1), "window_s": win_s,
            "label_purity": round(purity, 3),
            "bpm": round(bpm, 2) if bpm == bpm else np.nan,
            "hrv_rmssd": round(rmssd, 2) if rmssd == rmssd else np.nan,
            "hrv_sdnn": round(sdnn, 2) if sdnn == sdnn else np.nan,
            "n_ppg_peaks": int(len(pp)),
            "ecg_hr_ref": round(hr_ref, 2) if hr_ref == hr_ref else np.nan,
            "ecg_rmssd_ref": round(rmssd_ref, 2) if rmssd_ref == rmssd_ref else np.nan,
            "ecg_sdnn_ref": round(sdnn_ref, 2) if sdnn_ref == sdnn_ref else np.nan,
            "n_ecg_peaks": int(len(rp)),
            "label_hr_truth": round(truth, 2) if truth == truth else np.nan,
            "acc_rms_g": round(rms_g, 4) if rms_g == rms_g else np.nan,
            "acc_unit": unit,
            "motion_rejected": bool(rms_g == rms_g and rms_g > MOTION_THRESHOLD_G),
        })
    return rows

def peek(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    print("top keys:", list(d.keys()))
    print("subject field:", repr(d.get("subject")))
    for dev in d.get("signal", {}):
        for k, v in d["signal"][dev].items():
            v = np.asarray(v, dtype=float)
            extra = ""
            if v.ndim == 2 and v.shape[1] == 3:
                norm = np.sqrt((v ** 2).sum(axis=1))
                extra = (f" | |3D| median={np.median(norm):.3f} p5={np.percentile(norm,5):.3f} "
                         f"p95={np.percentile(norm,95):.3f}")
            print(f"  signal[{dev}][{k}]: shape={v.shape} "
                  f"median={np.median(v):.4g} min={v.min():.4g} max={v.max():.4g}{extra}")
    if "label" in d:
        lab = np.asarray(d["label"]).ravel()
        u, c = np.unique(lab, return_counts=True)
        print("  label: len", len(lab), "| unique:", dict(zip(u.tolist(), c.tolist())))
        ecg_shape = np.asarray(d["signal"]["chest"]["ECG"]).shape
        print(f"  label-vs-ECG length ratio: {len(lab)} / {ecg_shape[0]} = {len(lab)/ecg_shape[0]:.4f} "
              f"(1.0 = label at ECG rate; 0.0114 = 8 Hz)")
    for key in ("activity", "activity_name"):
        if key in d:
            raw = d[key]
            print(f"  {key} (type {type(raw).__name__}):", str(raw)[:300])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dalia", default=None)
    ap.add_argument("--wesad", default=None)
    ap.add_argument("--out", default="../data/ppg_wearable_features.csv")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--peek", default=None)
    ap.add_argument("--acc-unit", choices=["auto", "g", "m/s2"], default="auto")
    args = ap.parse_args()

    if args.peek:
        peek(args.peek)
        return
    unit_hint = None if args.acc_unit == "auto" else args.acc_unit

    rows = []
    done = set()
    if os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out)
            prev = prev.drop_duplicates(subset=["dataset", "subject_id", "t_start_s"], keep="last")
            done = set(zip(prev["dataset"], prev["subject_id"].astype(str)))
            rows = prev.to_dict("records")
            print(f"resuming: {len(rows)} rows / {len(done)} subjects already in {args.out}")
            print("NOTE: rows in the CSV were computed with THIS script's settings; "
                  "delete the CSV to force a full rebuild.")
        except Exception:
            pass

    jobs = []
    if args.wesad:
        jobs += [("WESAD", s, os.path.join(args.wesad, s, f"{s}.pkl"), read_wesad)
                 for s in sorted(os.listdir(args.wesad))]
    if args.dalia:
        jobs += [("PPG-DaLiA", s, os.path.join(args.dalia, s, f"{s}.pkl"), read_dalia)
                 for s in sorted(os.listdir(args.dalia))]

    for ds_name, s, p, reader in jobs:
        if not os.path.exists(p):
            continue
        if (ds_name, _sid(s, s)) in done:
            print(f"{ds_name} {s}: already done - skipping", flush=True)
            continue
        print(f"{ds_name} {s} ...", flush=True)
        try:
            new_rows = reader(p, args.window, args.stride, unit_hint)
        except Exception as e:
            print(f"  FAILED {ds_name} {s}: {type(e).__name__}: {e} - continuing")
            continue
        rows += new_rows
        pd.DataFrame(rows).to_csv(args.out, index=False)
    if not rows:
        sys.exit("No pickles found. Check --dalia / --wesad paths.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} windows from {df.subject_id.nunique()} subjects -> {args.out}")
    print("\nmotion-rejected fraction by dataset/condition (M13 deliverable):")
    print(df.groupby(["dataset", "condition"])["motion_rejected"].mean().to_string())
    print("\nunits encountered:", df.acc_unit.value_counts().to_dict())
    print("\nPPG vs chest-ECG by dataset (validation anchor, M2/M13):")
    for ds, g in df.groupby("dataset"):
        ok = g.dropna(subset=["bpm", "ecg_hr_ref"])
        if len(ok):
            err = (ok.bpm - ok.ecg_hr_ref).abs()
            print(f"  {ds}: n={len(ok)}  median PPG HR {ok.bpm.median():.1f}  "
                  f"median ECG HR {ok.ecg_hr_ref.median():.1f}  "
                  f"|err| median {err.median():.2f} BPM  corr {ok.bpm.corr(ok.ecg_hr_ref):.3f}")

if __name__ == "__main__":
    main()
