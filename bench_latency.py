import time, importlib.util, statistics, sys, os, tracemalloc
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def find_file(name):
    for d in [os.path.join(HERE, "pulseware_final"), HERE,
              os.path.join(HERE, "streamlit-app"), os.path.join(HERE, "web-app")]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name + " not found")

def load(name):
    path = find_file(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

ppg = load("ppg_preprocessing")
am  = load("anomaly_model")
ec  = load("ecg_classifier")

def med_p(fn, n=1000):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1e3)
    return statistics.median(ts), float(np.percentile(ts, 95))

fs = 64; dur = 30
t = np.arange(fs*dur)/fs
hr_true = 62 + 2*np.sin(2*np.pi*0.1*t)
ppg_sig = 0.6*np.sin(2*np.pi*hr_true/60*t) + 0.05*np.random.randn(len(t)) + 0.2

hist = pd.DataFrame({
    "timestamp": pd.date_range("2026-11-01", periods=240, freq="15min"),
    "bpm": np.r_[np.random.normal(64, 4, 180), np.random.normal(128, 8, 60)],
    "hrv_rmssd": np.r_[np.random.normal(42, 6, 180), np.random.normal(18, 4, 60)],
    "hrv_sdnn": np.r_[np.random.normal(50, 8, 180), np.random.normal(22, 5, 60)],
    "activity_state": ["resting"]*180 + ["exercise"]*60,
})
new_row = pd.DataFrame([{"timestamp": pd.Timestamp("2026-11-01 20:00"),
                         "bpm": 80.0, "hrv_rmssd": 39.0, "hrv_sdnn": 39.0,
                         "activity_state": "resting"}])

csv_path = os.path.join(HERE, "data", "mitbih_features.csv")
use_csv = os.path.exists(csv_path)

try:
    clf = ec.ECGSnapshotClassifier(seed=42)
except TypeError:
    clf = ec.ECGSnapshotClassifier()
if use_csv:
    clf.fit_on_real_data(csv_path)
else:
    try:
        clf.fit_on_demo_data()
    except TypeError:
        clf.fit_on_demo_data()

def new_t1():
    try:
        return am.PersonalBaselineModel(contamination=0.05)
    except TypeError:
        return am.PersonalBaselineModel()
model = new_t1().fit(hist)

t_clean, p_clean = med_p(lambda: ppg.clean_ppg(ppg_sig, fs=fs))
t_peaks, p_peaks = med_p(lambda: ppg.find_heartbeats(ppg.clean_ppg(ppg_sig, fs=fs), fs=fs))
clean = ppg.clean_ppg(ppg_sig, fs=fs)
_, rr_ms = ppg.find_heartbeats(clean, fs=fs)
def hrv():
    return float(np.std(rr_ms, ddof=1)), float(np.sqrt(np.mean(np.diff(rr_ms)**2)))
t_hrv, p_hrv = med_p(hrv)

state = "resting"
iforest = model.models[state]
scaler = model.scalers[state]
X1 = new_row[am.FEATURE_COLS].values
t_bare, p_bare = med_p(lambda: iforest.decision_function(scaler.transform(X1)))
t_full, p_full = med_p(lambda: am.PersonalBaselineModel.score(model, new_row))

rng = np.random.default_rng(7)
if use_csv:
    row = pd.read_csv(csv_path)[ec.FEATURE_COLS].iloc[[5]].to_numpy()
else:
    row = rng.normal(size=(1, len(ec.FEATURE_COLS)))
t_rf, p_rf = med_p(lambda: clf.multi_model.predict_proba(row))

def whole():
    c = ppg.clean_ppg(ppg_sig, fs=fs)
    _, rr = ppg.find_heartbeats(c, fs=fs)
    r = new_row.copy(); r["bpm"] = float(60_000.0/np.mean(rr))
    return am.PersonalBaselineModel.score(model, r)
t_whole, p_whole = med_p(whole, 300)

tracemalloc.start()
m2 = new_t1().fit(hist)
try:
    clf2 = ec.ECGSnapshotClassifier(seed=42)
except TypeError:
    clf2 = ec.ECGSnapshotClassifier()
if use_csv:
    clf2.fit_on_real_data(csv_path)
else:
    try:
        clf2.fit_on_demo_data()
    except TypeError:
        clf2.fit_on_demo_data()
_ = am.PersonalBaselineModel.score(m2, new_row)
_ = clf2.multi_model.predict_proba(row)
_, peak_b = tracemalloc.get_traced_memory()
tracemalloc.stop()

import platform
print("models: Tier-1 IsolationForest x2 states (200 trees) + Tier-2 RF 150 trees; trained on " +
      ("real mitbih_features.csv" if use_csv else "synthetic demo data"))
print(f"hardware: {platform.processor() or platform.machine()} | python {sys.version.split()[0]}")
rows = [
    ("PPG filter (30 s @ 64 Hz)",                 t_clean, p_clean),
    ("PPG peak detection (250 ms refractory)",    t_peaks, p_peaks),
    ("HRV features (SDNN+RMSSD)",                 t_hrv,   p_hrv),
    ("Tier-1 IF decision_function (bare, 1 row)", t_bare,  p_bare),
    ("Tier-1 full scoring path (baseline+IF+verdict)", t_full, p_full),
    ("Tier-2 RF predict_proba (1 window)",        t_rf,    p_rf),
    ("WHOLE pipeline PPG->HRV->Tier-1",           t_whole, p_whole),
]
w = max(len(r[0]) for r in rows)
for name, m, p95 in rows:
    print(f"{name:<{w}} : median {m:6.2f} ms | p95 {p95:6.2f} ms")
print(f"{'peak Python memory (both models loaded + one pass)':<{w}} : {peak_b/1024:,.0f} KB")
