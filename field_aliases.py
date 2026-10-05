"""
field_aliases.py

Shared across both tiers: different watches and different ECG reports
label the exact same measurement differently (e.g. "RR Interval" vs
"R-R" vs "NN Interval" vs "Cycle Length" all mean the same thing). This
module maps those alternate names back to one internal canonical field
name, so CSV/Excel uploads and file-reading prompts work regardless of
which label a given device or report happens to use.

Mirrors the equivalent field definitions in web-app/src/App.jsx (the
React version) so both apps recognize the same synonyms.
"""

import re

WEARABLE_FIELD_DEFS = [
    {"key": "bpm", "label": "BPM", "unit": "", "mandatory": True,
     "aliases": ["heart rate", "hr", "heartrate"]},
    {"key": "hrv_rmssd", "label": "HRV (RMSSD)", "unit": "ms", "mandatory": False,
     "aliases": ["rmssd", "hrv rmssd", "hrv (rmssd)"]},
    {"key": "hrv_sdnn", "label": "HRV (SDNN)", "unit": "ms", "mandatory": False,
     "aliases": ["sdnn", "hrv sdnn", "hrv (sdnn)", "heart rate variability"]},
    {"key": "pnn50", "label": "pNN50", "unit": "%", "mandatory": False,
     "aliases": ["pnn50", "pnn 50"]},
    {"key": "spo2", "label": "SpO2", "unit": "%", "mandatory": False,
     "aliases": ["spo2", "blood oxygen", "oxygen saturation", "o2 saturation"]},
    {"key": "resp_rate", "label": "Respiratory rate", "unit": "breaths/min", "mandatory": False,
     "aliases": ["respiratory rate", "breathing rate", "resp rate"]},
    {"key": "skin_temp_c", "label": "Skin temp deviation", "unit": "\u00b0C", "mandatory": False,
     "aliases": ["skin temperature", "temperature deviation", "body temperature deviation", "wrist temperature"]},
    {"key": "vo2max", "label": "VO2 max", "unit": "", "mandatory": False,
     "aliases": ["vo2max", "vo2 max", "cardio fitness"]},
    {"key": "readiness_score", "label": "Readiness / recovery score", "unit": "", "mandatory": False,
     "aliases": ["readiness score", "recovery score", "body battery", "readiness", "recovery"]},
]

WEARABLE_CORE_KEYS = ["bpm", "hrv_rmssd", "hrv_sdnn"]
WEARABLE_MANDATORY_KEYS = [f["key"] for f in WEARABLE_FIELD_DEFS if f["mandatory"]] + ["activity_state"]
WEARABLE_OPTIONAL_KEYS = [f["key"] for f in WEARABLE_FIELD_DEFS if not f["mandatory"]]

ECG_FIELD_DEFS = [
    {"key": "mean_rr_ms", "label": "Mean RR / NN interval", "unit": "ms", "mandatory": True,
     "aliases": ["rr interval", "nn interval", "r-r", "rr int", "mean rr", "rr", "nn"]},
    {"key": "std_rr_ms", "label": "RR variation", "unit": "ms", "mandatory": True,
     "aliases": ["rr variation", "sdrr", "rr sd", "rr std"]},
    {"key": "pnn50", "label": "pNN50", "unit": "%", "mandatory": True,
     "aliases": ["pnn50", "pnn 50"]},
    {"key": "qrs_width_ms", "label": "QRS duration", "unit": "ms", "mandatory": True,
     "aliases": ["qrs duration", "qrs width", "qrsd", "qrs dur"]},
    {"key": "heart_rate_bpm", "label": "Heart rate", "unit": "bpm", "mandatory": False,
     "aliases": ["heart rate", "hr", "vent. rate", "ventricular rate"]},
    {"key": "pr_interval_ms", "label": "PR interval", "unit": "ms", "mandatory": False,
     "aliases": ["pr interval", "p-r interval", "pr int"]},
    {"key": "p_duration_ms", "label": "P duration", "unit": "ms", "mandatory": False,
     "aliases": ["p duration", "p dur", "p wave duration"]},
    {"key": "qt_ms", "label": "QT interval", "unit": "ms", "mandatory": False,
     "aliases": ["qt interval", "qt int", "qt"]},
    {"key": "qtc_ms", "label": "QTc", "unit": "ms", "mandatory": False,
     "aliases": ["qtc", "qtc bazett", "qtcb", "corrected qt"]},
    {"key": "p_axis_deg", "label": "P axis", "unit": "\u00b0", "mandatory": False,
     "aliases": ["p axis"]},
    {"key": "qrs_axis_deg", "label": "QRS axis", "unit": "\u00b0", "mandatory": False,
     "aliases": ["qrs axis"]},
    {"key": "t_axis_deg", "label": "T axis", "unit": "\u00b0", "mandatory": False,
     "aliases": ["t axis"]},
    {"key": "rv5_mv", "label": "RV5", "unit": "mV", "mandatory": False,
     "aliases": ["rv5", "r wave v5"]},
    {"key": "sv1_mv", "label": "SV1", "unit": "mV", "mandatory": False,
     "aliases": ["sv1", "s wave v1"]},
]

ECG_CORE_KEYS = ["mean_rr_ms", "std_rr_ms", "pnn50", "qrs_width_ms"]
ECG_MANDATORY_KEYS = [f["key"] for f in ECG_FIELD_DEFS if f["mandatory"]]
ECG_OPTIONAL_KEYS = [f["key"] for f in ECG_FIELD_DEFS if not f["mandatory"]]

PLAUSIBLE_RANGES = {
    "bpm": (30, 220, "BPM"),
    "hrv_rmssd": (0, 300, "HRV (RMSSD)"),
    "hrv_sdnn": (0, 300, "HRV (SDNN)"),
    "pnn50": (0, 100, "pNN50"),
    "spo2": (60, 100, "SpO2"),
    "resp_rate": (4, 40, "Respiratory rate"),
    "skin_temp_c": (-6, 6, "Skin temp deviation"),
    "vo2max": (10, 85, "VO2 max"),
    "readiness_score": (0, 100, "Readiness / recovery score"),
    "mean_rr_ms": (250, 2200, "Mean RR"),
    "std_rr_ms": (0, 400, "RR variation"),
    "qrs_width_ms": (40, 220, "QRS duration"),
    "heart_rate_bpm": (20, 300, "Heart rate"),
    "pr_interval_ms": (80, 400, "PR interval"),
    "p_duration_ms": (30, 200, "P duration"),
    "qt_ms": (200, 700, "QT interval"),
    "qtc_ms": (250, 700, "QTc"),
    "p_axis_deg": (-180, 180, "P axis"),
    "qrs_axis_deg": (-180, 180, "QRS axis"),
    "t_axis_deg": (-180, 180, "T axis"),
    "rv5_mv": (0, 6, "RV5"),
    "sv1_mv": (0, 6, "SV1"),
}


def normalize_key(s):
    return re.sub(r"[\s_\-().%]+", "", str(s).strip().lower())


def normalize_row_by_aliases(row: dict, field_defs) -> dict:
    """
    row: a dict of {raw_column_name: value} (e.g. one row from an uploaded
    CSV/Excel file, or a dict of extracted values from a file-reading call).
    Returns a dict with only the CANONICAL keys that were actually found.
    """
    lookup = {}
    for k, v in row.items():
        if v is not None and v != "":
            lookup[normalize_key(k)] = v
    out = {}
    for field in field_defs:
        for name in [field["key"]] + field["aliases"]:
            norm = normalize_key(name)
            if norm in lookup:
                out[field["key"]] = lookup[norm]
                break
    return out


def implausible_fields(reading: dict) -> list:
    """Returns human-readable descriptions of any field whose value is outside a plausible range."""
    problems = []
    for key, (lo, hi, label) in PLAUSIBLE_RANGES.items():
        val = reading.get(key)
        if val is not None and (val < lo or val > hi):
            problems.append(f"{label} of {val} (expected {lo}\u2013{hi})")
    return problems


def null_out_implausible(reading: dict) -> dict:
    """Mutates and returns `reading`, setting any implausible value to None rather than trusting it."""
    for key, (lo, hi, _label) in PLAUSIBLE_RANGES.items():
        val = reading.get(key)
        if val is not None and (val < lo or val > hi):
            reading[key] = None
    return reading
