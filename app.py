

import os
import re
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from anomaly_model import (PersonalBaselineModel, RESTING_GUARDRAILS, BUCKET_LABELS, RECOVERY_WINDOW_MIN, RECOVERY_LIMITS,
                           MAX_ASSUMED_WORKOUT_MIN, SYMPTOM_OPTIONS, apply_recovery_context, symptom_escalation,
                           load_real_wearable_features, DEFAULT_REAL_DATA_PATH as PPG_REAL_PATH)
from ecg_classifier import ECGSnapshotClassifier, FEATURE_COLS as ECG_FEATURE_COLS, DEFAULT_REAL_DATA_PATH as MITBIH_REAL_PATH
from generate_dummy_data import make_history, make_new_readings, make_ecg_snapshots
from field_aliases import (
    WEARABLE_FIELD_DEFS, WEARABLE_CORE_KEYS, WEARABLE_OPTIONAL_KEYS,
    ECG_FIELD_DEFS, ECG_CORE_KEYS, ECG_OPTIONAL_KEYS,
    implausible_fields, null_out_implausible,
)
from file_reading import (read_any_file, UnrecognizedReadingError, screenshot_extras, screenshot_readings_to_rows,
                          detected_workout, reading_activity, _base_date)


def resolve_data_path(default_path: str, filename: str) -> str:
    """
    Search for the data file at default_path, and also in data_prep/ subfolder
    relative to app.py and the current working directory.
    """
    if default_path and os.path.exists(default_path):
        return default_path
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
    return default_path or candidates[0]


def handle_file_upload(uploaded_file, field_defs, session_key, subject):
    """
    Shared upload-handling logic for both tabs: reads the file, updates the
    given session_state dict in place, and syncs directly into the input widget
    keys so that values populate immediately in the UI.

    UnrecognizedReadingError (OCR worked, but this doesn't look like the
    right kind of document, or none of the expected fields were present)
    gets a calm st.info - it's not a technical failure, just a mismatched
    or incomplete upload, and a red error box would overstate the problem.
    Anything else (missing Tesseract, unreadable file, etc.) is a real
    technical failure and still shows as st.error.
    """
    try:
        matched, extra_info = read_any_file(uploaded_file, field_defs, subject=subject)
        found_before_validation = list(matched.keys())
        null_out_implausible(matched)
        st.session_state[session_key].update(matched)

                                                                                
        if session_key == "ecg_manual":
            ecg_map = {
                "mean_rr_ms": "e_meanrr",
                "std_rr_ms": "e_stdrr",
                "pnn50": "e_pnn50",
                "qrs_width_ms": "e_qrs",
            }
            for k, val in matched.items():
                if val is not None:
                    wkey = ecg_map.get(k, f"e_{k}")
                    st.session_state[wkey] = float(val)
        elif session_key == "wearable_manual":
            wearable_map = {
                "bpm": "w_bpm",
                "activity_state": "w_activity",
                "hrv_rmssd": "w_rmssd",
                "hrv_sdnn": "w_sdnn",
            }
            for k, val in matched.items():
                if val is not None and k != "activity_state":
                    st.session_state[wearable_map.get(k, f"w_{k}")] = float(val)
                                                                                  
                                                                                
            activity = reading_activity(matched, extra_info)
            st.session_state["w_activity"] = activity
            st.session_state[session_key]["activity_state"] = activity

        all_keys = [f["key"] for f in field_defs]
        labels = {f["key"]: f["label"] for f in field_defs}
        found = [labels[k] for k in all_keys if k in found_before_validation and matched.get(k) is not None]
        if session_key == "wearable_manual" and st.session_state.get("w_activity"):
            found.append("Activity")
        rejected = [labels.get(k, k) for k in found_before_validation if matched.get(k) is None]
        msg = f"Found: {', '.join(found) if found else 'nothing recognizable'}."
        if rejected:
            msg += f" Ignored implausible value(s) for: {', '.join(rejected)}."
        st.session_state[f"{session_key}_banner"] = ("success", msg, extra_info)
        if session_key == "wearable_manual":
                                                                            
            workout = detected_workout(screenshot_extras(extra_info)) if isinstance(extra_info, str) else None
            st.session_state["ctx_detected"] = workout
            st.session_state["ctx_exercise"] = "Yes" if workout else "Not sure"
            if workout:
                st.session_state["ctx_workout_end"] = workout["end"].to_pydatetime().time()
            st.session_state.pop("screenshot_batch", None)
    except UnrecognizedReadingError as e:
        st.session_state[f"{session_key}_banner"] = ("info", str(e), None)
    except Exception as e:
        if session_key == "ecg_manual":
            st.session_state["_last_ecg_file"] = None
        elif session_key == "wearable_manual":
            st.session_state["_last_wearable_file"] = None
        st.session_state[f"{session_key}_banner"] = ("error", f"Couldn't read that file: {e}", None)


                                                                                     
                                                                                         
                                                                               
                                                                    
def _cached_ppg(path):
    return load_real_wearable_features(path)

@st.cache_resource(show_spinner=False)
def _cached_ecg_model(path):
    m = ECGSnapshotClassifier()
    try:
        m.fit_auto(path)
    except Exception:
        m.fit_on_demo_data()
    return m

st.set_page_config(page_title="Heart Rhythm & Health Prediction", layout="centered", initial_sidebar_state="collapsed")

                                                                             
                                                                          
                                                                         
                                                                        
                                           
                                                                             
st.markdown("""
<style>
:root {
  --bg: #F2F2F7; --card: #FFFFFF;
  --label: #000000; --label-2: rgba(60,60,67,.6); --label-3: rgba(60,60,67,.3);
  --separator: rgba(60,60,67,.29); --fill-3: rgba(118,118,128,.12);
  --blue: #007AFF; --heart: #FF2D55; --red: #FF3B30; --green: #34C759; --orange: #FF9500;
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
/* Type: system font everywhere Streamlit puts text (icons are spans, so they keep their icon font). */
.stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li, .stApp label, .stApp input,
.stApp button, .stApp select, .stApp textarea, .stApp small, .stApp b, .stApp span:not([data-testid="stIconMaterial"]),
[data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stWidgetLabel"], [data-baseweb="tab"], [data-testid="stTab"] { font-family: var(--font) !important; }
.stApp code, .stApp pre, .stApp kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important; }
.stApp [data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
.stApp { background-color: var(--bg); -webkit-font-smoothing: antialiased; }
#MainMenu, footer { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none; }
.block-container { max-width: 880px; padding-top: 2.2rem; padding-bottom: 5rem; }

/* Page header */
.pw-eyebrow { font-size: 13px; font-weight: 600; color: var(--label-2); text-transform: uppercase; letter-spacing: .02em; margin-bottom: 2px; }
.pw-title { font-size: 34px !important; font-weight: 700 !important; letter-spacing: -.02em; line-height: 1.15; color: var(--label); margin: 0; padding: 0; }
.pw-sub { font-size: 15px; color: var(--label-2); margin: 6px 0 4px; }

/* Headings: h1/h2 stay titles; h3 (st.subheader) becomes an iOS section header */
h1, h2 { font-weight: 700 !important; letter-spacing: -.02em; color: var(--label) !important; }
h3 {
  font-size: 13px !important; font-weight: 400 !important; color: var(--label-2) !important;
  text-transform: uppercase; letter-spacing: .03em; padding: 22px 16px 0 !important; margin: 0 !important;
}
h4 { font-size: 15px !important; font-weight: 600 !important; color: var(--label) !important; padding: 12px 0 0 !important; }
[data-testid="stHeaderActionElements"] { display: none; }
div[data-testid="stCaptionContainer"] { color: var(--label-2); font-size: 13px; padding: 0 16px; }

/* Sidebar */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] h2 { font-size: 13px !important; font-weight: 400 !important; text-transform: uppercase; letter-spacing: .03em; color: var(--label-2) !important; padding-bottom: 4px; }
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div { background: var(--card); border-radius: 12px; padding: 4px 12px; }
section[data-testid="stSidebar"] div[data-testid="stRadio"] label { padding: 9px 0; min-height: 42px; margin: 0; }
section[data-testid="stSidebar"] div[data-testid="stRadio"] label + label { border-top: .5px solid var(--separator); }
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label { display: none; }

/* Tabs → segmented control (covers both the react-aria tabs in newer
   Streamlit and the baseweb tabs in older releases) */
[data-testid="stTabs"] [role="tablist"], div[data-baseweb="tab-list"] {
  display: inline-flex !important; gap: 2px; background: var(--fill-3); border-radius: 9px; padding: 2px !important;
  border: none !important; box-shadow: none !important; width: auto !important; align-self: flex-start;
}
[data-testid="stTabs"] [data-orientation="horizontal"] { border-bottom: none !important; }
[data-testid="stTab"], button[data-baseweb="tab"] {
  border-radius: 7px; padding: 4px 14px !important; background: transparent !important; border: none !important; margin: 0 !important;
}
[data-testid="stTab"] p, button[data-baseweb="tab"] p { font-size: 13px !important; font-weight: 500; color: var(--label) !important; line-height: 20px; }
[data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] {
  background: #fff !important; box-shadow: 0 3px 8px rgba(0,0,0,.12), 0 3px 1px rgba(0,0,0,.04);
}
[data-testid="stTab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] p { font-weight: 600; }
.react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }
/* Tier switcher: a radio driven by session state (so a button can jump to Tier 2), styled as the same segmented
   control. Covers both the react-aria radio (Streamlit >= 1.5x) and the older BaseWeb one. */
.st-key-tier_nav [data-testid="stRadio"] { margin-bottom: 4px; }
.st-key-tier_nav [role="radiogroup"] {
  display: inline-flex !important; flex-direction: row; gap: 2px; background: var(--fill-3); border-radius: 9px; padding: 2px;
}
.st-key-tier_nav [role="radiogroup"] > div { margin: 0 !important; }
.st-key-tier_nav [role="radiogroup"] label {
  margin: 0 !important; padding: 4px 14px !important; border-radius: 7px; background: transparent; cursor: pointer; align-items: center;
}
/* Hide only the radio circle: any div inside the option that is not the text, not inside the text, and not an
   ancestor of the text. Structure-agnostic, so it survives Streamlit's changing radio markup. */
.st-key-tier_nav label div:not([data-testid="stMarkdownContainer"]):not([data-testid="stMarkdownContainer"] *):not(:has([data-testid="stMarkdownContainer"])) { display: none !important; }
.st-key-tier_nav label p { font-size: 13px !important; font-weight: 500; color: var(--label) !important; line-height: 20px; }
.st-key-tier_nav label[data-selected="true"], .st-key-tier_nav label:has(input:checked) {
  background: #fff; box-shadow: 0 3px 8px rgba(0,0,0,.12), 0 3px 1px rgba(0,0,0,.04);
}
.st-key-tier_nav label[data-selected="true"] p, .st-key-tier_nav label:has(input:checked) p { font-weight: 600; }
[data-testid="stTabPanel"] { padding-top: 4px; }

/* Cards */
div[data-testid="stMetric"] { background: var(--card); border-radius: 12px; padding: 14px 16px; border: none; }
[data-testid="stMetricLabel"] p { font-size: 12px !important; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: var(--label-2); }
div[data-testid="stMetricValue"] { font-size: 30px; font-weight: 700; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
div[data-testid="stAlert"] { border-radius: 12px; border: none; }
[data-testid="stAlertContainer"] { background: var(--card) !important; border-radius: 12px !important; border: none !important; }
/* st.container(border=True, key="hero_card") → white inset card. Streamlit puts the
   border on different elements depending on version, so strip it from all of them. */
.st-key-hero_card, .st-key-hero_card > div, .st-key-hero_card [data-testid="stVerticalBlock"],
.st-key-hero_card [data-testid="stVerticalBlockBorderWrapper"], .st-key-hero_card [data-testid="stVerticalBlockBorderWrapper"] > div {
  border: none !important; box-shadow: none !important;
}
.st-key-hero_card { background: var(--card); border-radius: 12px; padding: 14px 16px 6px; }
div[data-testid="stExpander"] { background: var(--card); border-radius: 12px; border: none; }
div[data-testid="stExpander"] details { border: none; border-radius: 12px; }
div[data-testid="stExpander"] summary p { font-size: 15px; color: var(--blue); }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { background: var(--card); border-radius: 12px; padding: 8px; }
div[data-testid="stTable"] table { width: 100%; border-collapse: collapse; font-size: 14px; }
div[data-testid="stTable"] th { text-align: left; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: .03em;
  color: var(--label-2, #6e6e73); padding: 8px 10px; border-bottom: 0.5px solid rgba(60,60,67,.29); white-space: nowrap; }
div[data-testid="stTable"] td { padding: 9px 10px; border-bottom: 0.5px solid rgba(60,60,67,.18); vertical-align: top; line-height: 1.4; white-space: nowrap; }
div[data-testid="stTable"] td:not(:first-child):not(:last-child) * { white-space: nowrap !important; }
div[data-testid="stTable"] td:first-child, div[data-testid="stTable"] td:last-child { white-space: normal; }
div[data-testid="stTable"] td:nth-child(n+2) { font-variant-numeric: tabular-nums; }
div[data-testid="stTable"] tr:last-child td { border-bottom: none; }
div[data-testid="stTable"] td:last-child { color: var(--label-2, #6e6e73); font-size: 13px; min-width: 260px; }
div[data-testid="stVegaLiteChart"], div[data-testid="stArrowVegaLiteChart"] { background: var(--card); border-radius: 12px; }
div[data-testid="stJson"] { background: var(--card); border-radius: 12px; padding: 6px 12px; }

/* Inputs */
div[data-testid="stNumberInputContainer"], div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-testid="stNumberInput"] > div > div {
  background: var(--card) !important; border-color: var(--separator) !important; border-radius: 10px !important;
}
div[data-testid="stNumberInputField"], div[data-testid="stNumberInput"] input { background: var(--card) !important; font-variant-numeric: tabular-nums; }
div[data-testid="stNumberInput"] button { background: var(--card) !important; border-color: var(--separator) !important; }
label[data-testid="stWidgetLabel"] p { font-size: 13px; color: var(--label-2); }
section[data-testid="stFileUploaderDropzone"], section[data-testid="stFileUploadDropzone"] {
  background: var(--card); border-radius: 12px; border: 1px dashed var(--separator);
}
div[data-testid="stFileUploader"] button { border-radius: 10px; border-color: var(--separator); color: var(--blue); }

/* Buttons */
div.stButton > button, div.stDownloadButton > button {
  width: 100%; min-height: 44px; border-radius: 12px; border: none;
  background-color: var(--blue); color: #FFFFFF; font-weight: 600; font-size: 15px;
}
div.stButton > button p, div.stDownloadButton > button p { color: #fff !important; font-weight: 600; }
div.stButton > button:hover, div.stDownloadButton > button:hover { background-color: #0071EB; color: #fff; border: none; }
div.stButton > button:focus:not(:active) { color: #fff; border: none; box-shadow: 0 0 0 3px rgba(0,122,255,.25); }

/* Group rows (used in HTML fragments) */
.pw-group { background: var(--card); border-radius: 12px; padding-left: 16px; margin: 6px 0 12px; }
.pw-row { display: flex; align-items: center; gap: 12px; min-height: 44px; padding: 9px 16px 9px 0; font-size: 15px; }
.pw-row + .pw-row { border-top: .5px solid var(--separator); }
.pw-row .grow { flex: 1; min-width: 0; }
.pw-row .sub { display: block; font-size: 13px; color: var(--label-2); margin-top: 2px; }
.pw-row .val { color: var(--label-2); font-variant-numeric: tabular-nums; white-space: nowrap; }
.pw-cat { display: flex; align-items: center; gap: 6px; color: var(--heart); font-weight: 600; font-size: 15px; }
.pw-hero .pw-row { padding: 0 0 8px; min-height: 0; }
.pw-metrics { display: flex; gap: 36px; padding: 6px 0 2px; }
.pw-metric-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: var(--label-2); }
.pw-metric-value { font-size: 30px; font-weight: 700; letter-spacing: -.03em; line-height: 1.1; font-variant-numeric: tabular-nums; margin-top: 2px; }
.pw-metric-value small { font-size: 14px; font-weight: 600; color: var(--label-2); margin-left: 4px; letter-spacing: 0; }
.pw-metrics .sub { font-size: 13px; color: var(--label-2); margin-top: 3px; }
/* Auth — Apple-ID style centered card for Streamlit login / signup / signed-out */
.pw-auth { display: flex; align-items: center; justify-content: center; min-height: 62vh; padding: 24px; }
.pw-auth-card { width: 100%; max-width: 480px; text-align: center; }
.pw-auth-icon { width: 76px; height: 76px; border-radius: 18px; margin: 0 auto 18px; background: #fff; color: var(--heart); box-shadow: 0 0 0 .5px rgba(0,0,0,.08), 0 8px 24px rgba(0,0,0,.08); display: flex; align-items: center; justify-content: center; font-size: 34px; }
.pw-auth-card h1 { font-size: 26px !important; font-weight: 700 !important; letter-spacing: -.02em; text-align: center; margin: 0 0 6px !important; }
.pw-auth-lede { text-align: center; color: var(--label-2); font-size: 15px; margin-bottom: 22px; }
.pw-auth-card .pw-group { text-align: left; }
.pw-auth-hint { margin-top: 12px; padding: 10px 14px; background: rgba(118,118,128,.08); border-radius: 10px; font-size: 13px; color: var(--label-2); line-height: 1.45; text-align: center; }
.pw-auth-foot { display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 18px; font-size: 12px; color: var(--label-2); }
</style>
""", unsafe_allow_html=True)


def fmt_when(ts) -> str:
    """'Sep 7, 9:15 AM' - platform-safe (no %-d, which Windows strftime rejects)."""
    if not hasattr(ts, "strftime"):
        return str(ts)
    return f"{ts.strftime('%b')} {ts.day}, {ts.strftime('%I:%M %p').lstrip('0')}"


                                                                             
                                                                       
                                                                           
                                     
                                                                             
COLUMN_LABELS = {
    "source": "Reading", "timestamp": "Time", "bpm": "BPM", "hrv_rmssd": "HRV (RMSSD)", "hrv_sdnn": "HRV (SDNN)",
    "activity_state": "Activity", "risk_bucket": "Status", "explanation": "Why",
}
SUBTYPE_LABELS = {
    "PVC-like": "early or extra beats (PVC-like)",
    "AFib-like": "an irregular rhythm (AFib-like)",
    "Bradycardia-like": "a slow rhythm (bradycardia-like)",
    "Abnormal": "an abnormal rhythm (outside Normal / PVC-like / AFib-like / Bradycardia-like)",
}


_BUCKET_RANK_UI = {"insufficient_baseline": 0, "normal": 0, "expected_recovery": 0, "during_exercise": 0, "monitor": 1, "flag_for_review": 2}


def bucket_label(bucket) -> str:
    return BUCKET_LABELS.get(bucket, str(bucket).replace("_", " ").capitalize())


def present_readings(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Copy of `df` limited to `columns`, formatted for display."""
    out = df[[c for c in columns if c in df.columns]].copy()
    if "timestamp" in out:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").map(lambda t: fmt_when(t) if pd.notna(t) else "")
    for c in ("bpm", "hrv_rmssd", "hrv_sdnn"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    if "activity_state" in out:
        out["activity_state"] = out["activity_state"].astype(str).str.capitalize()
    if "risk_bucket" in out:
        out["risk_bucket"] = out["risk_bucket"].map(bucket_label)
    return out.rename(columns=COLUMN_LABELS)


def style_by_status(display_df: pd.DataFrame):
    tint = {"Normal": "#F3FBF5", "Expected recovery": "#EAF7F4", "During exercise": "#EEF3FB",
            "Monitor": "#FFF7E8", "Flag for review": "#FFF0EF"}

    def _row(row):
        color = tint.get(row.get("Status"), "")
        return [f"background-color: {color}" if color else ""] * len(row)

    return display_df.style.apply(_row, axis=1)


def show_readings(df: pd.DataFrame, columns, max_wrapped_rows: int = 40):
    """Readings table with the full explanation visible (wrapped). Falls back to the
    scrolling grid only for long uploads, where a wrapped table would be unwieldy."""
    display = present_readings(df, columns)
    if len(display) <= max_wrapped_rows:
        if "Time" in display.columns:                                   
            display["Time"] = display["Time"].map(lambda v: v.replace(" ", "\u00a0") if isinstance(v, str) else v)
        st.table(style_by_status(display).hide(axis="index"))
    else:
        st.dataframe(style_by_status(display), width="stretch", hide_index=True,
                     column_config={"Why": st.column_config.TextColumn("Why", width="large")})
        st.caption("Long list: hover a Why cell to read the whole explanation.")


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


today = date.today()

                                                                             
                                                                  
                                                           
                                                                             
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"                  
if "just_signed_out" not in st.session_state:
    st.session_state.just_signed_out = False

                                                                                 
if os.getenv("PULSEWARE_E2E") == "1":
    if st.session_state.user is None and not st.session_state.just_signed_out:
        st.session_state.user = {"name": "Demo", "email": "demo@example.com"}
try:
    qp = st.query_params                    
    if ("e2e" in qp or "test" in qp) and st.session_state.user is None:
        st.session_state.user = {"name": "Demo", "email": "demo@example.com"}
except Exception:
    pass

if st.session_state.user is None:
                                  
    if st.session_state.just_signed_out:
        st.markdown(
            """<div class="pw-auth"><div class="pw-auth-card">
            <div class="pw-auth-icon">♥</div>
            <h1>You’re signed out</h1>
            <div class="pw-auth-lede">Thanks for using Pulseware. Your session was cleared — sign in again when you’re ready.</div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="pw-group" style="max-width:480px;margin:0 auto;"><div class="pw-row" style="justify-content:center;"><span class="sub" style="text-align:center;width:100%;">Tip: use the demo account jamie@example.com / secret123 to explore Monitoring and Rhythm Check.</span></div></div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Sign in again", key="st_back_to_login", width="stretch"):
            st.session_state.just_signed_out = False
            st.session_state.auth_page = "login"
            st.rerun()
        st.markdown('<div class="pw-auth-foot">🔒 Signed out safely. No data was sent to a server.</div>', unsafe_allow_html=True)
        st.stop()

    is_signup = st.session_state.auth_page == "signup"
    icon = "♥"
    title = "Create your account" if is_signup else "Sign in to Pulseware"
    lede = "Set up an account to get started — your data stays in this browser." if is_signup else "See how your heart rhythm has been trending."
    st.markdown(
        f"""<div class="pw-auth" style="min-height:28vh;padding-bottom:8px;"><div class="pw-auth-card">
        <div class="pw-auth-icon">{icon}</div>
        <h1>{title}</h1>
        <div class="pw-auth-lede">{lede}</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
                                                                        
    _, form_col, _ = st.columns([1, 1.5, 1])
    with form_col:
        if is_signup:
            name = st.text_input("Name", key="st_signup_name", placeholder="Jamie Rivera")
            email = st.text_input("Email", key="st_signup_email", placeholder="you@example.com")
            pw = st.text_input("Password", key="st_signup_pw", type="password", placeholder="At least 6 characters")
            pw2 = st.text_input("Confirm", key="st_signup_pw2", type="password", placeholder="Type it again")
            err = None
            if st.button("Create Account", key="st_do_signup", width="stretch"):
                if not name or len(name.strip()) < 2:
                    err = "Tell us what to call you."
                elif not re.match(r"^\S+@\S+\.\S+$", email or ""):
                    err = "Enter a valid email address."
                elif len(pw or "") < 6:
                    err = "Use at least 6 characters."
                elif pw != pw2:
                    err = "Passwords don’t match."
                if err:
                    st.error(err)
                else:
                    st.session_state.user = {"name": name.strip(), "email": email.strip()}
                    st.session_state.auth_page = "login"
                    st.rerun()
            st.write("")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.caption("Already have an account?")
            with c2:
                if st.button("Sign in", key="st_go_login"):
                    st.session_state.auth_page = "login"
                    st.rerun()
            st.markdown('<div class="pw-auth-foot" style="margin-top:8px;">By creating an account you agree to use Pulseware as a decision-support tool, not a medical device.</div>', unsafe_allow_html=True)
        else:
            email = st.text_input("Email", key="st_login_email", placeholder="you@example.com")
            pw = st.text_input("Password", key="st_login_pw", type="password", placeholder="At least 6 characters")
            if st.button("Sign In", key="st_do_login", width="stretch"):
                if not re.match(r"^\S+@\S+\.\S+$", email or ""):
                    st.error("Enter a valid email address.")
                elif len(pw or "") < 6:
                    st.error("Use at least 6 characters.")
                else:
                                                                                       
                    st.session_state.user = {"name": email.split("@")[0], "email": email.strip()}
                    st.rerun()
                                  
            if st.button("Forgot password?", key="st_forgot"):
                st.info("This is a demo — any valid email + 6-char password will sign you in. Try jamie@example.com / secret123.")
            st.write("")
            if st.button("Don’t have an account? Create one", key="st_go_signup"):
                st.session_state.auth_page = "signup"
                st.rerun()
            st.markdown('<div class="pw-auth-foot">🔒 Your account details stay in this browser session. Demo: jamie@example.com / secret123</div>', unsafe_allow_html=True)
    st.stop()

                      
st.markdown(
    f"""
    <div class="pw-eyebrow">{today.strftime('%A, %d %B')}</div>
    <h1 class="pw-title">Heart Rhythm &amp; Health Prediction</h1>
    <div class="pw-sub">A decision-support tool, not a medical device.</div>
    """,
    unsafe_allow_html=True,
)
                                                                               
_u = st.session_state.user
if _u:
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:10px;justify-content:flex-end;margin:6px 0 2px;font-size:13px;color:rgba(60,60,67,.6);">
        <span style="display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:rgba(120,120,128,.2);color:rgba(60,60,67,.7);font-weight:600;font-size:12px;">{_u['name'][:1].upper()}</span>
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px;">{_u['name']} · {_u['email']}</span>
        </div>""",
        unsafe_allow_html=True,
    )
                                                                                
    _lc, _rc = st.columns([0.78, 0.22])
    with _rc:
        if st.button("Log out", key="st_logout_top", width="stretch"):
            st.session_state.user = None
            st.session_state.just_signed_out = True
            st.rerun()

                                                                             
                                                                    
                                                                             
effective_mitbih_path = resolve_data_path(MITBIH_REAL_PATH, "mitbih_features.csv")
effective_ppg_path = resolve_data_path(PPG_REAL_PATH, "ppg_wearable_features.csv")

                                                                             
                                                                         
                                                                             
if "ecg_model" not in st.session_state:
    st.session_state.ecg_model = _cached_ecg_model(effective_mitbih_path)

ecg_model = st.session_state.ecg_model

                                                                             
                                                                         
                                                                             
raw_ppg_result = _cached_ppg(effective_ppg_path)
if isinstance(raw_ppg_result, tuple):
    real_ppg_df, real_ppg_err = raw_ppg_result
else:
    real_ppg_df, real_ppg_err = raw_ppg_result, None

data_mode = "Generate dummy data"
if real_ppg_df is not None:
    data_mode = "Use real WESAD/PPG-DaLiA data"

st.write("")
TIERS = ["Tier 1: Continuous Monitoring", "History", "Tier 2: ECG Snapshot", "About"]
if "tier" not in st.session_state:
    st.session_state.tier = TIERS[0]


def go_to_tier2(note=None):
    """Button callback: switch to the ECG section, remembering which reading sent the user there."""
    st.session_state.tier = TIERS[2]
    st.session_state.handoff_note = note


def go_to_tier1():
    st.session_state.tier = TIERS[0]
    st.session_state.handoff_note = None


def _tier_changed():
    if st.session_state.tier == TIERS[0]:
        st.session_state.handoff_note = None


with st.container(key="tier_nav"):
    page = st.radio("Section", TIERS, key="tier", horizontal=True, label_visibility="collapsed", on_change=_tier_changed)

                                                                             
                              
                                                                             
if page == TIERS[0]:
                                                                          
                                                       
    if data_mode == "Use real WESAD/PPG-DaLiA data":
        if real_ppg_df is None or len(real_ppg_df) == 0:
            st.error(f"Cannot load real PPG data: {real_ppg_err or 'No data available'}. Please switch to dummy data in the sidebar.")
            st.stop()

        subject_col = "subject_id" if "subject_id" in real_ppg_df.columns else None
        subjects = sorted(real_ppg_df[subject_col].unique()) if subject_col else [None]
        chosen_subject = subjects[0] if subjects else None
        subject_df = real_ppg_df[real_ppg_df[subject_col] == chosen_subject] if subject_col else real_ppg_df
        subject_df = subject_df.sort_values("window_start_s") if "window_start_s" in subject_df.columns else subject_df
        split_idx = int(len(subject_df) * 0.7)
        history_df, new_df = subject_df.iloc[:split_idx].copy(), subject_df.iloc[split_idx:].copy()
        if "timestamp" not in history_df.columns:
            if "window_start_s" in history_df.columns:
                history_df["timestamp"] = pd.to_datetime(history_df["window_start_s"], unit="s", origin="2026-01-01")
                new_df["timestamp"] = pd.to_datetime(new_df["window_start_s"], unit="s", origin="2026-01-01")
            else:
                history_df["timestamp"] = pd.date_range("2026-01-01", periods=len(history_df), freq="1min")
                new_df["timestamp"] = pd.date_range("2026-01-02", periods=len(new_df), freq="1min")
    elif data_mode == "Generate dummy data":
        history_df = make_history()
        new_df = make_new_readings()
    else:
        c1, c2 = st.columns(2)
        with c1:
            history_file = st.file_uploader(
                "Upload historical readings CSV (columns: timestamp, bpm, hrv_rmssd, hrv_sdnn, activity_state)",
                type="csv", key="history_upload",
            )
        with c2:
            new_file = st.file_uploader("Upload new readings CSV (same columns)", type="csv", key="new_upload")
        history_df = pd.read_csv(history_file) if history_file else None
        new_df = pd.read_csv(new_file) if new_file else None

    if history_df is not None and new_df is not None:
        model = PersonalBaselineModel(contamination=0.05).fit(history_df)
                                                                       
                                                                         
                                                                       
        st.subheader("Add a single reading to check")
        st.caption(
            "BPM and activity are required to run a check - everything else is optional and "
            "depends on what your watch actually reports. Optional fields are captured and "
            "shown, but only heart rate and HRV feed the check itself."
        )
        wearable_upload = st.file_uploader(
            "Upload a reading (CSV, Excel, image, or PDF)", type=["csv", "xlsx", "xls", "png", "jpg", "jpeg", "pdf"],
            key="wearable_single_upload",
        )
        if "wearable_manual" not in st.session_state:
            st.session_state.wearable_manual = {k: None for k in ["bpm"] + WEARABLE_OPTIONAL_KEYS}
            st.session_state.wearable_manual["activity_state"] = None
            st.session_state.w_bpm = None
            st.session_state.w_activity = None
            st.session_state.w_rmssd = None
            st.session_state.w_sdnn = None
            for k in WEARABLE_OPTIONAL_KEYS:
                if k not in ("hrv_rmssd", "hrv_sdnn"):
                    st.session_state[f"w_{k}"] = None

        if wearable_upload is not None:
            w_sig = f"{wearable_upload.name}_{wearable_upload.size}"
            if st.session_state.get("_last_wearable_file") != w_sig:
                st.session_state["_last_wearable_file"] = w_sig
                handle_file_upload(
                    wearable_upload, WEARABLE_FIELD_DEFS, "wearable_manual",
                    subject="a smartwatch or fitness tracker screen",
                )
        else:
            st.session_state["_last_wearable_file"] = None
            st.session_state.pop("wearable_manual_banner", None)
            st.session_state.pop("screenshot_batch", None)

        if "wearable_manual_banner" in st.session_state:
            b_type, b_msg, b_extra = st.session_state["wearable_manual_banner"]
            if b_type == "info":
                st.info(b_msg)
            elif b_type == "error":
                st.error(b_msg)
            else:
                st.success(b_msg + " Review the fields below before continuing.")

                                                                       
                                                                           
                                                                         
                                                                         
                                                                         
                                 
                                                                       
        extras = None
        include_high = False
        if "wearable_manual_banner" in st.session_state and st.session_state["wearable_manual_banner"][0] == "success":
            extras = screenshot_extras(st.session_state["wearable_manual_banner"][2])
        if extras:
            listed = extras["readings"]
            bpm_high = extras["bpm_high"]
            parts = []
            if listed:
                lo = min(r["bpm"] for r in listed); hi = max(r["bpm"] for r in listed)
                times = [r["time"] for r in listed if r["time"]]
                span = f", {times[0]}–{times[-1]}" if len(times) >= 2 else (f", {times[0]}" if times else "")
                label = listed[0]["label"].lower()
                parts.append(f"{len(listed)} listed {label} reading{'s' if len(listed) != 1 else ''} ({lo:.0f}–{hi:.0f} bpm{span})")
            if bpm_high is not None:
                parts.append(f"a day high of {bpm_high:.0f} bpm")
            st.subheader("Also on this screenshot")
            st.info("Besides the value in the form below, this screenshot shows " + " and ".join(parts) + ". "
                    "All of them are checked together when you press the button below.")
            include_high = True

                                                                       
                                                                      
                                                                          
                                                                           
                                                                        
                                                                        
                                                                       
        st.subheader("Context for this check")
        _clear_batch = lambda: st.session_state.pop("screenshot_batch", None)
        if "ctx_exercise" not in st.session_state:
            st.session_state.ctx_exercise = "Not sure"
        ctx1, ctx2 = st.columns([1.35, 1])
        exercised = ctx1.radio("Did you exercise today?", ["No", "Yes", "Not sure"], horizontal=True,
                               key="ctx_exercise", on_change=_clear_batch,
                               help="A watch's alert rule doesn't know a workout just ended. Telling us stops normal post-exercise recovery being flagged.")
        workout_end_ts = workout_start_ts = None
        detected = st.session_state.get("ctx_detected")
        if exercised == "Yes":
            if "ctx_workout_end" not in st.session_state:
                st.session_state.ctx_workout_end = (datetime.now() - timedelta(hours=1)).replace(second=0, microsecond=0).time()
            end_time = ctx2.time_input(
                "When did it end?", key="ctx_workout_end", step=timedelta(minutes=15), on_change=_clear_batch,
                help=(f"Taken from the screenshot ({detected['label']}, approximate) - adjust if that's wrong. " if detected else "")
                + f"Heart rate normally stays elevated for {RECOVERY_WINDOW_MIN // 60} hours after exercise; readings in that window are rated Expected recovery.",
            )
            screen_day = _base_date(extras.get("date")) if extras else pd.Timestamp.now().normalize()
            workout_end_ts = screen_day + pd.Timedelta(hours=end_time.hour, minutes=end_time.minute)
            if detected and detected.get("start") is not None and abs((detected["end"] - workout_end_ts).total_seconds()) <= 30 * 60:
                workout_start_ts = detected["start"]
        elif detected:
            st.caption(f"This screenshot seems to show a {detected['label']} - choose Yes if that's right.")
        symptoms = st.multiselect("Any symptoms today?", SYMPTOM_OPTIONS, key="ctx_symptoms",
                                  placeholder="None", on_change=_clear_batch)

        if extras and extras.get("bpm_high") is not None:
            bpm_high = extras["bpm_high"]
                                                                                  
                                                                              
                                                                              
            alert_max = max((r["bpm"] for r in extras["readings"]), default=None)
            high_above_alerts = alert_max is not None and bpm_high > alert_max
            if exercised == "Yes":
                include_high = True
                st.caption(f"The day high ({bpm_high:.0f} bpm) is treated as part of your workout, not as a resting reading.")
            elif high_above_alerts:
                include_high = st.checkbox(
                    f"Check the day high ({bpm_high:.0f} bpm) as a resting reading anyway. It is above every listed alert, so the watch "
                    f"did not record it as an inactive reading - it most likely happened during activity.",
                    value=False, key="screenshot_include_high", on_change=_clear_batch,
                )
            else:
                include_high = st.checkbox(
                    f"Include the day high ({bpm_high:.0f} bpm). Activity at that moment isn't shown, so it is checked as a resting reading.",
                    value=True, key="screenshot_include_high", on_change=_clear_batch,
                )

        wcols = st.columns(4)
        wm = st.session_state.wearable_manual
        if "w_bpm" not in st.session_state:
            st.session_state.w_bpm = None
        if "w_activity" not in st.session_state:
            st.session_state.w_activity = None
        if "w_rmssd" not in st.session_state:
            st.session_state.w_rmssd = None
        if "w_sdnn" not in st.session_state:
            st.session_state.w_sdnn = None

        wm["bpm"] = wcols[0].number_input("BPM *", key="w_bpm")
        wm["activity_state"] = wcols[1].selectbox("Activity *", ["resting", "exercise"], key="w_activity", index=None,
                                                  placeholder="Choose\u2026", format_func=str.capitalize,
                                                  help="Filled in only when the screenshot shows it (a Resting value, or a workout screen).")
        wm["hrv_rmssd"] = wcols[2].number_input("HRV RMSSD (optional)", key="w_rmssd")
        wm["hrv_sdnn"] = wcols[3].number_input("HRV SDNN (optional)", key="w_sdnn")

        with st.expander("More optional fields (recorded, not used in the check yet)"):
            ocols = st.columns(3)
            for idx, key in enumerate([k for k in WEARABLE_OPTIONAL_KEYS if k not in ("hrv_rmssd", "hrv_sdnn")]):
                label = next(f["label"] for f in WEARABLE_FIELD_DEFS if f["key"] == key)
                if f"w_{key}" not in st.session_state:
                    st.session_state[f"w_{key}"] = None
                wm[key] = ocols[idx % 3].number_input(label, key=f"w_{key}")

        n_extra = (len(extras["readings"]) + (1 if include_high and extras["bpm_high"] is not None else 0)) if extras else 0
        check_label = f"Check this screenshot ({n_extra + 1} readings)" if n_extra else "Check this reading"
        if st.button(check_label, key="check_single_wearable"):
            if wm.get("bpm") is None:
                st.error("BPM is required.")
            elif wm.get("activity_state") not in ("resting", "exercise"):
                st.error("Choose the activity - was this reading taken at rest or during exercise?")
            else:
                problems = implausible_fields(wm)
                if problems:
                    st.error(f"That doesn't look like a real reading — {'; '.join(problems)}. Double-check the numbers.")
                elif not n_extra:
                    single_row = pd.DataFrame([{**wm, "timestamp": pd.Timestamp.now()}])
                    single_scored = model.score(single_row)
                    single_scored, _ = apply_recovery_context(single_scored, workout_end_ts, workout_start_ts)
                    single_scored["explanation"] = single_scored.apply(model.explain, axis=1)
                    sym_bucket, sym_text = symptom_escalation(symptoms)
                    r = single_scored.iloc[0]
                    bucket = r["risk_bucket"]
                    if sym_bucket and _BUCKET_RANK_UI[sym_bucket] > _BUCKET_RANK_UI.get(bucket, 0):
                        bucket = sym_bucket
                    banner = {"normal": st.success, "expected_recovery": st.success, "during_exercise": st.info,
                              "monitor": st.warning}.get(bucket, st.error)
                    banner(f"**{bucket_label(bucket)}** — {r['explanation']}" + (f" {sym_text}" if sym_text else ""))
                else:
                                                                               
                                                                                     
                    primary_label = "Resting (day summary)" if extras.get("bpm_resting") == wm.get("bpm") else "Primary reading"
                    rows = [{**wm, "timestamp": pd.Timestamp.now(), "source": primary_label}]
                    rows += screenshot_readings_to_rows(extras, include_high=include_high)
                    for order, row in enumerate(rows):
                        row["_order"] = order
                        if row["source"].startswith("Day high") and exercised == "Yes":
                            if workout_end_ts is not None:
                                                                                                         
                                                                                                         
                                start_ts = workout_start_ts if workout_start_ts is not None else workout_end_ts - pd.Timedelta(minutes=MAX_ASSUMED_WORKOUT_MIN)
                                ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
                                if pd.isna(ts) or not (start_ts <= ts < workout_end_ts):
                                    row["timestamp"] = workout_end_ts - pd.Timedelta(minutes=1)
                            row["source"] = "Day high (during your workout)"
                    batch_scored = model.score(pd.DataFrame(rows))
                    batch_scored, recovery = apply_recovery_context(batch_scored, workout_end_ts, workout_start_ts)
                    batch_scored["explanation"] = batch_scored.apply(model.explain, axis=1)
                    if exercised != "Yes":
                        is_high = batch_scored["source"].str.startswith("Day high")
                        batch_scored.loc[is_high, "explanation"] = (
                            batch_scored.loc[is_high, "explanation"] + " Day high - activity unknown; expected if it was during exercise."
                        )
                    batch_scored = batch_scored.sort_values("_order").drop(columns="_order").reset_index(drop=True)
                    sym_bucket, sym_text = symptom_escalation(symptoms)
                    if sym_bucket and _BUCKET_RANK_UI[sym_bucket] > _BUCKET_RANK_UI.get(batch_scored.at[0, "risk_bucket"], 0):
                        batch_scored.at[0, "risk_bucket"] = sym_bucket
                        batch_scored.at[0, "explanation"] = f"{batch_scored.at[0, 'explanation']} {sym_text}"
                    st.session_state["screenshot_batch"] = (st.session_state.get("_last_wearable_file"), batch_scored,
                                                            {"recovery": recovery, "symptoms": sym_text, "exercised": exercised})

        batch = st.session_state.get("screenshot_batch")
        if batch and batch[0] == st.session_state.get("_last_wearable_file"):
            batch_scored = batch[1]
            meta = batch[2] if len(batch) > 2 else {"recovery": {}, "symptoms": "", "exercised": "Not sure"}
            recovery = meta.get("recovery") or {}
            bc = batch_scored["risk_bucket"].value_counts()
            n_norm, n_mon, n_flag = int(bc.get("normal", 0)), int(bc.get("monitor", 0)), int(bc.get("flag_for_review", 0))
            n_rec, n_ex = int(bc.get("expected_recovery", 0)), int(bc.get("during_exercise", 0))
            worst = "flag_for_review" if n_flag else ("monitor" if n_mon else "normal")
            primary = batch_scored.iloc[0]
            others = batch_scored.iloc[1:]
            judged = others[~others["risk_bucket"].isin(["expected_recovery", "during_exercise"])]
            above = judged[judged["bpm"] > primary["bpm"]]

            rec_rows = batch_scored[batch_scored["risk_bucket"] == "expected_recovery"]
            recovery_note = ""
            if n_rec:
                times = [fmt_when(t).split(", ")[-1] for t in pd.to_datetime(rec_rows["timestamp"]).sort_values()]
                span = f"{times[0]}–{times[-1]}" if len(times) > 1 else times[0]
                trend = f" - {recovery['trend']}" if recovery.get("trend") else ""
                recovery_note = (f"The {n_rec} alert reading{'s' if n_rec != 1 else ''} at {span} "
                                 f"({rec_rows['bpm'].min():.0f}–{rec_rows['bpm'].max():.0f} bpm) fall in the two hours after your workout, "
                                 f"when heart rate normally stays elevated{trend}: expected recovery, not a warning sign.")
            exercise_note = ""
            if n_ex:
                ex_rows = batch_scored[batch_scored["risk_bucket"] == "during_exercise"]
                exercise_note = f"The day high ({ex_rows['bpm'].max():.0f} bpm) happened during the workout itself."
            sym_text = meta.get("symptoms") or ""

            if worst == "normal":
                lead = (f"**Normal** — your {primary['source'].lower()} value ({primary['bpm']:.0f} bpm) is within your typical range"
                        if (n_rec or n_ex) else
                        f"**Normal** — all {len(batch_scored)} readings on this screenshot are within your typical {primary['activity_state']} range")
                st.success(" ".join(x for x in [lead + ".", recovery_note, exercise_note, sym_text] if x))
            else:
                pieces = []
                alerts = above[~above["source"].str.startswith("Day high")]
                if len(alerts):
                    pieces.append(f"{len(alerts)} listed alert reading{'s' if len(alerts) != 1 else ''} "
                                  f"({alerts['bpm'].min():.0f}–{alerts['bpm'].max():.0f} bpm)")
                if above["source"].str.startswith("Day high").any():
                    pieces.append(f"the day high ({above[above['source'].str.startswith('Day high')]['bpm'].iloc[0]:.0f} bpm)")
                primary_verdict = ("is within your typical range" if primary["risk_bucket"] in ("normal", "expected_recovery")
                                   else f"is itself rated {bucket_label(primary['risk_bucket'])}")
                headline = "Flag for review" if worst == "flag_for_review" else "Monitor"
                if len(above):
                    msg = (f"**{headline}** — your {primary['source'].lower()} value ({primary['bpm']:.0f} bpm) {primary_verdict}, "
                           f"but {len(above)} of the {len(batch_scored)} readings on this screenshot are above it"
                           + (": " + " and ".join(pieces) if pieces else "") + ".")
                else:
                    msg = f"**{headline}** — your {primary['source'].lower()} value ({primary['bpm']:.0f} bpm) {primary_verdict}."
                counts = [f"{n_flag} flagged for review", f"{n_mon} to monitor"] + ([f"{n_rec} expected recovery"] if n_rec else []) + [f"{n_norm} normal"]
                msg = " ".join(x for x in [msg, recovery_note, exercise_note, sym_text, ", ".join(counts) + "."] if x)
                (st.error if worst == "flag_for_review" else st.warning)(msg)
            metric_cols = st.columns(4 if n_rec else 3)
            metric_cols[0].metric("Normal", n_norm)
            if n_rec:
                metric_cols[1].metric("Expected recovery", n_rec)
            metric_cols[-2].metric("Monitor", n_mon)
            metric_cols[-1].metric("Flag for review", n_flag)
            batch_cols = ["source", "timestamp", "bpm", "activity_state", "risk_bucket", "explanation"]
            show_readings(batch_scored, batch_cols)

                                                                       
                                                                              
                                                             
                                                                       
        batch = st.session_state.get("screenshot_batch")
        if batch and batch[0] == st.session_state.get("_last_wearable_file"):
            batch_df = batch[1]
            flagged = batch_df[batch_df["risk_bucket"] == "flag_for_review"].reset_index(drop=True)
        else:
            flagged = pd.DataFrame()
        if len(flagged) > 0:
            n_flag = len(flagged)
            st.warning(f"{n_flag} reading{'s' if n_flag != 1 else ''} flagged for review. "
                       "You can look at each one below and, where the data allows, run a quick rhythm check.")
            st.subheader("Take a closer look at a flagged reading")
            st.caption(
                "A watch measures your pulse, not your heart's electrical signal, so anything here is an "
                "approximate check - not an ECG. For a complete answer, record an ECG on your watch (or at a "
                "clinic) and upload it under Tier 2: ECG Snapshot."
            )
            for i, row in flagged.iterrows():
                when = fmt_when(pd.to_datetime(row.get("timestamp"))) if not _missing(row.get("timestamp")) else "Unknown time"
                activity = "at rest" if str(row.get("activity_state")) == "resting" else "during exercise"
                source = row.get("source") if "source" in row and not _missing(row.get("source")) else None
                title = f"{when} — {row['bpm']:.0f} bpm {activity}" + (f" · {source}" if source else "")
                with st.expander(title):
                    st.write(row["explanation"])

                    hrv = row.get("hrv_sdnn")
                    pnn50 = row.get("pnn50") if "pnn50" in row else None
                    facts = [f"Heart rate: **{row['bpm']:.0f} bpm** (about {60000 / row['bpm']:.0f} ms between beats)"]
                    facts.append("Heart-rate variability: " + (f"**{hrv:.0f} ms**" if not _missing(hrv) else "not available for this reading"))
                    facts.append("Beat-to-beat detail: " + ("available" if not _missing(pnn50) else "not available for this reading"))
                    st.markdown("\n".join(f"- {f}" for f in facts))

                    can_check = not _missing(hrv) and not _missing(pnn50) and row["bpm"]
                    if not can_check:
                        st.markdown(
                            "This reading only carries heart rate"
                            + (" (it came from a screenshot)" if source else "")
                            + ", which isn't enough to judge the heart's rhythm. **An ECG is the right next step:** "
                            "record one on your watch (about 30 seconds) and upload it in Tier 2."
                        )
                        st.button("Check with an ECG in Tier 2 →", key=f"goto_t2_{i}", type="primary",
                                  on_click=go_to_tier2, kwargs={"note": title})
                    elif st.button("Run quick rhythm check", key=f"handoff_{i}"):
                        ASSUMED_TYPICAL_QRS_MS = 90.0                                                          
                        result = ecg_model.classify({
                            "mean_rr_ms": 60000 / row["bpm"], "std_rr_ms": float(hrv), "pnn50": float(pnn50),
                            "qrs_width_ms": ASSUMED_TYPICAL_QRS_MS,
                        })
                        if result["is_abnormal"]:
                            kind = SUBTYPE_LABELS.get(result["subtype"], result["subtype"])
                            st.error(f"**Looks irregular** — the pattern most resembles {kind} "
                                     f"({result['confidence']:.0%} confidence). Not a diagnosis; worth discussing with a doctor.")
                            if result.get("guardrail"):
                                st.info(result["guardrail"])
                        else:
                            st.success(f"**Looks normal** ({result['confidence']:.0%} confidence).")
                            if result.get("guardrail"):
                                st.info(result["guardrail"])
                        st.caption(
                            "This quick check assumes a typical beat shape, because a watch can't measure it - "
                            "so some rhythm problems can't be seen this way. An ECG upload gives a complete check."
                        )
    else:
        st.info("Upload both CSVs (or switch to dummy/real data in the sidebar) to continue.")

                                                                             
                                
                                                                             

elif page == TIERS[1]:
                                                                        

                                                                          
                                                       
    if data_mode == "Use real WESAD/PPG-DaLiA data":
        if real_ppg_df is None or len(real_ppg_df) == 0:
            st.error(f"Cannot load real PPG data: {real_ppg_err or 'No data available'}. Please switch to dummy data in the sidebar.")
            st.stop()

        subject_col = "subject_id" if "subject_id" in real_ppg_df.columns else None
        subjects = sorted(real_ppg_df[subject_col].unique()) if subject_col else [None]
        chosen_subject = subjects[0] if subjects else None
        subject_df = real_ppg_df[real_ppg_df[subject_col] == chosen_subject] if subject_col else real_ppg_df
        subject_df = subject_df.sort_values("window_start_s") if "window_start_s" in subject_df.columns else subject_df
        split_idx = int(len(subject_df) * 0.7)
        history_df, new_df = subject_df.iloc[:split_idx].copy(), subject_df.iloc[split_idx:].copy()
        if "timestamp" not in history_df.columns:
            if "window_start_s" in history_df.columns:
                history_df["timestamp"] = pd.to_datetime(history_df["window_start_s"], unit="s", origin="2026-01-01")
                new_df["timestamp"] = pd.to_datetime(new_df["window_start_s"], unit="s", origin="2026-01-01")
            else:
                history_df["timestamp"] = pd.date_range("2026-01-01", periods=len(history_df), freq="1min")
                new_df["timestamp"] = pd.date_range("2026-01-02", periods=len(new_df), freq="1min")
    elif data_mode == "Generate dummy data":
        history_df = make_history()
        new_df = make_new_readings()
    else:
        c1, c2 = st.columns(2)
        with c1:
            history_file = st.file_uploader(
                "Upload historical readings CSV (columns: timestamp, bpm, hrv_rmssd, hrv_sdnn, activity_state)",
                type="csv", key="history_upload",
            )
        with c2:
            new_file = st.file_uploader("Upload new readings CSV (same columns)", type="csv", key="new_upload")
        history_df = pd.read_csv(history_file) if history_file else None
        new_df = pd.read_csv(new_file) if new_file else None

    if history_df is not None and new_df is not None:
        st.subheader("Your recent heart rate & HRV")
        chart_df = history_df.copy()
        chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
        recent_chart = chart_df.set_index("timestamp")[["bpm"]].tail(200)
        latest_bpm = float(recent_chart["bpm"].iloc[-1])
        latest_at = recent_chart.index[-1]
        lo, hi = float(recent_chart["bpm"].min()), float(recent_chart["bpm"].max())
        try:
            hero_card = st.container(border=True, key="hero_card")                                         
        except TypeError:                                               
            hero_card = st.container(border=True)
        with hero_card:
            st.markdown(
                f"""
                <div class="pw-hero">
                  <div class="pw-row">
                    <span class="pw-cat">&#10084;&#65038; Heart Rate</span>
                    <span class="grow"></span>
                    <span class="val">{fmt_when(latest_at)}</span>
                  </div>
                  <div class="pw-metrics">
                    <div><div class="pw-metric-label">Range</div><div class="pw-metric-value">{lo:.0f}–{hi:.0f}<small>BPM</small></div><div class="sub">Last {len(recent_chart)} readings</div></div>
                    <div><div class="pw-metric-label">Latest</div><div class="pw-metric-value">{latest_bpm:.0f}<small>BPM</small></div><div class="sub">Most recent reading</div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.line_chart(recent_chart, color="#FF2D55", height=220)

        st.subheader("Recent checks")
        model = PersonalBaselineModel(contamination=0.05).fit(history_df)
        scored = model.score(new_df)
        scored["explanation"] = scored.apply(model.explain, axis=1)

        bucket_counts = scored["risk_bucket"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Normal", int(bucket_counts.get("normal", 0)))
        c2.metric("Monitor", int(bucket_counts.get("monitor", 0)))
        c3.metric("Flag for review", int(bucket_counts.get("flag_for_review", 0)))

        step3_cols = ["timestamp", "bpm", "hrv_rmssd", "hrv_sdnn", "activity_state", "risk_bucket", "explanation"]
        show_readings(scored, step3_cols)

                                                                       
                                                                                  
                                                                       
        flagged = scored[scored["risk_bucket"] == "flag_for_review"]
        batch = st.session_state.get("screenshot_batch")
        if batch and batch[0] == st.session_state.get("_last_wearable_file"):
            batch_flagged = batch[1][batch[1]["risk_bucket"] == "flag_for_review"]
            if len(batch_flagged) > 0:
                flagged = pd.concat([flagged, batch_flagged], ignore_index=True)
        flagged = flagged.reset_index(drop=True)
        if len(flagged) > 0:
            n_flag = len(flagged)
            st.warning(f"{n_flag} reading{'s' if n_flag != 1 else ''} flagged for review. "
                       "You can look at each one below and, where the data allows, run a quick rhythm check.")
            st.subheader("Take a closer look at a flagged reading")
            st.caption(
                "A watch measures your pulse, not your heart's electrical signal, so anything here is an "
                "approximate check - not an ECG. For a complete answer, record an ECG on your watch (or at a "
                "clinic) and upload it under Tier 2: ECG Snapshot."
            )
            for i, row in flagged.iterrows():
                when = fmt_when(pd.to_datetime(row.get("timestamp"))) if not _missing(row.get("timestamp")) else "Unknown time"
                activity = "at rest" if str(row.get("activity_state")) == "resting" else "during exercise"
                source = row.get("source") if "source" in row and not _missing(row.get("source")) else None
                title = f"{when} — {row['bpm']:.0f} bpm {activity}" + (f" · {source}" if source else "")
                with st.expander(title):
                    st.write(row["explanation"])

                    hrv = row.get("hrv_sdnn")
                    pnn50 = row.get("pnn50") if "pnn50" in row else None
                    facts = [f"Heart rate: **{row['bpm']:.0f} bpm** (about {60000 / row['bpm']:.0f} ms between beats)"]
                    facts.append("Heart-rate variability: " + (f"**{hrv:.0f} ms**" if not _missing(hrv) else "not available for this reading"))
                    facts.append("Beat-to-beat detail: " + ("available" if not _missing(pnn50) else "not available for this reading"))
                    st.markdown("\n".join(f"- {f}" for f in facts))

                    can_check = not _missing(hrv) and not _missing(pnn50) and row["bpm"]
                    if not can_check:
                        st.markdown(
                            "This reading only carries heart rate"
                            + (" (it came from a screenshot)" if source else "")
                            + ", which isn't enough to judge the heart's rhythm. **An ECG is the right next step:** "
                            "record one on your watch (about 30 seconds) and upload it in Tier 2."
                        )
                        st.button("Check with an ECG in Tier 2 →", key=f"goto_t2_hist_{i}", type="primary",
                                  on_click=go_to_tier2, kwargs={"note": title})
                    elif st.button("Run quick rhythm check", key=f"handoff_hist_{i}"):
                        ASSUMED_TYPICAL_QRS_MS = 90.0
                        result = ecg_model.classify({
                            "mean_rr_ms": 60000 / row["bpm"], "std_rr_ms": float(hrv), "pnn50": float(pnn50),
                            "qrs_width_ms": ASSUMED_TYPICAL_QRS_MS,
                        })
                        if result["is_abnormal"]:
                            kind = SUBTYPE_LABELS.get(result["subtype"], result["subtype"])
                            st.error(f"**Looks irregular** — the pattern most resembles {kind} "
                                     f"({result['confidence']:.0%} confidence). Not a diagnosis; worth discussing with a doctor.")
                            if result.get("guardrail"):
                                st.info(result["guardrail"])
                        else:
                            st.success(f"**Looks normal** ({result['confidence']:.0%} confidence).")
                            if result.get("guardrail"):
                                st.info(result["guardrail"])
                        st.caption(
                            "This quick check assumes a typical beat shape, because a watch can't measure it - "
                            "so some rhythm problems can't be seen this way. An ECG upload gives a complete check."
                        )

    else:
        st.info("Upload both CSVs (or switch to dummy/real data in the sidebar) to continue.")

                                                                             
                                
                                                                             

elif page == TIERS[2]:
    st.subheader("On-demand ECG snapshot classification")
    if st.session_state.get("handoff_note"):
        st.info(
            f"**Following up on a flagged reading** — {st.session_state.handoff_note}.\n\n"
            "Record an ECG on your watch (Apple Watch: the ECG app; Samsung: Samsung Health Monitor → ECG; "
            "keep still for 30 seconds), then share the result as a PDF or screenshot and upload it below - "
            "or type the values from the report into the fields."
        )
        st.button("← Back to monitoring", key="back_to_tier1", on_click=go_to_tier1)

    st.markdown("#### Or enter your own feature values")
    st.caption(
        "Mean RR, RR variation, pNN50, and QRS duration are required for a classification - "
        "everything else is optional context that different ECG reports may or may not include, "
        "and isn't fed into the classifier (it was trained on the 4 required features only). "
        "Note: pNN50 needs a multi-minute recording, so a single resting-ECG snapshot report "
        "(like a standard 12-lead printout) typically won't include it - you'll likely need to "
        "enter that one field manually even when the rest auto-fill from an upload."
    )

    ecg_upload = st.file_uploader(
        "Upload a reading (CSV, Excel, image, or PDF)", type=["csv", "xlsx", "xls", "png", "jpg", "jpeg", "pdf"],
        key="ecg_single_upload",
    )
    if "ecg_manual" not in st.session_state:
        st.session_state.ecg_manual = {k: None for k in [f["key"] for f in ECG_FIELD_DEFS]}
        st.session_state.e_meanrr = None
        st.session_state.e_stdrr = None
        st.session_state.e_pnn50 = None
        st.session_state.e_qrs = None
        for k in ECG_OPTIONAL_KEYS:
            st.session_state[f"e_{k}"] = None

    if ecg_upload is not None:
        e_sig = f"{ecg_upload.name}_{ecg_upload.size}"
        if st.session_state.get("_last_ecg_file") != e_sig:
            st.session_state["_last_ecg_file"] = e_sig
            handle_file_upload(
                ecg_upload, ECG_FIELD_DEFS, "ecg_manual",
                subject="a heart-rhythm or ECG report",
            )
    else:
        st.session_state["_last_ecg_file"] = None
        st.session_state.pop("ecg_manual_banner", None)

    if "ecg_manual_banner" in st.session_state:
        b_type, b_msg, b_extra = st.session_state["ecg_manual_banner"]
        if b_type == "info":
            st.info(b_msg)
        elif b_type == "error":
            st.error(b_msg)
        else:
            st.success(b_msg + " Review the fields below before continuing.")

    em = st.session_state.ecg_manual
    c1, c2, c3, c4 = st.columns(4)
    if "e_meanrr" not in st.session_state:
        st.session_state.e_meanrr = None
    if "e_stdrr" not in st.session_state:
        st.session_state.e_stdrr = None
    if "e_pnn50" not in st.session_state:
        st.session_state.e_pnn50 = None
    if "e_qrs" not in st.session_state:
        st.session_state.e_qrs = None

    em["mean_rr_ms"] = c1.number_input("Mean RR (ms) *", key="e_meanrr", placeholder="e.g. 850")
    em["std_rr_ms"] = c2.number_input("RR variation (ms) *", key="e_stdrr", placeholder="e.g. 35")
    em["pnn50"] = c3.number_input("pNN50 (%) *", key="e_pnn50", placeholder="e.g. 18")
    em["qrs_width_ms"] = c4.number_input("QRS duration (ms) *", key="e_qrs", placeholder="e.g. 90")

    with st.expander("More optional fields (QTc, QRS width & P wave help the safety checks)"):
        ocols = st.columns(3)
        for idx, key in enumerate(ECG_OPTIONAL_KEYS):
            label = next(f["label"] for f in ECG_FIELD_DEFS if f["key"] == key)
            if f"e_{key}" not in st.session_state:
                st.session_state[f"e_{key}"] = None
            em[key] = ocols[idx % 3].number_input(label, key=f"e_{key}")

    if st.button("Classify these values"):
        core_values = {k: em.get(k) for k in ECG_CORE_KEYS}
        if any(v is None for v in core_values.values()):
            st.error("Mean RR, RR variation, pNN50, and QRS duration are all required to classify.")
        else:
            problems = implausible_fields(em)
            if problems:
                st.error(f"That doesn't look like a real reading — {'; '.join(problems)}. Double-check the numbers.")
            else:
                                                                                               
                result = ecg_model.classify(em)
                if result["is_abnormal"]:
                    st.error(f"**Abnormal** — the pattern most resembles {SUBTYPE_LABELS.get(result['subtype'], result['subtype'])} ({result['confidence']:.0%} confidence). Not a diagnosis; worth discussing with a doctor.")
                    if result.get("guardrail"):
                        st.info(result["guardrail"])
                else:
                    st.success(f"**Normal** ({result['confidence']:.0%} confidence).")
                    if result.get("guardrail"):
                        st.info(result["guardrail"])

                                                                             
                  
                                                                             
else:
    st.markdown('<div class="pw-eyebrow">Pulseware</div><h1 class="pw-title">About</h1><div class="pw-sub">Getting to know your heart, day by day.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("""
<div class="pw-group">
<div class="pw-row" style="flex-direction:column; align-items:flex-start; gap:6px; padding:14px 16px 14px 0;">
<span style="font-size:15px; font-weight:600;">What Pulseware does</span>
<span class="sub" style="white-space:normal;">Your smartwatch already measures your heart rate all day long. Pulseware turns that stream of numbers into something you can use: a sense of what's normal for <em>you</em>, and a gentle nudge when something looks different from your usual pattern. It is a decision-support prototype, not a medical device.</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.subheader("How everyday monitoring works")
    st.markdown("""
<div class="pw-group">
<div class="pw-row" style="flex-direction:column; align-items:flex-start; gap:6px; padding:14px 16px 14px 0;">
<span class="sub" style="white-space:normal;">Instead of comparing you to a generic chart, Pulseware learns your own resting and active patterns from your history and checks new readings against that personal baseline. Each new reading is scored as <b>Normal</b>, <b>Monitor</b>, or <b>Flag for review</b>, with a short explanation. Context matters: the two hours after exercise are rated as <b>Expected recovery</b>, and readings taken during a workout are marked <b>During exercise</b> — so normal recovery is not flagged. Any symptoms you report escalate the result.</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.subheader("How the ECG snapshot check works")
    st.markdown("""
<div class="pw-group">
<div class="pw-row" style="flex-direction:column; align-items:flex-start; gap:6px; padding:14px 16px 14px 0;">
<span class="sub" style="white-space:normal;">Rhythm Check sorts single-lead ECG snapshots into <b>Normal, PVC-like, AFib-like, Bradycardia-like</b>, or generic <b>Abnormal</b> using four trained features: mean RR, RR variation, pNN50, and QRS width. Safety guardrails also look at heart rate, QTc, QRS width, and P-wave (e.g., fast sinus tachycardia with prolonged QTc is flagged as Abnormal, and a fast rhythm with absent P-wave is reinterpreted as AFib with rapid response). The model is trained on the MIT-BIH Arrhythmia Database when available, otherwise on synthetic placeholder data — the on-screen banner tells you which.</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.subheader("Reading screenshots and PDFs with Vision AI")
    st.markdown("""
<div class="pw-group">
<div class="pw-row" style="flex-direction:column; align-items:flex-start; gap:6px; padding:14px 16px 14px 0;">
<span class="sub" style="white-space:normal;">Pulseware uses Google Gemini Vision AI to read smartwatch screenshots, ECG reports, and PDFs. It extracts BPM, HRV, RR intervals, QRS, QTc, P-wave axis/duration, and also looks for extra readings on the screen (abnormal alerts, day high) and workout context. Your Gemini API key stays in this browser only and is never sent to our servers.</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.subheader("Data and privacy")
    st.markdown("""
<div class="pw-group">
<div class="pw-row" style="flex-direction:column; align-items:flex-start; gap:6px; padding:14px 16px 14px 0;">
<span class="sub" style="white-space:normal;">• <b>Local only:</b> Your CSVs, screenshots, and baseline live in this browser session. Nothing is uploaded to a cloud database.<br>• <b>Real data when installed:</b> Place WESAD/PPG-DaLiA features at <code>ppg_wearable_features.csv</code> and MIT-BIH features at <code>mitbih_features.csv</code> (or <code>data_prep/</code>) to train on real data; otherwise the app runs on synthetic demo data with a clear warning.<br>• <b>Not a diagnostic device:</b> Pulseware is a research prototype to help you notice patterns and talk to your doctor. It does not replace professional medical advice.</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.caption("Pulseware · Apple Health-inspired UI · Heart Rhythm & Health Prediction · v0.2")
