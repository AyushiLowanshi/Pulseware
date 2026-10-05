import hashlib
import io
import json
import os
import re

import pandas as pd
from PIL import Image

GEMINI_GENAI_NEW = False
GEMINI_GENAI_OLD = False

try:
    from google import genai
    from google.genai import types
    GEMINI_GENAI_NEW = True
except ImportError:
    pass

try:
    import warnings as _w
    _w.filterwarnings("ignore", category=FutureWarning, module="google.*")
    import google.generativeai as legacy_genai
    GEMINI_GENAI_OLD = True
except ImportError:
    pass

try:
    import pymupdf as fitz  
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # fallback for older install
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

try:
    from ppg_preprocessing import preprocess_dataframe  
except ImportError:
    preprocess_dataframe = None

from field_aliases import (
    ECG_FIELD_DEFS,
    WEARABLE_FIELD_DEFS,
    PLAUSIBLE_RANGES,
    normalize_row_by_aliases,
    implausible_fields,
    null_out_implausible,
)


class UnrecognizedReadingError(RuntimeError):
    """Raised when the document was readable, but contained no heart-rhythm/ECG metrics."""
    pass


# --------------------------------------------------------------------------
# Prompts for Gemini Vision
# --------------------------------------------------------------------------

TIER1_VISION_PROMPT = """You are a medical data extraction specialist.
This is a screenshot from a wearable device or smartwatch wellness app (Apple Health, Garmin, Oura, Whoop, Fitbit, Samsung).
Read and extract ONLY the numbers directly visible and labeled on screen. Do not invent or extrapolate numbers.

Guidelines:
- "bpm": the primary heart-rate value. ALWAYS fill this when any heart rate is visible. Pick in
  this order: (1) a value labeled Latest / Current / Now; (2) otherwise the Resting heart rate
  (and copy it to "bpm_resting" too); (3) otherwise the most recent entry of the readings list.
  Never use High, Max, Min, Average, Range, or Walking Heart Rate Average as bpm. Examples:
  Apple Health "Latest 80 / Resting 60 / Walking Average 107" -> bpm 80, bpm_resting 60;
  Samsung Health "63 bpm Resting / 125 bpm High" + an alerts list -> bpm 63, bpm_resting 63,
  bpm_high 125, readings = the list.
- "hrv_rmssd": HRV in milliseconds ONLY if explicitly labeled RMSSD.
- "hrv_sdnn": General HRV (e.g. Apple Health's SDNN HRV) in milliseconds.
- "spo2": Blood oxygen saturation percentage (e.g. 98% -> 98).
- "resp_rate": Breathing or respiratory rate in breaths per minute.
- "skin_temp_c": Skin temperature DEVIATION from baseline (e.g. +0.3 or -0.5), not absolute body temp.
- "vo2max": VO2 max / Cardio fitness score.
- "readiness_score": Daily readiness score, body battery, or recovery score (0 to 100).
- "activity_state": "resting" or "exercise" if a workout state is evident, otherwise null.
- "bpm_resting": the Resting heart rate if one is labeled (e.g. "63 bpm Resting", "Resting Heart Rate 60"), else null.
- "bpm_high": the day's High / Max heart rate if one is labeled (e.g. "125 bpm High", "Max 142"), else null.
- "date": the date the screen refers to, exactly as written ("Today", "Sep 7", "2026-09-07"), else null.
- "activity_periods": every period of exercise or sleep the screen shows - a workout icon on the
  timeline (runner, cyclist, dumbbell, walker), a labelled workout, a sleep/Zz icon, or a clearly
  sustained block of heart rate above about 130 bpm on the chart. One object per period:
  {"type": "run|workout|walk|sleep|other", "start": "<time read off the axis, e.g. 8:30 am>",
  "end": "<time, e.g. 10:15 am>", "evidence": "<icon | label | chart block>"}. Times may be
  approximate - round to the nearest 15 minutes. Use [] if none.
- "bpm_high_time": the time the day's High occurred if it can be read (a label, or the position
  of the chart's highest peak on the time axis, e.g. "9:30 am"), else null.
- "readings": EVERY individually listed heart-rate measurement that has its own time on screen -
  rows in an "Abnormal heart rate alerts" list, a history list, notification rows, etc. One object
  per row, in the order shown: {"time": "<exactly as written, e.g. 1:16 PM>", "bpm": <number>,
  "label": "<name of the list or row, e.g. Abnormal alert>"}. Use [] if there is no such list.
  Do NOT read values off a chart line, and do not repeat the Resting/High summary numbers here.

Respond ONLY with a valid JSON object matching this schema (use null for any value not clearly labeled):
{
  "bpm": null,
  "hrv_rmssd": null,
  "hrv_sdnn": null,
  "spo2": null,
  "resp_rate": null,
  "skin_temp_c": null,
  "vo2max": null,
  "readiness_score": null,
  "activity_state": null,
  "bpm_resting": null,
  "bpm_high": null,
  "bpm_high_time": null,
  "date": null,
  "activity_periods": [],
  "readings": []
}"""

TIER2_VISION_PROMPT = """You are a clinical electrophysiology assistant.
This file is an ECG/EKG or heart rhythm report (e.g. 12-lead printout, rhythm strip, or CineECG summary).
Read and extract the numerical measurements labeled on the page.

Guidelines:
- "mean_rr_ms": Mean RR or NN interval in milliseconds. (If shown as 'RR Interval: 843 ± 9 ms', the mean is 843).
- "std_rr_ms": RR interval variation / standard deviation in milliseconds. (If shown as '843 ± 9 ms', the variation is 9).
- "pnn50": Percentage of successive RR differences > 50 ms (0-100%). Return null if not explicitly reported.
- "qrs_width_ms": QRS duration or width in milliseconds (e.g. 'QRS duration: 98 ms' -> 98).
- "heart_rate_bpm": Ventricular rate or heart rate in BPM (e.g. '71 BPM' -> 71).
- "pr_interval_ms": PR interval in milliseconds (e.g. '166 ± 1 ms' -> 166).
- "p_duration_ms": P wave duration in milliseconds (e.g. '112 ms' -> 112).
- "qt_ms": QT interval in milliseconds (e.g. '396 ms' -> 396).
- "qtc_ms": QTc corrected interval in milliseconds (e.g. '431 ms' -> 431).
- "p_axis_deg", "qrs_axis_deg", "t_axis_deg": Electrical axes in degrees if present.
- "rv5_mv", "sv1_mv": Voltages in mV if present.

If "mean_rr_ms" is not explicitly written but "heart_rate_bpm" is given, compute mean_rr_ms = round(60000.0 / heart_rate_bpm, 1).

Respond ONLY with a valid JSON object matching this schema (use null for any unlisted or missing field):
{
  "mean_rr_ms": null,
  "std_rr_ms": null,
  "pnn50": null,
  "qrs_width_ms": null,
  "heart_rate_bpm": null,
  "pr_interval_ms": null,
  "p_duration_ms": null,
  "qt_ms": null,
  "qtc_ms": null,
  "p_axis_deg": null,
  "qrs_axis_deg": null,
  "t_axis_deg": null,
  "rv5_mv": null,
  "sv1_mv": null
}"""


def read_structured_file(uploaded_file, field_defs):
    """
    Reads structured CSV or Excel files, normalizes columns by aliases,
    and returns (matched_dict, dataframe).
    If the file contains raw PPG + accelerometer, the 4-step live cleaning
    (filter -> peak -> RR -> motion check) is applied first.
    """
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    if len(df) == 0:
        return {}, df

    try:
        cols_lower = {c.lower(): c for c in df.columns}
        if preprocess_dataframe and "ppg" in cols_lower:
            ppg_col = cols_lower["ppg"]
            # Only run if this looks like raw signal (many rows, not already bpm)
            if len(df) > 20 and "bpm" not in cols_lower:
                df_clean = preprocess_dataframe(df, ppg_col=ppg_col)
                if df_clean is not None and "bpm" in df_clean.columns:
                    df = df_clean
                    if len(df) == 0:
                        return {}, df 
    except Exception:
        pass 

    if len(df) == 0:
        return {}, df
    first_row = df.iloc[0].to_dict()
    matched = normalize_row_by_aliases(first_row, field_defs)
    return matched, df



def _get_api_key():
   
    key = os.environ.get("GEMINI_API_KEY") or ""
    if not key:
        raise RuntimeError(
            "No Gemini API key configured. Get a free key at https://aistudio.google.com "
            "and set the GEMINI_API_KEY environment variable."
        )
    return key


PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",
]
_EXCLUDED_MODEL_HINTS = ("tts", "image", "audio", "live", "embedding", "omni", "customtools", "preview")
_MODEL_CACHE = {}


def _is_transient(err_text: str) -> bool:
    """404 = model gone, 503/500 = overloaded: in both cases just try the next model."""
    return any(tok in err_text for tok in ("404", "not found", "503", "unavailable", "high demand", "500", "internal"))


def _is_quota(err_text: str) -> bool:
    return any(tok in err_text for tok in ("429", "quota", "resource_exhausted", "rate limit"))


def _is_bad_key(err_text: str) -> bool:
    return "api key not valid" in err_text or "api_key_invalid" in err_text or "permission_denied" in err_text or "401" in err_text


def _candidate_models(client, api_key):
    """Ordered list of model names to try, discovered from the live API when possible."""
    if api_key in _MODEL_CACHE:
        return _MODEL_CACHE[api_key]
    discovered = []
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            if actions and "generateContent" not in actions:
                continue
            discovered.append(name)
    except Exception:
        discovered = []

    if discovered:
        ordered = [m for m in PREFERRED_MODELS if m in discovered]
        extras = sorted(
            n for n in discovered
            if n.startswith("gemini-") and "flash" in n and n not in ordered
            and not any(h in n for h in _EXCLUDED_MODEL_HINTS)
        )
        candidates = ordered + extras
    else:
        candidates = list(PREFERRED_MODELS)
    _MODEL_CACHE[api_key] = candidates
    return candidates


def _call_gemini_with_fallback(api_key, prompt, img_bytes, mime_type):
    """
    Calls Gemini API with automatic candidate model iteration to prevent
    '404 model not found' errors across all Google API versions.
    """
    if GEMINI_GENAI_NEW:
        client = genai.Client(api_key=api_key)
        last_err = None
        for model_name in _candidate_models(client, api_key):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if response.text and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                last_err = e
                err_text = str(e).lower()
                if _is_bad_key(err_text):
                    raise RuntimeError(
                        "Gemini rejected the API key. Check GEMINI_API_KEY (keys come from https://aistudio.google.com)."
                    ) from e
                if _is_quota(err_text) or _is_transient(err_text):
                    continue  
                raise
        if last_err and not GEMINI_GENAI_OLD:
            raise last_err

    if GEMINI_GENAI_OLD:
        legacy_genai.configure(api_key=api_key)
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Dynamically discover models supported by this key
        discovered_models = []
        try:
            for m in legacy_genai.list_models():
                methods = getattr(m, "supported_generation_methods", [])
                if "generateContent" in methods:
                    discovered_models.append(m.name.replace("models/", ""))
        except Exception:
            pass

        candidates = []
        for pref in ["2.5-flash", "flash-latest", "flash", "pro"]:
            for name in discovered_models:
                if pref in name.lower() and name not in candidates:
                    candidates.append(name)

        # Fallback names if discovery was empty
        standard_fallbacks = list(PREFERRED_MODELS)
        for f in standard_fallbacks:
            if f not in candidates:
                candidates.append(f)

        last_err = None
        for m_name in candidates:
            try:
                model = legacy_genai.GenerativeModel(
                    model_name=m_name,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model.generate_content([prompt, pil_img])
                if response.text and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                # If 404 (model not found), continue trying the next candidate
                if "404" in str(e) or "not found" in str(e).lower():
                    last_err = e
                    continue
                raise e

        if last_err:
            raise last_err

    raise RuntimeError(
        "Neither 'google-genai' nor 'google-generativeai' is installed.\n"
        "Run: python -m pip install google-genai"
    )


_GEMINI_CACHE = {}


def read_file_with_gemini(uploaded_file, field_defs, is_tier1=False):
    """
    Sends an image or PDF to Google Gemini Vision AI.
    Returns (matched_dict, raw_debug_json_string).
    """
    api_key = _get_api_key()
    prompt = TIER1_VISION_PROMPT if is_tier1 else TIER2_VISION_PROMPT

    raw_bytes = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()
    is_pdf = name.endswith(".pdf") or (getattr(uploaded_file, "type", "") == "application/pdf")


    cache_key = hashlib.sha256(raw_bytes + str(is_tier1).encode("utf-8")).hexdigest()

    if cache_key in _GEMINI_CACHE:
        raw_text = _GEMINI_CACHE[cache_key]
    else:
        # If PDF, rasterize the first page to PNG using PyMuPDF for highest quality
        if is_pdf:
            if not PYMUPDF_AVAILABLE:
                raise RuntimeError("Reading PDFs requires pymupdf. Install with: python -m pip install pymupdf")
            with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
                if len(doc) == 0:
                    raise RuntimeError("Uploaded PDF is empty.")
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                mime_type = "image/png"
        else:
            img_bytes = raw_bytes
            mime_type = getattr(uploaded_file, "type", "image/jpeg") or "image/jpeg"

        try:
            raw_text = _call_gemini_with_fallback(api_key, prompt, img_bytes, mime_type)
            _GEMINI_CACHE[cache_key] = raw_text
        except RuntimeError:
            raise
        except Exception as e:
            err_msg = str(e).lower()
            if _is_quota(err_msg):
                raise RuntimeError(
                    "Gemini API rate limit reached (Free Tier limit: 5 requests/minute). "
                    "Please wait ~30-45 seconds for your free quota to reset, then try again."
                )
            if _is_transient(err_msg):
                raise RuntimeError(
                    "Gemini is busy or the model list changed (every candidate model returned 404/503). "
                    "Try again in a moment."
                )
            raise RuntimeError(f"Gemini API request failed: {e}")

    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError:
        clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        extracted = json.loads(clean_json)

    null_out_implausible(extracted)
    _sanitize_screenshot_extras(extracted)
    # The primary value must not depend on how the model ordered things:
    # bpm -> else the Resting value -> else the most recent listed reading.
    if extracted.get("bpm") is None:
        if extracted.get("bpm_resting") is not None:
            extracted["bpm"] = extracted["bpm_resting"]
        elif extracted.get("readings"):
            extracted["bpm"] = extracted["readings"][-1]["bpm"]
    # Activity is only ever taken from the screen, never assumed: the model's
    # explicit workout/rest state, or the watch's own "Resting" label when that
    # is the value being checked. Anything else stays unknown for the user to set.
    act = str(extracted.get("activity_state") or "").strip().lower()
    if act not in ("resting", "exercise"):
        act = None
    if act is None and _num(extracted.get("bpm")) is not None and _num(extracted.get("bpm_resting")) is not None \
            and _num(extracted["bpm"]) == _num(extracted["bpm_resting"]):
        act = "resting"
    extracted["activity_state"] = act

    non_null_keys = [k for k, v in extracted.items() if v is not None and v != []]
    if not non_null_keys:
        raise UnrecognizedReadingError(
            "Gemini read the document, but did not find any labeled cardiac or ECG metrics.\n"
            f"[Debug JSON received]: {raw_text}"
        )

    matched = normalize_row_by_aliases(extracted, field_defs)
    return matched, json.dumps(extracted, indent=2)


_BPM_LO, _BPM_HI, _ = PLAUSIBLE_RANGES["bpm"]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN guard


def _sanitize_screenshot_extras(extracted: dict) -> dict:
    """Mutates `extracted`: keeps only plausible bpm_resting/bpm_high and well-formed readings."""
    for key in ("bpm_resting", "bpm_high"):
        val = _num(extracted.get(key))
        extracted[key] = val if val is not None and _BPM_LO <= val <= _BPM_HI else None
    date = extracted.get("date")
    extracted["date"] = str(date).strip() if isinstance(date, (str, int, float)) and str(date).strip() else None

    clean = []
    summary_values = {extracted.get("bpm_resting"), extracted.get("bpm_high")}
    for item in extracted.get("readings") or []:
        if not isinstance(item, dict):
            continue
        bpm = _num(item.get("bpm"))
        if bpm is None or not (_BPM_LO <= bpm <= _BPM_HI):
            continue
        time_str = str(item.get("time") or "").strip() or None
        if time_str is None and bpm in summary_values:
            continue  # the Resting/High summary numbers echoed back without a time
        label = str(item.get("label") or "").strip() or "Listed reading"
        if label.isupper():  # screen titles like "ABNORMAL HEART RATE ALERTS" -> sentence case
            label = label.capitalize()
        clean.append({"time": time_str, "bpm": bpm, "label": label})
    extracted["readings"] = clean

    hi_t = extracted.get("bpm_high_time")
    extracted["bpm_high_time"] = str(hi_t).strip() if isinstance(hi_t, (str, int, float)) and str(hi_t).strip() else None
    periods = []
    for item in extracted.get("activity_periods") or []:
        if not isinstance(item, dict):
            continue
        ptype = str(item.get("type") or "other").strip().lower()
        start = str(item.get("start") or "").strip() or None
        end = str(item.get("end") or "").strip() or None
        if start is None and end is None:
            continue
        periods.append({"type": ptype, "start": start, "end": end,
                        "evidence": str(item.get("evidence") or "").strip() or None})
    extracted["activity_periods"] = periods
    return extracted


EXERCISE_TYPES = {"run", "workout", "walk", "cycle", "ride", "swim", "hike", "exercise", "other"}


def detected_workout(extras):
    """
    The most recent exercise period the screenshot shows, as
    {"type", "start": Timestamp|None, "end": Timestamp, "label": "run from about 8:30 am to 10:15 am"}
    or None. Sleep periods are ignored. Times are approximate (read off a chart axis).
    """
    if not extras:
        return None
    best = None
    for p in extras.get("activity_periods") or []:
        if p.get("type") not in EXERCISE_TYPES or not (p.get("end") or p.get("start")):
            continue
        end_str = p.get("end") or p.get("start")
        end = parse_screenshot_time(end_str, extras.get("date"))
        start = parse_screenshot_time(p["start"], extras.get("date")) if p.get("start") else None
        if best is None or end > best["end"]:
            span = f"from about {p['start']} to {p['end']}" if p.get("start") and p.get("end") else f"around {end_str}"
            best = {"type": p.get("type"), "start": start, "end": end, "label": f"{p.get('type')} {span}"}
    return best


def reading_activity(matched, extra_info):
    """
    The activity the uploaded reading was taken in - "resting" / "exercise" -
    when the file actually says so (Gemini's activity_state, or an activity
    column in a CSV/Excel row); otherwise None, so the form does not silently
    assume rest.
    """
    candidates = [matched.get("activity_state") if isinstance(matched, dict) else None]
    if isinstance(extra_info, str):
        try:
            candidates.append(json.loads(extra_info).get("activity_state"))
        except (ValueError, AttributeError):
            pass
    elif isinstance(extra_info, pd.DataFrame) and len(extra_info):
        row = extra_info.iloc[0]
        for col in extra_info.columns:
            if str(col).strip().lower().replace(" ", "_") in ("activity_state", "activity", "state"):
                candidates.append(row[col])
                break
    for c in candidates:
        v = str(c or "").strip().lower()
        if v in ("resting", "rest", "at rest"):
            return "resting"
        if v in ("exercise", "exercising", "workout", "active"):
            return "exercise"
    return None


def screenshot_extras(extra_info):
    """
    Given the second value returned by read_any_file for an image/PDF (the raw
    JSON string), return {"readings": [...], "bpm_high", "bpm_resting", "date"}
    or None when there is nothing beyond the primary reading (or the input was
    a CSV/Excel DataFrame).
    """
    if not isinstance(extra_info, str):
        return None
    try:
        data = json.loads(extra_info)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    data = _sanitize_screenshot_extras(dict(data))
    extras = {
        "readings": data.get("readings") or [],
        "bpm_high": data.get("bpm_high"),
        "bpm_high_time": data.get("bpm_high_time"),
        "bpm_resting": data.get("bpm_resting"),
        "date": data.get("date"),
        "activity_periods": data.get("activity_periods") or [],
    }
    if not extras["readings"] and extras["bpm_high"] is None and not extras["activity_periods"]:
        return None
    return extras


_TIME_RE = re.compile(r"(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*([AaPp]\.?[Mm]\.?)?")


def _base_date(date_str):
    """'Today' / None -> today; 'Yesterday' -> yesterday; anything pandas can parse -> that date."""
    today = pd.Timestamp.now().normalize()
    if not date_str:
        return today
    low = date_str.strip().lower()
    if low in ("today", "now"):
        return today
    if low == "yesterday":
        return today - pd.Timedelta(days=1)
    for candidate in (date_str, f"{date_str} {today.year}"):
        try:
            parsed = pd.to_datetime(candidate, errors="raise")
            if parsed.year < 2000:  # bare "Sep 7" parsed without a year
                parsed = parsed.replace(year=today.year)
            return parsed.normalize()
        except (ValueError, TypeError, OverflowError):
            continue
    return today


def parse_screenshot_time(time_str, date_str=None):
    """'1:16 PM' (+ optional date) -> pandas Timestamp; falls back to now when unparseable."""
    base = _base_date(date_str)
    if not time_str:
        return pd.Timestamp.now()
    m = _TIME_RE.search(time_str)
    if not m:
        return pd.Timestamp.now()
    hour = int(m.group(1)); minute = int(m.group(2) or 0); second = int(m.group(3) or 0)
    ampm = (m.group(4) or "").replace(".", "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return pd.Timestamp.now()
    return base + pd.Timedelta(hours=hour, minutes=minute, seconds=second)


def screenshot_readings_to_rows(extras, include_high=True, activity_state="resting"):
    """
    Turn screenshot_extras() output into rows shaped like a wearable CSV
    (timestamp, bpm, hrv_*, activity_state) plus a human-readable 'source'.
    The day High has no timestamp or activity of its own; it is dated to the
    screen's day and labelled so the caller can caveat it.
    """
    rows = []
    for r in extras.get("readings") or []:
        rows.append({
            "timestamp": parse_screenshot_time(r.get("time"), extras.get("date")),
            "bpm": float(r["bpm"]), "hrv_rmssd": None, "hrv_sdnn": None,
            "activity_state": activity_state,
            "source": f"{r.get('label') or 'Listed reading'} {r.get('time') or ''}".strip(),
        })
    if include_high and extras.get("bpm_high") is not None:
        last_ts = max((row["timestamp"] for row in rows), default=None)
        if extras.get("bpm_high_time"):
            ts = parse_screenshot_time(extras["bpm_high_time"], extras.get("date"))
        elif last_ts is not None:
            ts = last_ts + pd.Timedelta(minutes=1)
        else:
            ts = _base_date(extras.get("date")) + pd.Timedelta(hours=23, minutes=59)
        rows.append({
            "timestamp": ts,
            "bpm": float(extras["bpm_high"]), "hrv_rmssd": None, "hrv_sdnn": None,
            "activity_state": activity_state,
            "source": "Day high (activity unknown)",
        })
    return rows

def read_any_file(uploaded_file, field_defs, subject="a wearable device or heart-rhythm report"):
    """
    Dispatches to structured reader (CSV/Excel) or Vision AI (Images/PDFs).
    Returns (matched_dict, extra_info).
    For CSV/Excel, extra_info is the full DataFrame.
    For Image/PDF, extra_info is the formatted JSON debug string from Gemini.
    """
    name = (uploaded_file.name or "").lower()
    file_type = getattr(uploaded_file, "type", "") or ""

    if name.endswith((".csv", ".xlsx", ".xls")):
        return read_structured_file(uploaded_file, field_defs)
    elif name.endswith(".pdf") or file_type.startswith("image/") or file_type == "application/pdf":
        is_tier1 = "bpm" in [f["key"] for f in field_defs]
        return read_file_with_gemini(uploaded_file, field_defs, is_tier1=is_tier1)
    else:
        raise RuntimeError(f"Unsupported file format: {uploaded_file.name}. Please upload CSV, Excel, Image, or PDF.")
