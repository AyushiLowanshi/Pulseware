# Heart Rhythm & Health Prediction — UI redesign

Both front-ends were restyled to follow **Apple Health's** design language:
a grouped light-gray canvas, white inset cards with hairline separators (no
borders, no shadows), system type with a large bold title and small uppercase
section headers, one category tint (heart pink `#FF2D55`) plus iOS blue for
actions, and semantic green / orange / red used *only* for status.

All features, flows, model logic, prompts, CSV/alias handling and Gemini
reading are unchanged — only presentation and layout moved.

```
heart-app/
├── app.py                    Streamlit app (restyled; logic untouched)
├── .streamlit/config.toml    Streamlit theme (canvas, card, accent colours, font)
├── anomaly_model.py          unchanged
├── ecg_classifier.py         unchanged
├── field_aliases.py          unchanged
├── file_reading.py           unchanged (API key now read from GEMINI_API_KEY first)
├── baseline_threshold.py     unchanged
├── generate_dummy_data.py    sample-data generators the Streamlit app imports
├── requirements.txt
├── screenshots/              what the redesign looks like
└── web-app/                  React app (Vite)
    ├── src/App.jsx           redesigned UI, same state/logic
    ├── src/styles.css        the design system (new — replaces inline styles)
    ├── src/main.jsx          imports styles.css
    ├── index.html, vite.config.js, package.json
    └── .env.example          VITE_GEMINI_API_KEY for screenshot / PDF reading
```

## Run

```bash
# React
cd web-app
npm install
cp .env.example .env        # add your Gemini key if you want image/PDF reading
npm run dev                 # http://localhost:5173

# Streamlit
pip install -r requirements.txt
streamlit run app.py        # http://localhost:8501
```

If you are dropping this into your existing repo, the only React files you
need are `web-app/src/App.jsx`, `web-app/src/styles.css` and the one-line
`import "./styles.css";` in `main.jsx`. For Streamlit, copy `app.py`,
`.streamlit/config.toml` and `generate_dummy_data.py` (if you don't already
have one).

## What changed, and why it reads as designed rather than generated

| Before | After |
| --- | --- |
| Google-Fonts pairing (Fraunces serif + IBM Plex) | System font stack (SF Pro on Apple devices, Segoe/Inter elsewhere) — what Health actually uses |
| Every element in a 1 px-bordered rounded card | White inset groups on a gray canvas; hairline separators *inside* groups, no borders around them |
| Three equal "stat cards" | One hero **Heart Rate** card: `RANGE 50–155 BPM`, `LATEST`, D / W / M segmented control, min–max range bars with the y-axis on the right (Health's chart idiom) |
| Icon + label pill badges | Dot + coloured text status (`● Normal`, `● Flag for review`) and a legend row |
| Bordered inputs in a grid | iOS inline form rows: label left, value right, unit in gray; segmented control for Resting / Exercise; disclosure row for optional metrics |
| "What do these mean?" toggle panel | `ABOUT THESE MEASUREMENTS` grouped list with expandable rows |
| Top nav bar with three buttons | iPad-Health-style sidebar on desktop, iOS bottom tab bar on phones (the check table also becomes a list on phones) |
| Centered card login with icon-in-field inputs | App-icon + grouped-row sign-in, filled blue button, privacy footnote |
| Emoji title, Streamlit chrome, coloured alert blocks | Date eyebrow + large title, tabs drawn as a segmented control, white alerts, metric cards |

## Notes

* `file_reading.py` had a Gemini API key hard-coded. It now reads
  `GEMINI_API_KEY` from the environment first and only falls back to the
  literal — move the key out of the file and rotate it before sharing the repo.
* Gemini model names are no longer hard-coded. `gemini-1.5-flash` (the old
  default) has been retired by Google and returns 404, so both apps now ask
  the API for its current model list and try, in order: `gemini-2.5-flash` →
  `gemini-flash-latest` → `gemini-2.5-flash-lite` → `gemini-flash-lite-latest`
  → `gemini-2.5-pro` → any other `gemini-*flash*` model the key can use. A
  404/429/500/503 moves on to the next model; an invalid key fails immediately
  with a clear message.
* The React app takes the key from the "Gemini API key" row (Monitoring →
  Vision AI, also inside the Rhythm Check form; stored in `localStorage`
  only), falling back to `VITE_GEMINI_API_KEY` in `web-app/.env`. A
  `web-app/.env` with the same key as `file_reading.py` was created for the
  preview — it is gitignored, and the key should still be rotated.
* Prompt rule for wearable screenshots that show several heart-rate numbers
  (identical in both apps, so they agree): `bpm` = the most recent single
  reading (Latest / Current / Now) if one is shown, otherwise the Resting
  heart rate; High / Max / Min / Average / Walking Average are never used.
  Apple Health "Latest 80 / Resting 60 / Walking Avg 107" → 80; Samsung
  Health "63 bpm Resting / 125 bpm High" → 63. Apple's HRV is SDNN, so it
  lands in `hrv_sdnn`; VO₂ max goes to `vo2max`.
* Screens that carry more than one heart-rate number are no longer reduced to
  a single value. The Tier 1 prompt also returns `bpm_resting`, `bpm_high`,
  `date` and `readings[]` (every individually timestamped value on screen —
  e.g. Samsung Health's "Abnormal heart rate alerts" list). The primary value
  still fills the form (bpm → else Resting → else the most recent listed
  reading, enforced in code), an **"Also on this screenshot"** panel lists
  what else was found, and the single check button ("Check this screenshot
  (N readings)" / "Add all N readings from this screenshot") scores the form
  value *and* every listed reading *and* the day high together — the
  headline verdict is the worst of them, with a per-reading table (Python: `screenshot_extras()` /
  `screenshot_readings_to_rows()` in `file_reading.py`; React:
  `sanitizeScreenshotExtras()` / `screenshotReadingsToEntries()`). Listed
  readings are dated to the screen's day; the day high has no time or
  activity of its own, so it is dated after the last alert, checked as a
  resting reading, and labelled as such (a checkbox lets you leave it out).
* **Resting guardrails** (`RESTING_GUARDRAILS` in `anomaly_model.py`, mirrored
  in `App.jsx`): after the personal-baseline model has scored a reading, a
  *resting* reading at or above 100 bpm / at or below 50 bpm is never rated
  below Monitor, and at or above 120 / at or below 40 is Flag for review.
  The baseline can rate a reading higher than these, never lower. Needed
  because a baseline can only say "unusual for this history" — with the
  WESAD/PPG-DaLiA option the "resting" history includes sitting, working,
  lunch, driving and stress windows well past 100 bpm, so the IsolationForest
  rated 125 bpm at rest as normal. Exercise readings are untouched. Expect
  more Monitor rows in Step 3 for that dataset's own "resting" windows, for
  the same reason.
* **No false flags after exercise (both apps).** A watch's "abnormal heart
  rate" rule fires above 100 bpm while inactive and has no idea a run just
  ended, so it alarms during the 1–2 hours of normal post-exercise recovery
  (the classic Garmin/Samsung forum complaint). Pulseware asks — and reads the
  workout off the screenshot when the timeline shows one — so it can tell
  recovery from unexplained resting tachycardia:
  * *Context for this check*: "Did you exercise today?" (No / Yes + end time /
    Not sure; prefilled from a detected workout, marked approximate) and
    "Any symptoms today?".
  * Elevated resting readings 0–2 h after the workout → **Expected recovery**
    (not Monitor/Flag) unless ≥120 bpm (Monitor), ≥140 bpm (Flag), or rising
    ≥10 bpm across the window instead of settling (last one → Monitor).
    2–3 h after: ≥110 bpm → Monitor, 100–109 → still expected recovery with a
    "should settle within a few hours" note. Beyond 3 h the normal rules apply.
  * Readings inside the workout → **During exercise**, not judged against
    resting limits. When the user answers Yes, the day high always counts as
    part of the workout (chart times are read approximately, so a high that
    "misses" the window by an hour is not turned into a false flag). A day
    high above every listed alert defaults to "not a resting reading" even
    without the exercise answer — the watch would have alerted on it otherwise.
  * Symptoms escalate whatever the numbers say: chest pain / fainting → Flag
    for review with a seek-care line; palpitations, dizziness, breathlessness,
    unusual fatigue → at least Monitor.
  * Constants: `RECOVERY_WINDOW_MIN`, `LATE_RECOVERY_MIN`, `RECOVERY_LIMITS`,
    `RECOVERY_RISE_BPM`, `SYMPTOM_RULES` in `anomaly_model.py` and `App.jsx`
    (`apply_recovery_context` / `applyRecoveryContext`). Gemini prompt fields:
    `activity_periods`, `bpm_high_time`. Verified: Garmin post-run screenshot
    (60 resting, alerts 103–112 at 10:51–11:58, day high 171) → Normal + 4
    expected recovery + day high during exercise in both apps.
* **Activity is never assumed (both apps).** The single-reading form's
  Activity field has no default. It is filled from the upload only when the
  file actually says so — Gemini's `activity_state` (a workout screen), the
  watch's own *Resting* label when that is the value being checked, or an
  activity column in a CSV/Excel row (`reading_activity()` in
  `file_reading.py`) — and then shows up in the "Found:" banner. Otherwise it
  stays "Choose…" and the check asks for it instead of quietly scoring the
  reading as resting. Listed watch alerts are still scored as resting: the
  watch only raises them while you are inactive.
* **User-facing copy only.** Tables use plain headers (Reading, Time, BPM,
  Activity, Status, Why) and human statuses ("Flag for review", never
  `flag_for_review`); the Why column is shown in full (wrapped rows via
  `show_readings()` in Streamlit — the scrolling grid is only used for lists
  over 40 rows — and a wrapping column in the web app); how the rules work
  lives in the ⓘ tooltips and this README, not in captions on the page; "Found:" banners name fields by label; Step 4 shows a
  plain-language summary of a flagged reading instead of a JSON dump, and
  the quick rhythm check button appears only when the reading actually
  carries the HRV + beat-to-beat detail the ECG classifier needs (it never
  did for watch/screenshot readings, so the old button always errored) —
  otherwise the panel says so and offers **"Check with an ECG in Tier 2 →"**,
  which jumps to the ECG section with a note about which reading sent you
  there (plus a "Back to monitoring" button). To make that jump possible the
  Tier 1 / Tier 2 switcher is a session-state-driven control instead of
  `st.tabs` (which cannot be switched programmatically); it is styled the
  same, with CSS that only hides the radio circle (verified on Streamlit
  1.45, 1.55, 1.60 and 1.64, whose radio markup all differs). Model
  subtypes are worded ("early or extra beats (PVC-like)"). Setup/instruction
  text that mentions files or scripts lives here in the README, not in the UI:
  real MIT-BIH features come from `data_prep/download_preprocess_mitbih_features.py`,
  real wearable features from `data_prep/ppg_wearable_features.csv`.
* `anomaly_model.py` now fills a feature column that is entirely missing
  (e.g. HRV on a screenshot reading) with that feature's baseline mean before
  scoring. Previously the NaNs made IsolationForest return "normal" for
  every bpm-only reading — even 125 bpm at rest.
* `.streamlit/config.toml` uses a system-font stack for `theme.font`. Streamlit
  ≥ 1.45 applies it directly; older versions ignore it and the CSS in `app.py`
  still forces the same font.
* The Streamlit app wraps the heart-rate chart in `st.container(border=True,
  key="hero_card")` so CSS can turn it into a card; on Streamlit < 1.39 it
  falls back to a plain bordered container automatically.
