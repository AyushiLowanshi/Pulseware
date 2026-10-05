import React, { useState, useMemo, useCallback } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import {
  Heart, Activity, Info, ChevronRight, ChevronDown, Image as ImageIcon, Upload,
  Loader2, LogOut, Lock, Check, AlertCircle, FileText, Clock,
} from "lucide-react";
import Papa from "papaparse";

/* ------------------------------------------------------------------ */
/* Dummy data generators                                               */
/* ------------------------------------------------------------------ */
function gaussian(mean, std, rand) {
  const u1 = Math.max(rand(), 1e-9);
  const u2 = rand();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}
function seededRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}
function generateDummyHistory() {
  const rand = seededRandom(42);
  const rows = [];
  const start = new Date("2026-08-24T00:00:00");
  for (let day = 0; day < 14; day++) {
    for (let i = 0; i < 40; i++) {
      const t = new Date(start.getTime() + day * 86400000 + i * 30 * 60000);
      rows.push({ timestamp: t.toISOString(), bpm: gaussian(66, 5, rand), hrv_rmssd: gaussian(45, 8, rand), hrv_sdnn: gaussian(50, 9, rand), activity_state: "resting" });
    }
    for (let i = 0; i < 6; i++) {
      const t = new Date(start.getTime() + day * 86400000 + (7 + i) * 3600000);
      rows.push({ timestamp: t.toISOString(), bpm: gaussian(128, 12, rand), hrv_rmssd: gaussian(18, 5, rand), hrv_sdnn: gaussian(22, 6, rand), activity_state: "exercise" });
    }
  }
  rows.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  return rows;
}
function generateDummyNewReadings() {
  const base = new Date("2026-09-07T08:00:00");
  const raw = [
    { bpm: 65, hrv_rmssd: 44, hrv_sdnn: 49, activity_state: "resting" },
    { bpm: 68, hrv_rmssd: 43, hrv_sdnn: 51, activity_state: "resting" },
    { bpm: 63, hrv_rmssd: 47, hrv_sdnn: 52, activity_state: "resting" },
    { bpm: 128, hrv_rmssd: 11, hrv_sdnn: 14, activity_state: "resting" },
    { bpm: 66, hrv_rmssd: 45, hrv_sdnn: 50, activity_state: "resting" },
    { bpm: 125, hrv_rmssd: 19, hrv_sdnn: 21, activity_state: "exercise" },
    { bpm: 132, hrv_rmssd: 17, hrv_sdnn: 20, activity_state: "exercise" },
    { bpm: 58, hrv_rmssd: 46, hrv_sdnn: 48, activity_state: "exercise" },
    { bpm: 130, hrv_rmssd: 18, hrv_sdnn: 22, activity_state: "exercise" },
    { bpm: 64, hrv_rmssd: 46, hrv_sdnn: 50, activity_state: "resting" },
  ];
  return raw.map((r, i) => ({ timestamp: new Date(base.getTime() + i * 15 * 60000).toISOString(), ...r }));
}
const SAMPLE_ECG_READINGS = [
  { hint: "Typical reading", mean_rr_ms: 860, std_rr_ms: 32, pnn50: 17, qrs_width_ms: 88 },
  { hint: "Irregular beat sample", mean_rr_ms: 800, std_rr_ms: 95, pnn50: 11, qrs_width_ms: 145 },
  { hint: "Highly irregular sample", mean_rr_ms: 690, std_rr_ms: 150, pnn50: 38, qrs_width_ms: 93 },
  { hint: "Slow, steady sample", mean_rr_ms: 1210, std_rr_ms: 28, pnn50: 14, qrs_width_ms: 90 },
];

/* ------------------------------------------------------------------ */
/* Field aliases                                                       */
/* ------------------------------------------------------------------ */
function normalizeKey(s) {
  return String(s).trim().toLowerCase().replace(/[\s_\-().%]+/g, "");
}
function normalizeRowByAliases(row, fieldDefs) {
  const lookup = {};
  Object.keys(row).forEach((k) => { lookup[normalizeKey(k)] = row[k]; });
  const out = {};
  fieldDefs.forEach(({ key, aliases }) => {
    for (const name of [key, ...aliases]) {
      const norm = normalizeKey(name);
      if (lookup[norm] !== undefined && lookup[norm] !== "" && lookup[norm] !== null) {
        out[key] = lookup[norm];
        break;
      }
    }
  });
  return out;
}

const WEARABLE_FIELD_DEFS = [
  { key: "bpm", label: "BPM", unit: "", aliases: ["heart rate", "hr", "heartrate"] },
  { key: "hrv_rmssd", label: "HRV (RMSSD)", unit: "ms", aliases: ["rmssd", "hrv rmssd", "hrv (rmssd)"] },
  { key: "hrv_sdnn", label: "HRV (SDNN)", unit: "ms", aliases: ["sdnn", "hrv sdnn", "hrv (sdnn)", "heart rate variability"] },
  { key: "spo2", label: "SpO2", unit: "%", aliases: ["spo2", "blood oxygen", "oxygen saturation", "o2 saturation"] },
  { key: "resp_rate", label: "Respiratory rate", unit: "breaths/min", aliases: ["respiratory rate", "breathing rate", "resp rate"] },
  { key: "skin_temp_c", label: "Skin temp (\u0394\u00B0C)", unit: "\u00B0C", aliases: ["skin temperature", "temperature deviation", "body temperature deviation", "wrist temperature"] },
  { key: "vo2max", label: "VO2 max", unit: "", aliases: ["vo2max", "vo2 max", "cardio fitness"] },
  { key: "readiness_score", label: "Readiness / recovery score", unit: "", aliases: ["readiness score", "recovery score", "body battery", "readiness", "recovery"] },
];
const WEARABLE_CORE_KEYS = ["bpm", "hrv_rmssd", "hrv_sdnn"];
const WEARABLE_OPTIONAL_KEYS = WEARABLE_FIELD_DEFS.map((f) => f.key).filter((k) => !WEARABLE_CORE_KEYS.includes(k));

const ECG_FIELD_DEFS = [
  { key: "mean_rr_ms", label: "Mean RR / NN interval", unit: "ms", aliases: ["rr interval", "nn interval", "r-r", "rr int", "mean rr", "rr", "nn"] },
  { key: "std_rr_ms", label: "RR variation", unit: "ms", aliases: ["rr variation", "sdrr", "rr sd", "rr std"] },
  { key: "pnn50", label: "pNN50", unit: "%", aliases: ["pnn50", "pnn 50"] },
  { key: "qrs_width_ms", label: "QRS duration", unit: "ms", aliases: ["qrs duration", "qrs width", "qrsd", "qrs dur"] },
  { key: "heart_rate_bpm", label: "Heart rate", unit: "bpm", aliases: ["heart rate", "hr", "vent. rate", "ventricular rate"] },
  { key: "pr_interval_ms", label: "PR interval", unit: "ms", aliases: ["pr interval", "p-r interval", "pr int"] },
  { key: "p_duration_ms", label: "P duration", unit: "ms", aliases: ["p duration", "p dur", "p wave duration"] },
  { key: "qt_ms", label: "QT interval", unit: "ms", aliases: ["qt interval", "qt int", "qt"] },
  { key: "qtc_ms", label: "QTc", unit: "ms", aliases: ["qtc", "qtc bazett", "qtcb", "corrected qt"] },
  { key: "p_axis_deg", label: "P axis", unit: "\u00B0", aliases: ["p axis"] },
  { key: "qrs_axis_deg", label: "QRS axis", unit: "\u00B0", aliases: ["qrs axis"] },
  { key: "t_axis_deg", label: "T axis", unit: "\u00B0", aliases: ["t axis"] },
  { key: "rv5_mv", label: "RV5", unit: "mV", aliases: ["rv5", "r wave v5"] },
  { key: "sv1_mv", label: "SV1", unit: "mV", aliases: ["sv1", "s wave v1"] },
];
const ECG_CORE_KEYS = ["mean_rr_ms", "std_rr_ms", "pnn50", "qrs_width_ms"];
const ECG_EXTRA_KEYS = ECG_FIELD_DEFS.map((f) => f.key).filter((k) => !ECG_CORE_KEYS.includes(k));

/* ------------------------------------------------------------------ */
/* Physiological ranges                                                */
/* ------------------------------------------------------------------ */
const PLAUSIBLE_RANGES = {
  bpm: { min: 30, max: 220, label: "BPM" },
  hrv_rmssd: { min: 0, max: 300, label: "HRV (RMSSD)" },
  hrv_sdnn: { min: 0, max: 300, label: "HRV (SDNN)" },
  spo2: { min: 60, max: 100, label: "SpO2" },
  resp_rate: { min: 4, max: 40, label: "Respiratory rate" },
  skin_temp_c: { min: -6, max: 6, label: "Skin temp deviation" },
  vo2max: { min: 10, max: 85, label: "VO2 max" },
  readiness_score: { min: 0, max: 100, label: "Readiness / recovery score" },
  mean_rr_ms: { min: 250, max: 2200, label: "Mean RR" },
  std_rr_ms: { min: 0, max: 400, label: "RR variation" },
  pnn50: { min: 0, max: 100, label: "pNN50" },
  qrs_width_ms: { min: 40, max: 220, label: "QRS duration" },
  heart_rate_bpm: { min: 20, max: 300, label: "Heart rate" },
  pr_interval_ms: { min: 80, max: 400, label: "PR interval" },
  p_duration_ms: { min: 30, max: 200, label: "P duration" },
  qt_ms: { min: 200, max: 700, label: "QT interval" },
  qtc_ms: { min: 250, max: 700, label: "QTc" },
  p_axis_deg: { min: -180, max: 180, label: "P axis" },
  qrs_axis_deg: { min: -180, max: 180, label: "QRS axis" },
  t_axis_deg: { min: -180, max: 180, label: "T axis" },
  rv5_mv: { min: 0, max: 6, label: "RV5" },
  sv1_mv: { min: 0, max: 6, label: "SV1" },
};
function implausibleFields(reading) {
  return Object.entries(PLAUSIBLE_RANGES)
    .filter(([key, { min, max }]) => reading[key] != null && (reading[key] < min || reading[key] > max))
    .map(([key, { label, min, max }]) => `${label} of ${reading[key]} (expected ${min}\u2013${max})`);
}
function nullOutImplausible(reading) {
  Object.keys(PLAUSIBLE_RANGES).forEach((k) => {
    if (reading[k] != null && (reading[k] < PLAUSIBLE_RANGES[k].min || reading[k] > PLAUSIBLE_RANGES[k].max)) reading[k] = null;
  });
  return reading;
}

/* ------------------------------------------------------------------ */
/* Tier 1: personal baseline                                           */
/* ------------------------------------------------------------------ */
function buildBaseline(history) {
  const byState = {};
  history.forEach((row) => {
    if (!byState[row.activity_state]) byState[row.activity_state] = [];
    byState[row.activity_state].push(row);
  });
  const baseline = {};
  Object.entries(byState).forEach(([state, rows]) => {
    if (rows.length < 15) return;
    const stats = {};
    WEARABLE_FIELD_DEFS.map((f) => f.key).forEach((f) => {
      const vals = rows.map((r) => r[f]).filter((v) => v != null && !Number.isNaN(v));
      if (vals.length === 0) { stats[f] = null; return; }
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
      stats[f] = { mean, std: Math.max(Math.sqrt(variance), 0.5) };
    });
    baseline[state] = { stats, n: rows.length };
  });
  return baseline;
}
function bucketFor(zRms) {
  if (zRms < 1.6) return "normal";
  if (zRms < 3) return "monitor";
  return "flag_for_review";
}
/* Clinical guardrails for RESTING readings, applied after the personal
   baseline. A baseline can only say "unusual for this history"; if the history
   is wide, 125 bpm at rest can pass as normal. These absolute limits make sure
   a resting reading beyond the standard tachycardia / bradycardia cut-offs is
   never labelled normal. The baseline can rate a reading higher, never lower. */
const RESTING_GUARDRAILS = {
  monitor: { high: 100, low: 50 },          // >= 100 bpm at rest = tachycardia, <= 50 = bradycardia
  flag_for_review: { high: 120, low: 40 },
};
const BUCKET_RANK = { insufficient_baseline: 0, normal: 0, expected_recovery: 0, during_exercise: 0, monitor: 1, flag_for_review: 2 };

/* Post-exercise recovery. A watch's "abnormal HR" rule (>100 bpm while
   inactive) has no idea a run just ended, so it fires during the 1-2 hours in
   which heart rate normally stays elevated. We do know — the user or the
   screenshot tells us when the workout ended — so elevated resting readings in
   that window are rated "expected recovery" instead of monitor/flag, unless
   they are high even for recovery, keep rising, or outlast the window. */
const RECOVERY_WINDOW_MIN = 120;                       // 0-2 h after exercise: elevated heart rate is expected
const LATE_RECOVERY_MIN = 180;                         // 2-3 h: mostly settled; >=110 is worth watching
const RECOVERY_LIMITS = { monitor: 120, flag_for_review: 140 };  // resting bpm inside the window
const LATE_RECOVERY_MONITOR_BPM = 110;
const RECOVERY_RISE_BPM = 10;                          // rising by this much across the window -> monitor
const MAX_ASSUMED_WORKOUT_MIN = 180;                   // only the end known: readings up to 3 h before count as "during"

function applyRecoveryContext(rows, workoutEnd, workoutStart = null) {
  const summary = { nRecovery: 0, nExercise: 0, trend: "", rising: false };
  if (!workoutEnd || !rows.length) return { rows, summary };
  const endMs = workoutEnd.getTime();
  const startMs = workoutStart ? workoutStart.getTime() : endMs - MAX_ASSUMED_WORKOUT_MIN * 60_000;
  const out = rows.map((r) => {
    if (r.activity_state !== "resting" || r.bpm == null || Number.isNaN(r.bpm) || !r.timestamp) return r;
    const t = new Date(r.timestamp).getTime();
    const mins = (t - endMs) / 60_000;
    const bpm = r.bpm;
    const elevated = (BUCKET_RANK[r.risk_bucket] || 0) >= BUCKET_RANK.monitor || bpm >= RESTING_GUARDRAILS.monitor.high;
    if (t >= startMs && t < endMs) {
      summary.nExercise += 1;
      return { ...r, risk_bucket: "during_exercise", recoveryMins: mins,
        explanation: `${Math.round(bpm)} bpm during your workout — an exercise heart rate, not a resting reading, so it is not judged against resting limits.` };
    }
    if (mins >= 0 && mins <= RECOVERY_WINDOW_MIN && elevated) {
      const when = `${Math.round(mins)} min after your workout ended`;
      if (bpm >= RECOVERY_LIMITS.flag_for_review) {
        return { ...r, risk_bucket: "flag_for_review", recoveryMins: mins,
          explanation: `${Math.round(bpm)} bpm ${when} is above ${RECOVERY_LIMITS.flag_for_review} bpm — higher than normal recovery, so it is flagged for review.` };
      }
      if (bpm >= RECOVERY_LIMITS.monitor) {
        return { ...r, risk_bucket: "monitor", recoveryMins: mins,
          explanation: `${Math.round(bpm)} bpm ${when} is on the high side even for recovery (above ${RECOVERY_LIMITS.monitor} bpm), so it is worth monitoring.` };
      }
      summary.nRecovery += 1;
      return { ...r, risk_bucket: "expected_recovery", recoveryMins: mins,
        explanation: `${Math.round(bpm)} bpm ${when}. Heart rate normally stays above resting for one to two hours after exercise, so this is expected recovery, not a resting reading.` };
    }
    if (mins > RECOVERY_WINDOW_MIN && mins <= LATE_RECOVERY_MIN) {
      const hours = (mins / 60).toFixed(1);
      if (bpm >= LATE_RECOVERY_MONITOR_BPM) {
        const bucket = (BUCKET_RANK[r.risk_bucket] || 0) < BUCKET_RANK.monitor ? "monitor" : r.risk_bucket;
        return { ...r, risk_bucket: bucket, recoveryMins: mins,
          explanation: `Still ${Math.round(bpm)} bpm ${hours} hours after your workout ended. Recovery usually settles within two hours, so this is worth monitoring.` };
      }
      if (elevated) {
        summary.nRecovery += 1;
        return { ...r, risk_bucket: "expected_recovery", recoveryMins: mins,
          explanation: `${Math.round(bpm)} bpm ${hours} hours after your workout ended — still mildly elevated, which is common after long or hot sessions. It should settle within a few hours; if it doesn't, that is worth monitoring.` };
      }
    }
    return r;
  });
  const rec = out.filter((r) => r.risk_bucket === "expected_recovery").sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  if (rec.length >= 2) {
    const first = rec[0].bpm, last = rec[rec.length - 1].bpm, max = Math.max(...rec.map((r) => r.bpm));
    if (last - first >= RECOVERY_RISE_BPM) {
      summary.rising = true;
      summary.trend = `rising from ${Math.round(first)} to ${Math.round(last)} bpm instead of settling`;
      const lastRow = rec[rec.length - 1];
      const i = out.indexOf(lastRow);
      out[i] = { ...lastRow, risk_bucket: "monitor",
        explanation: `${Math.round(last)} bpm ${Math.round(lastRow.recoveryMins)} min after your workout, and rising (from ${Math.round(first)} bpm earlier) instead of settling — worth monitoring.` };
      summary.nRecovery -= 1;
    } else if (last <= max - 3) {
      summary.trend = `settling from ${Math.round(max)} to ${Math.round(last)} bpm`;
    } else {
      summary.trend = `holding around ${Math.round(rec.reduce((a, r) => a + r.bpm, 0) / rec.length)} bpm`;
    }
  }
  return { rows: out, summary };
}

/* Symptoms escalate the verdict whatever the numbers say. */
const SYMPTOM_OPTIONS = ["Chest pain or pressure", "Fainting or nearly fainting", "Palpitations or fluttering", "Dizziness or light-headedness", "Breathlessness at rest", "Unusual fatigue"];
const SYMPTOM_RULES = {
  "Chest pain or pressure": "flag_for_review", "Fainting or nearly fainting": "flag_for_review",
  "Palpitations or fluttering": "monitor", "Dizziness or light-headedness": "monitor",
  "Breathlessness at rest": "monitor", "Unusual fatigue": "monitor",
};
const URGENT_SYMPTOMS = new Set(["Chest pain or pressure", "Fainting or nearly fainting"]);
function symptomEscalation(symptoms) {
  const chosen = (symptoms || []).filter((x) => SYMPTOM_RULES[x]);
  if (!chosen.length) return { bucket: null, text: "" };
  const bucket = chosen.map((x) => SYMPTOM_RULES[x]).sort((a, b) => BUCKET_RANK[b] - BUCKET_RANK[a])[0];
  const listed = chosen.map((x) => x.toLowerCase()).join(", ");
  const text = chosen.some((x) => URGENT_SYMPTOMS.has(x))
    ? `You reported ${listed}. With symptoms like these, get medical advice promptly (emergency services if the chest pain is ongoing) — whatever the numbers say.`
    : `You reported ${listed}. Symptoms count for more than numbers: this is rated at least ${BUCKET_LABELS[bucket]} because of them, and worth mentioning to a doctor if it keeps happening.`;
  return { bucket, text };
}
function applyRestingGuardrail(scored, baselineStd) {
  if (scored.activity_state !== "resting" || scored.bpm == null || Number.isNaN(scored.bpm)) return scored;
  let hit = null;
  for (const bucket of ["monitor", "flag_for_review"]) {   // stricter level last so it wins
    const lim = RESTING_GUARDRAILS[bucket];
    if (scored.bpm >= lim.high) hit = { bucket, note: `at or above the ${lim.high} bpm resting limit` };
    else if (scored.bpm <= lim.low) hit = { bucket, note: `at or below the ${lim.low} bpm resting limit` };
  }
  if (!hit) return scored;
  const risk_bucket = BUCKET_RANK[hit.bucket] > (BUCKET_RANK[scored.risk_bucket] || 0) ? hit.bucket : scored.risk_bucket;
  const guardTxt = `${Math.round(scored.bpm)} bpm while resting is ${hit.note} (the standard tachycardia/bradycardia cut-off), so it's rated ${BUCKET_LABELS[risk_bucket]} regardless of your baseline.`;
  const wasWithinHistory = scored.risk_bucket === "normal" || scored.risk_bucket === "insufficient_baseline";
  const explanation = wasWithinHistory
    ? `${scored.risk_bucket === "normal" ? "Inside the range of your baseline history, but" : "No baseline for this activity yet, but"} ${guardTxt.charAt(0).toLowerCase()}${guardTxt.slice(1)}`
    : `${scored.explanation} ${guardTxt}`;
  return { ...scored, risk_bucket, explanation, guardrail: hit.note };
}
function scoreReadings(baseline, readings) {
  return scoreReadingsAgainstBaseline(baseline, readings).map((r) => applyRestingGuardrail(r));
}
function scoreReadingsAgainstBaseline(baseline, readings) {
  return readings.map((row) => {
    const b = baseline[row.activity_state];
    if (!b) return { ...row, risk_bucket: "insufficient_baseline", explanation: "Still learning your pattern for this activity." };
    const trackedFeatures = WEARABLE_FIELD_DEFS.map((f) => f.key).filter((f) => b.stats[f] != null);
    const availableFeatures = trackedFeatures.filter((f) => row[f] != null && !Number.isNaN(row[f]));
    if (availableFeatures.length === 0) {
      return { ...row, risk_bucket: "insufficient_baseline", explanation: "No usable measurements in this reading to check." };
    }
    const zs = availableFeatures.map((f) => (row[f] - b.stats[f].mean) / b.stats[f].std);
    const zRms = Math.sqrt(zs.reduce((a, z) => a + z * z, 0) / zs.length);
    const bucket = bucketFor(zRms);
    const missingNote = availableFeatures.length < trackedFeatures.length
      ? ` (based on ${availableFeatures.length} of ${trackedFeatures.length} tracked measurements — the rest weren't available for this reading)`
      : "";
    let explanation;
    if (bucket === "normal") {
      explanation = `Within your typical ${row.activity_state} range${missingNote}.`;
    } else if (row.bpm != null) {
      const bpmDiff = row.bpm - b.stats.bpm.mean;
      const direction = bpmDiff >= 0 ? "higher" : "lower";
      explanation = `${Math.abs(bpmDiff).toFixed(0)} bpm ${direction} than your usual ${row.activity_state} average (${b.stats.bpm.mean.toFixed(0)} bpm)${missingNote}. Worth a look if this keeps happening.`;
    } else {
      explanation = `Outside your typical ${row.activity_state} pattern${missingNote}. Worth a look if this keeps happening.`;
    }
    return { ...row, risk_bucket: bucket, explanation };
  });
}

/* ------------------------------------------------------------------ */
/* Tier 2: ECG-style classifier                                        */
/* ------------------------------------------------------------------ */
const ECG_CENTROIDS = {
  Normal: { mean_rr_ms: 850, std_rr_ms: 35, pnn50: 18, qrs_width_ms: 90 },
  "Irregular (PVC-like)": { mean_rr_ms: 820, std_rr_ms: 90, pnn50: 12, qrs_width_ms: 140 },
  "Irregular (AFib-like)": { mean_rr_ms: 700, std_rr_ms: 140, pnn50: 35, qrs_width_ms: 95 },
  "Slow (Bradycardia-like)": { mean_rr_ms: 1200, std_rr_ms: 30, pnn50: 15, qrs_width_ms: 92 },
  // Internal only — fast regular kept distinct from AFib so AFib recall stays high; never shown to user (maps to generic Abnormal)
  "Fast (Tachycardia-like)": { mean_rr_ms: 500, std_rr_ms: 30, pnn50: 15, qrs_width_ms: 92 },
};
const ECG_SCALE = { mean_rr_ms: 110, std_rr_ms: 40, pnn50: 10, qrs_width_ms: 20 };
// Guardrails so a fast/slow/wide/prolonged rhythm is never called Normal
const ECG_GUARD = { tachyMonitor: 100, tachyFlag: 130, bradyMonitor: 60, bradyFlag: 50, qtcMonitor: 460, qtcFlag: 500, qrsFlag: 120 };
function classifyECG(features) {
  const availableKeys = Object.keys(ECG_SCALE).filter((f) => features[f] != null && !Number.isNaN(features[f]));
  if (availableKeys.length === 0) return { predicted: null, confidence: 0, insufficientData: true };
  const dists = {};
  Object.entries(ECG_CENTROIDS).forEach(([label, centroid]) => {
    let sumSq = 0;
    availableKeys.forEach((f) => { const z = (features[f] - centroid[f]) / ECG_SCALE[f]; sumSq += z * z; });
    dists[label] = Math.sqrt(sumSq);
  });
  const negDists = Object.fromEntries(Object.entries(dists).map(([k, d]) => [k, -d]));
  const maxNeg = Math.max(...Object.values(negDists));
  const expVals = Object.fromEntries(Object.entries(negDists).map(([k, v]) => [k, Math.exp(v - maxNeg)]));
  const sumExp = Object.values(expVals).reduce((a, b) => a + b, 0);
  const probs = Object.fromEntries(Object.entries(expVals).map(([k, v]) => [k, v / sumExp]));
  let predicted = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
  let confidence = probs[predicted];
  let guardrail = null;
  // Hide internal Tachycardia-like — user sees generic Abnormal (or AFib if P-wave suggests RVR)
  if (predicted === "Fast (Tachycardia-like)") {
    let hr0 = null;
    if (features.mean_rr_ms != null && Number.isFinite(features.mean_rr_ms) && features.mean_rr_ms > 0) hr0 = 60000 / features.mean_rr_ms;
    else if (features.heart_rate_bpm != null) hr0 = Number(features.heart_rate_bpm);
    const pAxis0 = features.p_axis_deg;
    const pDur0 = features.p_duration_ms;
    if (pAxis0 != null && Number(pAxis0) === 0 && pDur0 == null && hr0 != null && hr0 >= ECG_GUARD.tachyMonitor) {
      guardrail = `P wave abnormal (P axis 0°, no distinct P duration) suggests AFib with rapid ventricular response rather than sinus tachycardia.`;
      predicted = "Irregular (AFib-like)";
      confidence = Math.max(confidence, 0.88);
    } else {
      const hrStr = hr0 != null ? `${Math.round(hr0)} bpm` : "fast rate";
      guardrail = `Heart rate ${hrStr} is fast — flagged as abnormal (outside Normal / Irregular PVC-like / AFib-like / Slow patterns).`;
      predicted = "Abnormal";
      confidence = Math.max(confidence, 0.85);
    }
  }
  if (predicted === "Normal") {
    let hr = null;
    if (features.mean_rr_ms != null && Number.isFinite(features.mean_rr_ms) && features.mean_rr_ms > 0) hr = 60000 / features.mean_rr_ms;
    else if (features.heart_rate_bpm != null) hr = Number(features.heart_rate_bpm);
    const qtc = features.qtc_ms != null ? Number(features.qtc_ms) : null;
    const qrs = features.qrs_width_ms != null ? Number(features.qrs_width_ms) : null;
    if (hr != null) {
      if (hr >= ECG_GUARD.tachyFlag) { guardrail = `Heart rate ${Math.round(hr)} bpm ≥${ECG_GUARD.tachyFlag} — fast rhythm, flagged even though the beat shape looks regular.`; predicted = "Abnormal"; confidence = Math.max(confidence, 0.92); }
      else if (hr >= ECG_GUARD.tachyMonitor) { guardrail = `Heart rate ${Math.round(hr)} bpm ≥${ECG_GUARD.tachyMonitor} — fast, not Normal.`; predicted = "Abnormal"; confidence = Math.max(confidence, 0.85); }
      else if (hr <= ECG_GUARD.bradyFlag) { guardrail = `Heart rate ${Math.round(hr)} bpm ≤${ECG_GUARD.bradyFlag} — bradycardia, flagged.`; predicted = "Slow (Bradycardia-like)"; confidence = Math.max(confidence, 0.92); }
      else if (hr <= ECG_GUARD.bradyMonitor) { guardrail = `Heart rate ${Math.round(hr)} bpm ≤${ECG_GUARD.bradyMonitor} — slow, not Normal.`; predicted = "Slow (Bradycardia-like)"; confidence = Math.max(confidence, 0.85); }
    }
    if (predicted === "Normal" && qrs != null && qrs >= ECG_GUARD.qrsFlag) { guardrail = `QRS ${Math.round(qrs)} ms ≥${ECG_GUARD.qrsFlag} ms — wide, flagged.`; predicted = "Irregular (PVC-like)"; confidence = Math.max(confidence, 0.88); }
    if (predicted === "Normal" && qtc != null) {
      if (qtc >= ECG_GUARD.qtcFlag) { guardrail = `QTc ${Math.round(qtc)} ms ≥${ECG_GUARD.qtcFlag} — markedly prolonged, flagged.`; predicted = "Abnormal"; confidence = Math.max(confidence, 0.90); }
      else if (qtc >= ECG_GUARD.qtcMonitor) { guardrail = `QTc ${Math.round(qtc)} ms ≥${ECG_GUARD.qtcMonitor} — prolonged, not Normal.`; predicted = "Abnormal"; confidence = Math.max(confidence, 0.86); }
    }
  }
  // Fast generic Abnormal vs AFib with RVR: fast + absent P wave → likely AFib
  {
    let hr = null;
    if (features.mean_rr_ms != null && Number.isFinite(features.mean_rr_ms) && features.mean_rr_ms > 0) hr = 60000 / features.mean_rr_ms;
    else if (features.heart_rate_bpm != null) hr = Number(features.heart_rate_bpm);
    const pAxis = features.p_axis_deg;
    const pDur = features.p_duration_ms;
    if (predicted === "Abnormal" && hr != null && hr >= ECG_GUARD.tachyMonitor && pAxis != null && Number(pAxis) === 0 && pDur == null) {
      const extra = ` P wave abnormal (P axis 0°, no distinct P duration) suggests AFib with rapid ventricular response rather than sinus tachycardia.`;
      guardrail = (guardrail ? guardrail + extra : extra.trim());
      predicted = "Irregular (AFib-like)";
      confidence = Math.max(confidence, 0.88);
    }
  }
  return { predicted, confidence, guardrail };
}

/* ------------------------------------------------------------------ */
/* Vision AI image / PDF reading via the Gemini API                    */
/* ------------------------------------------------------------------ */
const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta";
const GEMINI_KEY_STORAGE = "pulseware.geminiKey";

// Best first. Names not in the API's live model list are skipped, and any other
// vision-capable "flash" model the API advertises is appended as a last resort —
// Google retires model names (1.5 Flash, 2.0 Flash are gone), so nothing is hard-wired.
const PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-2.5-pro"];
const EXCLUDED_MODEL_HINTS = ["tts", "image", "audio", "live", "embedding", "omni", "customtools", "preview"];
const modelCache = new Map();

function loadStoredGeminiKey() {
  try { return localStorage.getItem(GEMINI_KEY_STORAGE) || ""; } catch { return ""; }
}
function storeGeminiKey(key) {
  try { key ? localStorage.setItem(GEMINI_KEY_STORAGE, key) : localStorage.removeItem(GEMINI_KEY_STORAGE); } catch { /* private mode */ }
}
function getGeminiKey() {
  return loadStoredGeminiKey() || import.meta.env.VITE_GEMINI_API_KEY || "";
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(new Error("Could not read that file."));
    reader.readAsDataURL(file);
  });
}

async function candidateModels(apiKey) {
  if (modelCache.has(apiKey)) return modelCache.get(apiKey);
  let discovered = [];
  try {
    const res = await fetch(`${GEMINI_BASE}/models?pageSize=200&key=${encodeURIComponent(apiKey)}`);
    if (res.status === 400 || res.status === 401 || res.status === 403) {
      throw new Error("Gemini rejected that API key. Check it (keys come from https://aistudio.google.com) and paste it again.");
    }
    if (res.ok) {
      const data = await res.json();
      discovered = (data.models || [])
        .filter((m) => !m.supportedGenerationMethods || m.supportedGenerationMethods.includes("generateContent"))
        .map((m) => m.name.replace(/^models\//, ""));
    }
  } catch (err) {
    if (/rejected that API key/.test(err.message)) throw err;
    discovered = []; // offline / CORS hiccup — fall back to the static list
  }
  let candidates;
  if (discovered.length) {
    const ordered = PREFERRED_MODELS.filter((m) => discovered.includes(m));
    const extras = discovered
      .filter((n) => n.startsWith("gemini-") && n.includes("flash") && !ordered.includes(n) && !EXCLUDED_MODEL_HINTS.some((h) => n.includes(h)))
      .sort();
    candidates = [...ordered, ...extras];
  } else {
    candidates = [...PREFERRED_MODELS];
  }
  modelCache.set(apiKey, candidates);
  return candidates;
}

async function readFileWithGemini(file, promptText) {
  const apiKey = getGeminiKey();
  if (!apiKey) {
    throw new Error("No Gemini API key set. Paste one into the “Gemini API key” field below (free at https://aistudio.google.com).");
  }

  const base64Data = await fileToBase64(file);
  const mimeType = file.type || (file.name.toLowerCase().endsWith(".pdf") ? "application/pdf" : "image/jpeg");
  const requestBody = {
    contents: [{ parts: [{ text: promptText }, { inlineData: { mimeType, data: base64Data } }] }],
    generationConfig: { responseMimeType: "application/json", temperature: 0.1 },
  };

  const models = await candidateModels(apiKey);
  let lastStatus = null;
  let rawText = null;
  for (const model of models) {
    const response = await fetch(`${GEMINI_BASE}/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    if (response.ok) {
      const data = await response.json();
      rawText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      if (rawText) break;
      lastStatus = 204;
      continue;
    }
    const details = await response.text();
    console.warn(`Gemini ${model} -> ${response.status}:`, details.slice(0, 300));
    lastStatus = response.status;
    if (response.status === 400 && /api key not valid|API_KEY_INVALID/i.test(details)) {
      throw new Error("Gemini rejected that API key. Check it (keys come from https://aistudio.google.com) and paste it again.");
    }
    if ([404, 429, 500, 503].includes(response.status)) continue; // retired, per-model quota, or overloaded → next model
    throw new Error(`The Vision API request failed (${response.status}). Check your API key and connection.`);
  }

  if (!rawText) {
    if (lastStatus === 429) throw new Error("Gemini rate limit reached (the free tier allows a few requests per minute). Wait ~30 seconds and try again.");
    if (lastStatus === 503 || lastStatus === 500) throw new Error("Gemini is busy right now (every available model returned 503). Try again in a moment.");
    if (lastStatus === 404) throw new Error("None of the Gemini models are available to this key (404). Check the key's project at https://aistudio.google.com.");
    throw new Error("No readable text came back from the vision model.");
  }

  const cleaned = rawText.replace(/```json|```/g, "").trim();
  const parsed = JSON.parse(cleaned);

  // If heart_rate_bpm is present but mean_rr_ms is missing, compute it:
  if (parsed.mean_rr_ms == null && parsed.heart_rate_bpm != null && parsed.heart_rate_bpm > 0) {
    parsed.mean_rr_ms = Math.round(60000.0 / parsed.heart_rate_bpm);
  }
  return parsed;
}

function parseCsvAsync(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, { header: true, dynamicTyping: true, skipEmptyLines: true, complete: (res) => resolve(res.data), error: reject });
  });
}

const TIER1_IMAGE_PROMPT = `This is a screenshot from a fitness tracker or smartwatch app (e.g. Apple Health, Fitbit, Garmin, Oura, Whoop, Samsung) showing heart and wellness data. Read ONLY numbers directly visible as labeled values on screen — never estimate, infer, or calculate a number that isn't directly shown.

Mapping notes:
- Apple Health's single "Heart Rate Variability" metric is computed using the SDNN method — map it to hrv_sdnn, not hrv_rmssd.
- Only fill hrv_rmssd if a value is separately and explicitly labeled RMSSD somewhere on screen (common on Whoop, Oura, Garmin HRV status).
- "bpm": the primary heart-rate value. ALWAYS fill this when any heart rate is visible. Pick in this order: (1) a value labeled Latest / Current / Now; (2) otherwise the Resting heart rate (and copy it to "bpm_resting" too); (3) otherwise the most recent entry of the readings list. Never use High, Max, Min, Average, Range, or Walking Heart Rate Average as bpm. Examples: Apple Health "Latest 80 / Resting 60 / Walking Average 107" -> bpm 80, bpm_resting 60; Samsung Health "63 bpm Resting / 125 bpm High" + an alerts list -> bpm 63, bpm_resting 63, bpm_high 125, readings = the list.
- skin_temp_c means a TEMPERATURE DEVIATION from the person's own baseline (e.g. Oura "Body Temperature Deviation", Fitbit "Skin Temperature Variation") — usually a small number like +0.3 or -0.5, not a normal body temperature like 37. If only an absolute temperature is shown with no baseline/deviation framing, leave this null.
- readiness_score covers any 0-100 "how ready/recovered are you" style score — Garmin Body Battery, Oura Readiness, Whoop Recovery, Fitbit Daily Readiness all map here.
- vo2max may appear as "Cardio Fitness" (Apple) or "VO2 Max" (Garmin, Fitbit).
- "bpm_resting": the Resting heart rate if one is labeled (e.g. "63 bpm Resting", "Resting Heart Rate 60"), else null.
- "bpm_high": the day's High / Max heart rate if one is labeled (e.g. "125 bpm High", "Max 142"), else null.
- "date": the date the screen refers to, exactly as written ("Today", "Sep 7", "2026-09-07"), else null.
- "activity_periods": every period of exercise or sleep the screen shows — a workout icon on the timeline (runner, cyclist, dumbbell, walker), a labelled workout, a sleep/Zz icon, or a clearly sustained block of heart rate above about 130 bpm on the chart. One object per period: {"type": "run|workout|walk|sleep|other", "start": "<time read off the axis, e.g. 8:30 am>", "end": "<time, e.g. 10:15 am>", "evidence": "<icon | label | chart block>"}. Times may be approximate — round to the nearest 15 minutes. Use [] if none.
- "bpm_high_time": the time the day's High occurred if it can be read (a label, or the position of the chart's highest peak on the time axis, e.g. "9:30 am"), else null.
- "readings": EVERY individually listed heart-rate measurement that has its own time on screen — rows in an "Abnormal heart rate alerts" list, a history list, notification rows, etc. One object per row, in the order shown: {"time": "<exactly as written, e.g. 1:16 PM>", "bpm": <number>, "label": "<name of the list or row, e.g. Abnormal alert>"}. Use [] if there is no such list. Do NOT read values off a chart line, and do not repeat the Resting/High summary numbers here.

Respond with ONLY a JSON object (no other text, no markdown) in exactly this format:
{"bpm": <number or null>, "hrv_rmssd": <number or null>, "hrv_sdnn": <number or null>, "spo2": <number or null>, "resp_rate": <number or null>, "skin_temp_c": <number or null>, "vo2max": <number or null>, "readiness_score": <number or null>, "activity_state": "resting" or "exercise" or null, "bpm_resting": <number or null>, "bpm_high": <number or null>, "bpm_high_time": <string or null>, "date": <string or null>, "activity_periods": [{"type": <string>, "start": <string or null>, "end": <string or null>, "evidence": <string>}], "readings": [{"time": <string>, "bpm": <number>, "label": <string>}]}
Use null for any field with no directly visible, labeled value. Only set activity_state if there's a clear workout indicator on screen; otherwise null.`;

/* --- Everything else on a screenshot: alert lists, history rows, the day High.
   A summary screen (e.g. Samsung Health "Today") carries more than the one
   number that fills the form. These helpers keep the rest so it can be
   checked as a batch instead of being dropped. --- */
function normalizeLabel(v) {
  const label = (typeof v === "string" && v.trim()) || "Listed reading";
  // Screen titles like "ABNORMAL HEART RATE ALERTS" -> sentence case
  return label === label.toUpperCase() && /[A-Z]/.test(label) ? label.charAt(0) + label.slice(1).toLowerCase() : label;
}
function sanitizeScreenshotExtras(extracted) {
  const { min: lo, max: hi } = PLAUSIBLE_RANGES.bpm;
  const num = (v) => { const n = typeof v === "number" ? v : parseFloat(v); return Number.isFinite(n) && n >= lo && n <= hi ? n : null; };
  const bpmResting = num(extracted.bpm_resting);
  const bpmHigh = num(extracted.bpm_high);
  const date = typeof extracted.date === "string" && extracted.date.trim() ? extracted.date.trim() : null;
  const readings = (Array.isArray(extracted.readings) ? extracted.readings : [])
    .filter((r) => r && typeof r === "object")
    .map((r) => ({ time: typeof r.time === "string" && r.time.trim() ? r.time.trim() : null, bpm: num(r.bpm), label: normalizeLabel(r.label) }))
    .filter((r) => r.bpm != null && !(r.time == null && (r.bpm === bpmResting || r.bpm === bpmHigh)));
  const bpmHighTime = typeof extracted.bpm_high_time === "string" && extracted.bpm_high_time.trim() ? extracted.bpm_high_time.trim() : null;
  const activityPeriods = (Array.isArray(extracted.activity_periods) ? extracted.activity_periods : [])
    .filter((p) => p && typeof p === "object")
    .map((p) => ({
      type: (typeof p.type === "string" && p.type.trim().toLowerCase()) || "other",
      start: typeof p.start === "string" && p.start.trim() ? p.start.trim() : null,
      end: typeof p.end === "string" && p.end.trim() ? p.end.trim() : null,
    }))
    .filter((p) => p.start || p.end);
  if (readings.length === 0 && bpmHigh == null && activityPeriods.length === 0) return null;
  return { readings, bpmHigh, bpmHighTime, bpmResting, date, activityPeriods };
}
const EXERCISE_TYPES = new Set(["run", "workout", "walk", "cycle", "ride", "swim", "hike", "exercise", "other"]);
/* The most recent exercise period a screenshot shows (sleep is ignored). Times
   are approximate — they are read off a chart axis — so the user confirms them. */
function detectedWorkout(extras) {
  if (!extras) return null;
  let best = null;
  for (const p of extras.activityPeriods || []) {
    if (!EXERCISE_TYPES.has(p.type) || !(p.end || p.start)) continue;
    const endStr = p.end || p.start;
    const end = parseScreenshotTime(endStr, extras.date);
    const start = p.start ? parseScreenshotTime(p.start, extras.date) : null;
    if (!best || end > best.end) {
      const span = p.start && p.end ? `from about ${p.start} to ${p.end}` : `around ${endStr}`;
      best = { type: p.type, start, end, label: `${p.type} ${span}` };
    }
  }
  return best;
}
function screenshotBaseDate(dateStr) {
  const today = new Date(); today.setHours(0, 0, 0, 0);
  if (!dateStr) return today;
  const low = dateStr.toLowerCase();
  if (low === "today" || low === "now") return today;
  if (low === "yesterday") { const d = new Date(today); d.setDate(d.getDate() - 1); return d; }
  for (const candidate of [dateStr, `${dateStr} ${today.getFullYear()}`]) {
    const d = new Date(candidate);
    if (!Number.isNaN(d.getTime())) { d.setHours(0, 0, 0, 0); if (d.getFullYear() < 2000) d.setFullYear(today.getFullYear()); return d; }
  }
  return today;
}
function parseScreenshotTime(timeStr, dateStr) {
  const base = screenshotBaseDate(dateStr);
  const m = timeStr && timeStr.match(/(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*([AaPp]\.?[Mm]\.?)?/);
  if (!m) return new Date();
  let hour = parseInt(m[1], 10); const minute = parseInt(m[2] || "0", 10); const second = parseInt(m[3] || "0", 10);
  const ampm = (m[4] || "").replace(/\./g, "").toLowerCase();
  if (ampm === "pm" && hour < 12) hour += 12;
  if (ampm === "am" && hour === 12) hour = 0;
  if (hour > 23 || minute > 59) return new Date();
  const d = new Date(base); d.setHours(hour, minute, second, 0); return d;
}
function screenshotReadingsToEntries(extras, includeHigh, workoutEnd = null, workoutStart = null) {
  const entries = extras.readings.map((r) => ({
    bpm: r.bpm, hrv_rmssd: null, hrv_sdnn: null, activity_state: "resting",
    timestamp: parseScreenshotTime(r.time, extras.date).toISOString(),
    source: `${r.label}${r.time ? ` ${r.time}` : ""}`,
  }));
  if (includeHigh && extras.bpmHigh != null) {
    const last = entries.length ? new Date(entries[entries.length - 1].timestamp) : null;
    let ts;
    let source = "Day high (activity unknown)";
    if (extras.bpmHighTime) ts = parseScreenshotTime(extras.bpmHighTime, extras.date);
    if (workoutEnd) {
      // You exercised, so the day's peak belongs to the workout - even when the
      // chart time was read a little off (or not at all) and misses the window.
      const startMs = workoutStart ? workoutStart.getTime() : workoutEnd.getTime() - MAX_ASSUMED_WORKOUT_MIN * 60_000;
      if (!ts || ts.getTime() < startMs || ts.getTime() >= workoutEnd.getTime()) ts = new Date(workoutEnd.getTime() - 60_000);
      source = "Day high (during your workout)";
    } else if (!ts) {
      if (last) ts = new Date(last.getTime() + 60_000);
      else { ts = screenshotBaseDate(extras.date); ts.setHours(23, 59, 0, 0); }
    }
    entries.push({ bpm: extras.bpmHigh, hrv_rmssd: null, hrv_sdnn: null, activity_state: "resting", timestamp: ts.toISOString(), source });
  }
  return entries.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
}

const TIER2_FILE_PROMPT = `This file is an ECG/EKG or heart rhythm report (e.g., 12-lead printout, rhythm strip, CineECG report).
Read and extract the numerical measurements labeled on the page.

Important guidelines:
- "mean_rr_ms": Mean RR or NN interval in ms (e.g., if 'RR Interval: 843 ± 9 ms' or 'NN Interval: 843 ms', value is 843).
- "std_rr_ms": RR variation or SD in ms (e.g., if '843 ± 9 ms', value is 9).
- "pnn50": Percentage of successive RR differences > 50ms (0-100%). Return null if not explicitly reported.
- "qrs_width_ms": QRS duration or width in ms (e.g., 'QRS duration: 98 ms' -> 98).
- "heart_rate_bpm": Heart rate or ventricular rate in BPM (e.g., '71 BPM' -> 71).
- "pr_interval_ms": PR interval in ms (e.g., '166 ± 1 ms' -> 166).
- "p_duration_ms": P wave duration in ms (e.g., '112 ms' -> 112).
- "qt_ms": QT interval in ms (e.g., '396 ms' -> 396).
- "qtc_ms": QTc interval in ms (e.g., '431 ms' -> 431).
- "p_axis_deg", "qrs_axis_deg", "t_axis_deg": Electrical axes in degrees if present.
- "rv5_mv", "sv1_mv": Voltages in mV if present.

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
}`;

/* ------------------------------------------------------------------ */
/* Copy                                                                */
/* ------------------------------------------------------------------ */
const TIER1_GLOSSARY = [
  { term: "BPM", plain: "Beats per minute — how fast your heart is beating right now." },
  { term: "HRV (RMSSD)", plain: "How much the gap between your heartbeats changes moment to moment. Generally, higher variation suggests your body is relaxed and well-recovered." },
  { term: "HRV (SDNN)", plain: "A wider-lens version of the same idea — how much your heartbeat timing varies across the whole reading." },
  { term: "SpO2", plain: "The percentage of oxygen your blood is carrying. Healthy readings are typically 95% and above." },
  { term: "Respiratory rate", plain: "How many breaths you take per minute." },
  { term: "Skin temp (\u0394\u00B0C)", plain: "How much your skin temperature has shifted from your own personal baseline — not absolute temperature." },
  { term: "VO2 max", plain: "A general cardio fitness measure — roughly, how efficiently your body uses oxygen during exercise." },
  { term: "Readiness / recovery score", plain: "A single 0\u2013100 summary score (Garmin Body Battery, Oura Readiness, Whoop Recovery, Fitbit Daily Readiness)." },
  { term: "Normal / Monitor / Flag for review", plain: "How this reading compares to your own usual pattern." },
];
const TIER2_GLOSSARY = [
  { term: "Mean RR / NN interval", plain: "The average time gap between your heartbeats, in milliseconds. A smaller number means a faster heart rate." },
  { term: "RR variation", plain: "How much that gap between beats changes from one beat to the next across the reading." },
  { term: "pNN50", plain: "The percentage of heartbeats that differ by more than 50 milliseconds from the prior beat — an indicator of autonomic adaptability." },
  { term: "QRS duration / width", plain: "How long each individual heartbeat's ventricular depolarization takes. Normal is typically 80–120 ms." },
  { term: "Heart rate", plain: "Your pulse in beats per minute (often 60,000 / Mean RR)." },
  { term: "PR interval", plain: "The time from atrial to ventricular excitation (typically 120–200 ms)." },
  { term: "P duration", plain: "Duration of atrial depolarization." },
  { term: "QT / QTc", plain: "Total ventricular electrical activity time. QTc corrects for heart rate." },
  { term: "P / QRS / T axis", plain: "The general electrical direction through the heart in degrees." },
  { term: "RV5 / SV1", plain: "Voltage measurements helping evaluate ventricular hypertrophy." },
];
const CLASSIFICATION_MEANINGS = {
  Normal: "Your rhythm looks steady and falls within a typical pattern.",
  "Irregular (PVC-like)": "This looks like occasional early extra beats. These are common and often harmless on their own, but worth mentioning to a doctor if persistent.",
  "Irregular (AFib-like)": "This pattern looks chaotic in a way that resembles atrial fibrillation. Worth discussing with a doctor.",
  "Slow (Bradycardia-like)": "Your heart rate looks unusually slow and steady (< 60 BPM). Often normal in fit individuals, but consult a doctor if experiencing fatigue or dizziness.",
  Abnormal: "This rhythm is abnormal — fast rate and/or prolonged QTc outside the Normal / PVC-like / AFib-like / Bradycardia-like patterns. Worth discussing with a doctor.",
};
const BUCKET_LABELS = {
  normal: "Normal",
  expected_recovery: "Expected recovery",
  during_exercise: "During exercise",
  monitor: "Monitor",
  flag_for_review: "Flag for review",
  insufficient_baseline: "Still learning",
};

/* ------------------------------------------------------------------ */
/* Formatting                                                          */
/* ------------------------------------------------------------------ */
const HOUR = 3600e3;
const DAY = 86400e3;
const fmt = (v) => (v == null || Number.isNaN(v) ? "—" : Math.round(v));
const fmtTime = (ms) => new Date(ms).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const fmtDate = (ms) => new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const fmtDateYear = (ms) => new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
const todayEyebrow = () => new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });

// Health-style range bars: one bar per hour (D) or per day (W / M), min–max of the readings inside it.
function bucketHeartRate(history, range) {
  const empty = { bars: [], lo: null, hi: null, caption: "", domain: [40, 160], tickEvery: 0 };
  if (!history.length) return empty;

  let end = -Infinity;
  const pts = history.map((r) => {
    const t = new Date(r.timestamp).getTime();
    if (t > end) end = t;
    return { t, bpm: r.bpm };
  });
  if (!Number.isFinite(end)) return empty;

  const isDay = range === "D";
  const bucket = isDay ? HOUR : DAY;
  const span = range === "D" ? DAY : range === "W" ? 7 * DAY : 30 * DAY;
  const start = new Date(end - span);
  if (isDay) start.setMinutes(0, 0, 0); else start.setHours(0, 0, 0, 0);
  const startMs = start.getTime();
  const n = Math.floor((end - startMs) / bucket) + 1;

  const bars = Array.from({ length: n }, (_, i) => ({ t: startMs + i * bucket, lo: Infinity, hi: -Infinity, count: 0 }));
  let lo = Infinity, hi = -Infinity;
  pts.forEach(({ t, bpm }) => {
    if (t < startMs || bpm == null || Number.isNaN(bpm)) return;
    const i = Math.floor((t - startMs) / bucket);
    if (i < 0 || i >= n) return;
    const b = bars[i];
    if (bpm < b.lo) b.lo = bpm;
    if (bpm > b.hi) b.hi = bpm;
    b.count += 1;
    if (bpm < lo) lo = bpm;
    if (bpm > hi) hi = bpm;
  });

  const labelOf = (t) => {
    if (isDay) return new Date(t).toLocaleTimeString([], { hour: "numeric" });
    if (range === "W") return new Date(t).toLocaleDateString(undefined, { weekday: "short" });
    return String(new Date(t).getDate());
  };
  const whenOf = (t) => (isDay ? `${fmtDate(t)}, ${new Date(t).toLocaleTimeString([], { hour: "numeric" })}` : fmtDateYear(t));

  const out = bars.map((b) => ({
    label: labelOf(b.t),
    when: whenOf(b.t),
    count: b.count,
    range: b.count ? [Math.round(b.lo), Math.max(Math.round(b.hi), Math.round(b.lo) + 1)] : null,
  }));
  const has = lo !== Infinity;
  // Round the axis out to a clean step so the gridlines land on round numbers.
  const step = has && hi - lo > 90 ? 40 : 20;
  const domain = has
    ? [Math.max(0, Math.floor((lo - 5) / step) * step), Math.ceil((hi + 5) / step) * step]
    : [40, 160];
  const ticks = [];
  for (let v = domain[0]; v <= domain[1]; v += step) ticks.push(v);
  return {
    bars: out,
    lo: has ? Math.round(lo) : null,
    hi: has ? Math.round(hi) : null,
    caption: isDay ? fmtDateYear(end) : `${fmtDate(startMs)} – ${fmtDateYear(end)}`,
    domain,
    ticks,
    tickEvery: isDay ? 5 : range === "W" ? 0 : 4,
  };
}

/* ------------------------------------------------------------------ */
/* Small building blocks                                               */
/* ------------------------------------------------------------------ */
function Section({ title, action, footer, footerTone, children }) {
  return (
    <section className="section">
      {title && (
        <div className="section-h">
          <h2>{title}</h2>
          {action}
        </div>
      )}
      {children}
      {footer && <div className={`section-f${footerTone ? ` is-${footerTone}` : ""}`}>{footer}</div>}
    </section>
  );
}

function Segmented({ value, options, onChange, small }) {
  return (
    <div className={`seg${small ? " is-small" : ""}`}>
      {options.map(([v, label]) => (
        <button type="button" key={v} className={value === v ? "is-on" : ""} onClick={() => onChange(v)}>{label}</button>
      ))}
    </div>
  );
}

function NumberRow({ label, unit, value, placeholder, onChange }) {
  return (
    <div className="row">
      <span className="field-label">{label}</span>
      <input
        type="number" inputMode="decimal" value={value ?? ""} placeholder={placeholder || "—"}
        onChange={(e) => { const v = e.target.value; onChange(v === "" ? null : parseFloat(v)); }}
      />
      {unit ? <span className="unit">{unit}</span> : <span className="unit" />}
    </div>
  );
}

function FileRow({ label, sub, icon: Icon, accept, onFile, loading }) {
  return (
    <label className={`row is-action${loading ? " is-disabled" : ""}`}>
      <span className="ico">{loading ? <Loader2 size={18} className="spin" /> : <Icon size={18} />}</span>
      <span className="row-main">
        <span className="row-title">{loading ? "Reading with Vision AI…" : label}</span>
        {sub && <span className="row-sub">{sub}</span>}
      </span>
      <input type="file" accept={accept} onChange={onFile} disabled={loading} hidden />
    </label>
  );
}

function ApiKeyRow({ value, onChange }) {
  const fromEnv = !loadStoredGeminiKey() && !!import.meta.env.VITE_GEMINI_API_KEY;
  return (
    <div className="row">
      <span className="field-label">Gemini API key</span>
      <input
        type="password" autoComplete="off" spellCheck={false}
        value={value} placeholder={fromEnv ? "Using key from .env" : "Not set"}
        onChange={(e) => onChange(e.target.value.trim())}
        style={{ textAlign: "right" }}
      />
      <span className="unit" style={{ minWidth: 0, color: value || fromEnv ? "var(--green)" : "var(--label-3)" }}>
        {value || fromEnv ? <Check size={16} /> : null}
      </span>
    </div>
  );
}

function DisclosureRow({ open, onToggle, label, sub }) {
  return (
    <button type="button" className="row is-action" style={{ width: "100%", background: "none", border: "none", textAlign: "left" }} onClick={onToggle}>
      <span className="row-main">
        <span className="row-title">{label}</span>
        {sub && <span className="row-sub">{sub}</span>}
      </span>
      <ChevronDown size={16} className={`chev${open ? " is-open" : ""}`} />
    </button>
  );
}

function Glossary({ items }) {
  const [open, setOpen] = useState(null);
  return (
    <div className="group">
      {items.map((it) => {
        const isOpen = open === it.term;
        return (
          <div key={it.term} className="cell">
            <button type="button" className="cell-head" onClick={() => setOpen(isOpen ? null : it.term)}>
              <span className="row-title">{it.term}</span>
              <ChevronDown size={16} className={`chev${isOpen ? " is-open" : ""}`} />
            </button>
            {isOpen && <div className="row-sub">{it.plain}</div>}
          </div>
        );
      })}
    </div>
  );
}

function Verdict({ result }) {
  if (!result) return null;
  if (result.insufficientData) {
    return <div className="result is-muted">Enter or upload at least one value to check a reading.</div>;
  }
  const normal = result.predicted === "Normal";
  return (
    <div className="result">
      <div className={`result-title ${normal ? "is-normal" : "is-abnormal"}`}>{result.predicted}</div>
      <div className="result-sub">{(result.confidence * 100).toFixed(0)}% confidence</div>
      <div className="result-body">{CLASSIFICATION_MEANINGS[result.predicted]}</div>
      {result.guardrail && <div className="result-body" style={{ marginTop: 8, color: "var(--label-2)", fontSize: 13 }}>{result.guardrail}</div>}
    </div>
  );
}

function RangeTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  if (!p.range) return null;
  return (
    <div className="tip">
      <div className="tip-label">Range</div>
      <div className="tip-value">{p.range[0]}–{p.range[1]}<small>BPM</small></div>
      <div className="tip-when">{p.when}{p.count > 1 ? ` · ${p.count} readings` : ""}</div>
    </div>
  );
}

function HeartRateCard({ history }) {
  const [range, setRange] = useState("W");
  const view = useMemo(() => bucketHeartRate(history, range), [history, range]);
  const latest = useMemo(() => {
    let best = null;
    history.forEach((r) => {
      const t = new Date(r.timestamp).getTime();
      if (r.bpm != null && (!best || t > best.t)) best = { t, bpm: r.bpm };
    });
    return best;
  }, [history]);

  return (
    <div className="hero">
      <div className="hero-top">
        <div className="cat"><Heart size={16} fill="currentColor" strokeWidth={0} /> Heart Rate</div>
        <Segmented small value={range} onChange={setRange} options={[["D", "D"], ["W", "W"], ["M", "M"]]} />
      </div>
      <div className="hero-metrics">
        <div>
          <div className="metric-label">Range</div>
          <div className="metric-value">{view.lo != null ? `${view.lo}–${view.hi}` : "—"}<small>BPM</small></div>
          <div className="metric-when">{view.caption || "No readings in this window"}</div>
        </div>
        <div>
          <div className="metric-label">Latest</div>
          <div className="metric-value">{fmt(latest?.bpm)}<small>BPM</small></div>
          <div className="metric-when">{latest ? fmtTime(latest.t) : "—"}</div>
        </div>
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={view.bars} margin={{ top: 8, right: 0, left: 0, bottom: 0 }} barCategoryGap="28%">
            <CartesianGrid vertical={false} stroke="#E5E5EA" />
            <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: "#8E8E93" }} interval={view.tickEvery} />
            <YAxis orientation="right" domain={view.domain} ticks={view.ticks} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: "#8E8E93" }} width={34} />
            <Tooltip content={<RangeTip />} cursor={{ fill: "rgba(0,0,0,0.04)" }} />
            <Bar dataKey="range" fill="#FF2D55" radius={3} maxBarSize={range === "W" ? 26 : 14} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Auth — Login / Signup / Signed-out (Apple Health / Apple-ID style) */
/* Three pages share one shell: heart icon, grouped form, blue button. */
/* ------------------------------------------------------------------ */
function AuthShell({ title, lede, children, footer }) {
  return (
    <div className="auth">
      <div className="auth-card">
        <div className="auth-icon"><Heart size={40} fill="currentColor" strokeWidth={0} /></div>
        <h1>{title}</h1>
        {lede && <p className="lede">{lede}</p>}
        {children}
        {footer}
      </div>
    </div>
  );
}

function LoginPage({ onAuthed, onGoSignup }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState({});
  const [showHint, setShowHint] = useState(false);

  const validate = () => {
    const e = {};
    if (!/^\S+@\S+\.\S+$/.test(email)) e.email = "Enter a valid email address.";
    if (password.length < 6) e.password = "Use at least 6 characters.";
    setErrors(e);
    return Object.keys(e).length === 0;
  };
  const submit = (ev) => {
    ev.preventDefault();
    if (!validate()) return;
    onAuthed({ name: email.split("@")[0], email });
  };
  const errs = Object.values(errors);
  return (
    <AuthShell title="Sign in to Pulseware" lede="See how your heart rhythm has been trending." footer={<div className="auth-foot"><Lock size={12} /> Your account details stay in this browser session. Demo: jamie@example.com / secret123</div>}>
      <form onSubmit={submit} noValidate>
        <div className="group">
          <div className="row">
            <span className="field-label">Email</span>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" autoComplete="email" />
          </div>
          <div className="row">
            <span className="field-label">Password</span>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 6 characters" autoComplete="current-password" />
          </div>
        </div>
        {errs.length > 0 && <div className="errors">{errs.map((e) => <div key={e}>{e}</div>)}</div>}
        <button type="submit" className="btn">Sign In</button>
        <button type="button" className="link" style={{ display: "block", width: "100%", textAlign: "center", marginTop: 12, fontSize: 13 }} onClick={() => setShowHint((v) => !v)}>Forgot password?</button>
        {showHint && <div className="auth-hint">This is a demo — any valid email + 6-char password will sign you in. Use jamie@example.com / secret123 for the tests.</div>}
      </form>
      <div className="auth-switch">Don’t have an account? <button type="button" className="link" onClick={onGoSignup}>Create one</button></div>
    </AuthShell>
  );
}

function SignupPage({ onAuthed, onGoLogin }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [errors, setErrors] = useState({});

  const validate = () => {
    const e = {};
    if (name.trim().length < 2) e.name = "Tell us what to call you.";
    if (!/^\S+@\S+\.\S+$/.test(email)) e.email = "Enter a valid email address.";
    if (password.length < 6) e.password = "Use at least 6 characters.";
    if (confirm !== password) e.confirm = "Passwords don’t match.";
    setErrors(e);
    return Object.keys(e).length === 0;
  };
  const submit = (ev) => {
    ev.preventDefault();
    if (!validate()) return;
    onAuthed({ name: name.trim(), email });
  };
  const errs = Object.values(errors);
  return (
    <AuthShell title="Create your account" lede="Set up an account to get started — your data stays in this browser." footer={<div className="auth-foot">By creating an account you agree to use Pulseware as a decision-support tool, not a medical device.</div>}>
      <form onSubmit={submit} noValidate>
        <div className="group">
          <div className="row">
            <span className="field-label">Name</span>
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Jamie Rivera" autoComplete="name" />
          </div>
          <div className="row">
            <span className="field-label">Email</span>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" autoComplete="email" />
          </div>
          <div className="row">
            <span className="field-label">Password</span>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 6 characters" autoComplete="new-password" />
          </div>
          <div className="row">
            <span className="field-label">Confirm</span>
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="Type it again" autoComplete="new-password" />
          </div>
        </div>
        {errs.length > 0 && <div className="errors">{errs.map((e) => <div key={e}>{e}</div>)}</div>}
        <button type="submit" className="btn">Create Account</button>
      </form>
      <div className="auth-switch">Already have an account? <button type="button" className="link" onClick={onGoLogin}>Sign in</button></div>
    </AuthShell>
  );
}

function SignedOutPage({ onSignIn }) {
  return (
    <AuthShell
      title="You’re signed out"
      lede="Thanks for using Pulseware. Your session was cleared — sign in again when you’re ready."
      footer={<div className="auth-foot"><Lock size={12} /> Signed out safely. No data was sent to a server.</div>}
    >
      <div className="group">
        <div className="row is-action" style={{ justifyContent: "center", color: "var(--label-2)", cursor: "default" }}>
          <span className="row-sub" style={{ textAlign: "center", width: "100%" }}>Tip: use the demo account jamie@example.com / secret123 to explore Monitoring and Rhythm Check.</span>
        </div>
      </div>
      <button type="button" className="btn" onClick={onSignIn}>Sign in again</button>
    </AuthShell>
  );
}

function SignIn({ onAuthed }) {
  const [page, setPage] = useState("login"); // login | signup
  if (page === "signup") return <SignupPage onAuthed={onAuthed} onGoLogin={() => setPage("login")} />;
  return <LoginPage onAuthed={onAuthed} onGoSignup={() => setPage("signup")} />;
}

/* ------------------------------------------------------------------ */
/* About                                                               */
/* ------------------------------------------------------------------ */
function AboutPage() {
  return (
    <div className="section">
      <div className="prose">
        <h2>Getting to know your heart, day by day</h2>

        <h3>What Pulseware does</h3>
        <p>
          Your smartwatch already measures your heart rate all day long. Pulseware turns that
          stream of numbers into something you can actually use: a sense of what's normal for
          <em>you</em>, and a gentle nudge when something looks different from your usual
          pattern. It is a decision-support prototype, not a medical device.
        </p>

        <h3>How everyday monitoring works</h3>
        <p>
          Instead of comparing you to a generic chart, Pulseware learns your own resting and
          active patterns from your history and checks new readings against that personal
          baseline. Each reading is scored as <b>Normal</b>, <b>Monitor</b>, or <b>Flag for
          review</b> with a short explanation. Context matters: the two hours after exercise are
          rated <b>Expected recovery</b>, and readings taken during a workout are marked{" "}
          <b>During exercise</b> — so normal recovery is not flagged. Any symptoms you report
          escalate the result.
        </p>

        <h3>How the ECG snapshot check works</h3>
        <p>
          Rhythm Check sorts single-lead ECG snapshots into <b>Normal, PVC-like, AFib-like,
          Bradycardia-like</b>, or generic <b>Abnormal</b> using four trained features: mean
          RR, RR variation, pNN50, and QRS width. Safety guardrails also look at heart rate,
          QTc, QRS width, and P-wave — e.g., a fast sinus tachycardia with prolonged QTc is
          flagged as Abnormal, and a fast rhythm with an absent P-wave is reinterpreted as AFib
          with rapid response. The model is trained on the MIT-BIH Arrhythmia Database when
          available, otherwise on synthetic placeholder data.
        </p>

        <h3>Reading screenshots and PDFs with Vision AI</h3>
        <p>
          Pulseware uses Google Gemini Vision AI to read smartwatch screenshots, ECG reports, and
          PDFs. It extracts BPM, HRV, RR intervals, QRS, QTc, P-wave axis/duration, and also looks
          for extra readings on the screen (abnormal alerts, day high) and workout context. Your
          Gemini API key stays in this browser only and is never sent to a server.
        </p>

        <h3>Data and privacy</h3>
        <p>
          • <b>Local only:</b> Your CSVs, screenshots, and baseline live in this browser session
          — nothing is uploaded to a cloud database.
          <br />• <b>Real data when installed:</b> Place WESAD/PPG-DaLiA features at{" "}
          <code>ppg_wearable_features.csv</code> and MIT-BIH features at{" "}
          <code>mitbih_features.csv</code> (or <code>data_prep/</code>) to train on real data;
          otherwise the app runs on synthetic demo data with a clear warning.
          <br />• <b>Not a diagnostic device:</b> A research prototype to help you notice patterns
          and talk to your doctor — it does not replace professional medical advice.
        </p>

        <p className="dim" style={{ marginTop: 16, fontSize: 13 }}>
          Pulseware · Apple Health-inspired UI · Heart Rhythm &amp; Health Prediction · v0.2
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Dashboard                                                           */
/* ------------------------------------------------------------------ */
const NAV = [
  { id: "tier1", label: "Monitoring", Icon: Heart, tint: "#FF2D55", filled: true },
  { id: "history", label: "History", Icon: Clock, tint: "#8E8E93" },
  { id: "tier2", label: "Rhythm Check", Icon: Activity, tint: "#FF3B30" },
  { id: "about", label: "About", Icon: Info, tint: "#007AFF" },
];

function Dashboard({ user, onLogout }) {
  const [tab, setTab] = useState("tier1");
  const [mode, setMode] = useState("dummy");
  const [uploadError, setUploadError] = useState("");
  const [geminiKey, setGeminiKeyState] = useState(loadStoredGeminiKey);
  const setGeminiKey = (k) => { storeGeminiKey(k); setGeminiKeyState(k); };

  const buildInitialSample = () => {
    const baseHistory = generateDummyHistory();
    const incoming = generateDummyNewReadings();
    const baselineNow = buildBaseline(baseHistory);
    const scoredIncoming = scoreReadings(baselineNow, incoming);
    const mergedHistory = [...baseHistory, ...incoming].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    return { history: mergedHistory, checkLog: scoredIncoming };
  };

  const [history, setHistory] = useState(() => buildInitialSample().history);
  const [checkLog, setCheckLog] = useState(() => buildInitialSample().checkLog);

  const baseline = useMemo(() => buildBaseline(history), [history]);

  const counts = useMemo(() => {
    const c = { normal: 0, monitor: 0, flag_for_review: 0, insufficient_baseline: 0 };
    checkLog.forEach((r) => { c[r.risk_bucket] = (c[r.risk_bucket] || 0) + 1; });
    return c;
  }, [checkLog]);

  const visibleOptionalCols = useMemo(
    () => WEARABLE_FIELD_DEFS.filter((f) => WEARABLE_OPTIONAL_KEYS.includes(f.key) && checkLog.some((r) => r[f.key] != null)),
    [checkLog]
  );

  const parseCsv = useCallback((file, onDone) => {
    Papa.parse(file, { header: true, dynamicTyping: true, skipEmptyLines: true,
      complete: (res) => onDone(res.data),
      error: () => setUploadError("Couldn't read that file. Check the column names match the expected format."),
    });
  }, []);

  const normalizeWearableRow = (row) => {
    const matched = normalizeRowByAliases(row, WEARABLE_FIELD_DEFS);
    const lookup = {};
    Object.keys(row).forEach((k) => { lookup[normalizeKey(k)] = row[k]; });
    const activity = lookup[normalizeKey("activity_state")] ?? lookup[normalizeKey("activity")] ?? lookup[normalizeKey("state")];
    const timestamp = lookup[normalizeKey("timestamp")] ?? lookup[normalizeKey("time")] ?? lookup[normalizeKey("date")] ?? lookup[normalizeKey("datetime")];
    return { ...matched, activity_state: activity, timestamp };
  };

  const handleHistoryUpload = (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    setUploadError("");
    setMode("upload");
    parseCsv(file, (rows) => {
      const mapped = rows.map(normalizeWearableRow);
      const withBasicFields = mapped.filter((r) => r.bpm != null && r.activity_state);
      const clean = withBasicFields.filter((r) => implausibleFields(r).length === 0);
      const droppedCount = withBasicFields.length - clean.length;
      if (clean.length === 0) { setUploadError("No valid, physiologically plausible rows found in that file."); return; }
      setHistory(clean);
      setCheckLog([]);
      setUploadError(droppedCount > 0 ? `Loaded ${clean.length} rows; skipped ${droppedCount} with implausible values.` : "");
    });
    e.target.value = "";
  };

  const handleNewUpload = (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    setUploadError("");
    parseCsv(file, (rows) => {
      const mapped = rows.map(normalizeWearableRow);
      const withBasicFields = mapped.filter((r) => r.bpm != null && r.activity_state);
      const clean = withBasicFields.filter((r) => implausibleFields(r).length === 0);
      const droppedCount = withBasicFields.length - clean.length;
      if (clean.length === 0) { setUploadError("No valid rows found in that file."); return; }
      const withTimestamps = clean.map((r) => ({ ...r, timestamp: r.timestamp || new Date().toISOString() }));
      const scoredBatch = scoreReadings(baseline, withTimestamps);
      setCheckLog((prev) => [...prev, ...scoredBatch]);
      setHistory((prev) => [...prev, ...withTimestamps]);
      setMode("upload");
      setUploadError(droppedCount > 0 ? `Added ${clean.length} rows; skipped ${droppedCount} with implausible values.` : "");
    });
    e.target.value = "";
  };

  const useSampleData = () => {
    const s = buildInitialSample();
    setMode("dummy"); setHistory(s.history); setCheckLog(s.checkLog); setUploadError("");
  };

  /* --- manual reading (Tier 1) --- */
  // Activity has no default: it is filled from a screenshot when the screen shows it, otherwise the user picks it.
  const emptyManualReading = () => ({ bpm: null, hrv_rmssd: null, hrv_sdnn: null, spo2: null, resp_rate: null, skin_temp_c: null, vo2max: null, readiness_score: null, activity_state: null });
  const [manualReading, setManualReading] = useState(emptyManualReading);
  const [manualError, setManualError] = useState("");
  const [tier1ImgLoading, setTier1ImgLoading] = useState(false);
  const [tier1ImgError, setTier1ImgError] = useState("");
  const [tier1ImgNote, setTier1ImgNote] = useState("");
  const [justAdded, setJustAdded] = useState(false);
  const [showMoreWearableFields, setShowMoreWearableFields] = useState(false);
  const [screenshotExtras, setScreenshotExtras] = useState(null);
  const [includeDayHigh, setIncludeDayHigh] = useState(true);
  const [screenshotBatchNote, setScreenshotBatchNote] = useState("");
  // Context for the recovery check: a watch's alert rule has no idea a run just ended; we ask.
  const [exercised, setExercised] = useState("unsure");          // "no" | "yes" | "unsure"
  const [workoutEnd, setWorkoutEnd] = useState(() => { const d = new Date(Date.now() - 3600_000); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; });
  const [workoutDetected, setWorkoutDetected] = useState(null);
  const [symptoms, setSymptoms] = useState([]);
  const [showSymptoms, setShowSymptoms] = useState(false);

  const highAboveAlerts = !!(screenshotExtras && screenshotExtras.bpmHigh != null && screenshotExtras.readings.length
    && screenshotExtras.bpmHigh > Math.max(...screenshotExtras.readings.map((r) => r.bpm)));
  const workoutEndDate = (() => {
    if (exercised !== "yes" || !/^\d{2}:\d{2}$/.test(workoutEnd)) return null;
    const d = screenshotExtras ? screenshotBaseDate(screenshotExtras.date) : (() => { const t = new Date(); t.setHours(0, 0, 0, 0); return t; })();
    const [h, m] = workoutEnd.split(":").map(Number);
    d.setHours(h, m, 0, 0);
    return d;
  })();
  const workoutStartDate = workoutEndDate && workoutDetected && workoutDetected.start && Math.abs(workoutDetected.end - workoutEndDate) <= 30 * 60_000
    ? workoutDetected.start : null;

  const setManualField = (key, v) => { setManualError(""); setManualReading((p) => ({ ...p, [key]: v })); };

  const dayHighIncluded = screenshotExtras && screenshotExtras.bpmHigh != null && (exercised === "yes" || includeDayHigh);
  const screenshotExtraCount = screenshotExtras
    ? screenshotExtras.readings.length + (dayHighIncluded ? 1 : 0)
    : 0;

  const addManualReading = () => {
    if (manualReading.bpm == null) return;
    if (manualReading.activity_state !== "resting" && manualReading.activity_state !== "exercise") {
      setManualError("Choose the activity — was this reading taken at rest or during exercise?");
      return;
    }
    const problems = implausibleFields(manualReading);
    if (problems.length > 0) {
      setManualError(`That doesn't look like a real reading — ${problems.join("; ")}. Double-check the numbers.`);
      return;
    }
    setManualError("");
    const timestamp = new Date().toISOString();
    const entry = { ...manualReading, timestamp };
    if (screenshotExtraCount > 0) {
      // One press covers the value in the form plus everything else on the
      // screen, so the verdict reflects the whole screenshot, not just the
      // one number that happened to fill the form.
      entry.source = screenshotExtras.bpmResting === manualReading.bpm ? "Resting (day summary)" : "Primary reading";
      const entries = [entry, ...screenshotReadingsToEntries(screenshotExtras, dayHighIncluded, workoutEndDate, workoutStartDate)];
      let scored = scoreReadings(baseline, entries).map((r) => (
        r.source && r.source.startsWith("Day high (activity unknown)")
          ? { ...r, explanation: `${r.explanation} Day high — activity unknown; expected if it was during exercise.` }
          : r
      ));
      const { rows: withContext, summary: recovery } = applyRecoveryContext(scored, workoutEndDate, workoutStartDate);
      scored = withContext;
      const sym = symptomEscalation(symptoms);
      if (sym.bucket && BUCKET_RANK[sym.bucket] > (BUCKET_RANK[scored[0].risk_bucket] || 0)) {
        scored[0] = { ...scored[0], risk_bucket: sym.bucket, explanation: `${scored[0].explanation} ${sym.text}` };
      }
      setCheckLog((prev) => [...prev, ...scored]);
      setHistory((prev) => [...prev, ...entries]);
      const c = { normal: 0, expected_recovery: 0, during_exercise: 0, monitor: 0, flag_for_review: 0, insufficient_baseline: 0 };
      scored.forEach((r) => { c[r.risk_bucket] = (c[r.risk_bucket] || 0) + 1; });
      const summary = [c.flag_for_review && `${c.flag_for_review} flagged for review`, c.monitor && `${c.monitor} to monitor`, c.expected_recovery && `${c.expected_recovery} expected recovery`, c.normal && `${c.normal} normal`].filter(Boolean).join(", ");
      const primary = scored[0];
      const recRows = scored.filter((r) => r.risk_bucket === "expected_recovery");
      const recoveryNote = recRows.length
        ? ` The ${recRows.length} alert reading${recRows.length === 1 ? "" : "s"} (${fmt(Math.min(...recRows.map((r) => r.bpm)))}–${fmt(Math.max(...recRows.map((r) => r.bpm)))} bpm) fall in the two hours after your workout, when heart rate normally stays elevated${recovery.trend ? ` — ${recovery.trend}` : ""}: expected recovery, not a warning sign.`
        : "";
      const exRows = scored.filter((r) => r.risk_bucket === "during_exercise");
      const exerciseNote = exRows.length ? ` The day high (${fmt(Math.max(...exRows.map((r) => r.bpm)))} bpm) happened during the workout itself.` : "";
      const symptomNote = sym.text ? ` ${sym.text}` : "";
      const lead = primary.risk_bucket === "normal" || primary.risk_bucket === "expected_recovery"
        ? `Your ${entry.source.toLowerCase()} value (${fmt(entry.bpm)} bpm) is within your typical range, but the rest of the screenshot isn't all clear`
        : `Your ${entry.source.toLowerCase()} value (${fmt(entry.bpm)} bpm) is rated ${BUCKET_LABELS[primary.risk_bucket] || primary.risk_bucket}`;
      const allNormal = c.monitor === 0 && c.flag_for_review === 0;
      setScreenshotBatchNote(allNormal
        ? `Checked ${scored.length} readings from the screenshot — nothing to worry about.${recoveryNote}${exerciseNote}${symptomNote}`
        : `Checked ${scored.length} readings from the screenshot: ${summary}. ${lead} — see Recent checks above.${recoveryNote}${exerciseNote}${symptomNote}`);
      setScreenshotExtras(null);
      setWorkoutDetected(null);
    } else {
      let [scoredEntry] = scoreReadings(baseline, [entry]);
      scoredEntry = applyRecoveryContext([scoredEntry], workoutEndDate, workoutStartDate).rows[0];
      const sym = symptomEscalation(symptoms);
      if (sym.bucket && BUCKET_RANK[sym.bucket] > (BUCKET_RANK[scoredEntry.risk_bucket] || 0)) {
        scoredEntry = { ...scoredEntry, risk_bucket: sym.bucket, explanation: `${scoredEntry.explanation} ${sym.text}` };
      }
      setCheckLog((prev) => [...prev, scoredEntry]);
      setHistory((prev) => [...prev, entry]);
    }
    setManualReading(emptyManualReading());
    setJustAdded(true);
    setTimeout(() => setJustAdded(false), 2200);
  };

  const TIER1_FIELD_LABELS = Object.fromEntries(WEARABLE_FIELD_DEFS.map((f) => [f.key, f.label]));
  TIER1_FIELD_LABELS.activity_state = "Activity";
  const handleTier1Screenshot = async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    setTier1ImgLoading(true); setTier1ImgError(""); setTier1ImgNote(""); setManualError(""); setScreenshotBatchNote("");
    try {
      const extracted = await readFileWithGemini(file, TIER1_IMAGE_PROMPT);
      const rejected = implausibleFields(extracted);
      nullOutImplausible(extracted);
      const extras = sanitizeScreenshotExtras(extracted);
      // The primary value must not depend on how the model ordered things:
      // bpm -> else the Resting value -> else the most recent listed reading.
      if (extracted.bpm == null) {
        const { min, max } = PLAUSIBLE_RANGES.bpm;
        const resting = parseFloat(extracted.bpm_resting);
        if (Number.isFinite(resting) && resting >= min && resting <= max) extracted.bpm = resting;
        else if (extras && extras.readings.length) extracted.bpm = extras.readings[extras.readings.length - 1].bpm;
      }
      // Activity is only ever taken from the screen, never assumed: the model's
      // explicit workout/rest state, or the watch's own "Resting" label when that
      // is the value being checked. Anything else stays unknown for the user to set.
      {
        const act = String(extracted.activity_state ?? "").trim().toLowerCase();
        const restingLabelled = extracted.bpm != null && Number.isFinite(parseFloat(extracted.bpm_resting)) && Number(extracted.bpm) === parseFloat(extracted.bpm_resting);
        extracted.activity_state = act === "resting" || act === "exercise" ? act : restingLabelled ? "resting" : null;
      }
      setScreenshotExtras(extras);
      const wk = detectedWorkout(extras);
      setWorkoutDetected(wk);
      setExercised(wk ? "yes" : "unsure");
      if (wk) setWorkoutEnd(`${String(wk.end.getHours()).padStart(2, "0")}:${String(wk.end.getMinutes()).padStart(2, "0")}`);
      // A watch alerts on every inactive reading above its limit: a day high above
      // every listed alert was not an inactive reading, so it is left out by default.
      const aboveAlerts = !!(extras && extras.bpmHigh != null && extras.readings.length && extras.bpmHigh > Math.max(...extras.readings.map((r) => r.bpm)));
      setIncludeDayHigh(!aboveAlerts);
      setManualReading((prev) => {
        const next = { ...prev };
        WEARABLE_FIELD_DEFS.forEach((f) => { next[f.key] = extracted[f.key] ?? prev[f.key]; });
        next.activity_state = extracted.activity_state ?? null;   // known from the screen, or left for the user to choose
        return next;
      });
      if (WEARABLE_OPTIONAL_KEYS.some((k) => extracted[k] != null)) setShowMoreWearableFields(true);
      const found = Object.keys(TIER1_FIELD_LABELS).filter((k) => extracted[k] != null);
      const foundText = found.length ? `Found: ${found.map((k) => TIER1_FIELD_LABELS[k]).join(", ")}.` : "Nothing recognizable was found.";
      const rejectedText = rejected.length ? ` Ignored implausible values: ${rejected.join("; ")}.` : "";
      setTier1ImgNote(`${foundText}${rejectedText} Review before adding.`);
    } catch (err) {
      setTier1ImgError(err.message || "Couldn't read that image. Check your API key or enter values manually.");
    } finally { setTier1ImgLoading(false); e.target.value = ""; }
  };

  /* --- Tier 2 --- */
  const [snapshotResult, setSnapshotResult] = useState({});
  const [customFeatures, setCustomFeatures] = useState({ mean_rr_ms: null, std_rr_ms: null, pnn50: null, qrs_width_ms: null });
  const [customResult, setCustomResult] = useState(null);
  const [customError, setCustomError] = useState("");
  const [tier2FileLoading, setTier2FileLoading] = useState(false);
  const [tier2FileError, setTier2FileError] = useState("");
  const [tier2FileNote, setTier2FileNote] = useState("");
  const [customExtraFeatures, setCustomExtraFeatures] = useState(Object.fromEntries(ECG_EXTRA_KEYS.map((k) => [k, null])));
  const [showMoreEcgFields, setShowMoreEcgFields] = useState(false);

  const classifySnapshot = (idx) => setSnapshotResult((prev) => ({ ...prev, [idx]: classifyECG(SAMPLE_ECG_READINGS[idx]) }));
  const classifyCustom = () => {
    const combined = { ...customFeatures, ...customExtraFeatures };
    const problems = implausibleFields(combined);
    if (problems.length > 0) {
      setCustomError(`That doesn't look like a real reading — ${problems.join("; ")}. Double-check the numbers.`);
      setCustomResult(null);
      return;
    }
    setCustomError("");
    setCustomResult(classifyECG(combined));
  };

  const TIER2_ALL_LABELS = Object.fromEntries(ECG_FIELD_DEFS.map((f) => [f.key, f.label]));
  const handleTier2FileUpload = async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    setTier2FileLoading(true); setTier2FileError(""); setTier2FileNote(""); setCustomResult(null); setCustomError("");
    const lowerName = (file.name || "").toLowerCase();
    const isCsv = file.type.includes("csv") || lowerName.endsWith(".csv");
    try {
      let matched;
      if (isCsv) {
        const rows = await parseCsvAsync(file);
        const normalizedRows = rows.map((r) => normalizeRowByAliases(r, ECG_FIELD_DEFS));
        matched = normalizedRows.find((r) => Object.keys(r).length > 0);
        if (!matched) throw new Error("No matching columns found in that CSV.");
      } else {
        matched = await readFileWithGemini(file, TIER2_FILE_PROMPT);
      }
      const rejected = implausibleFields(matched);
      nullOutImplausible(matched);
      setCustomFeatures((prev) => {
        const next = { ...prev };
        ECG_CORE_KEYS.forEach((k) => { next[k] = matched[k] ?? prev[k]; });
        return next;
      });
      setCustomExtraFeatures((prev) => {
        const next = { ...prev };
        ECG_EXTRA_KEYS.forEach((k) => { next[k] = matched[k] ?? prev[k]; });
        return next;
      });
      if (ECG_EXTRA_KEYS.some((k) => matched[k] != null)) setShowMoreEcgFields(true);
      const found = Object.keys(TIER2_ALL_LABELS).filter((k) => matched[k] != null);
      const foundText = found.length ? `Found: ${found.map((k) => TIER2_ALL_LABELS[k]).join(", ")}.` : "Nothing recognizable was found.";
      const rejectedText = rejected.length ? ` Ignored implausible value: ${rejected.join("; ")}.` : "";
      setTier2FileNote(`${isCsv ? "From CSV — " : "Vision AI — "}${foundText}${rejectedText} Review before checking.`);
    } catch (err) {
      setTier2FileError(err.message || "Couldn't read that file. Try a CSV, PDF, or image, or enter values by hand.");
    } finally { setTier2FileLoading(false); e.target.value = ""; }
  };

  /* --- derived bits for the page --- */
  const displayName = user.name || user.email.split("@")[0];
  const initials = displayName.slice(0, 1).toUpperCase();
  const sourceLine = mode === "dummy"
    ? `Sample data · ${history.length} readings`
    : `Your data · ${history.length} readings`;
  const recent = useMemo(() => [...checkLog].reverse(), [checkLog]);

  const renderNav = (cls, activeCls) => NAV.map(({ id, label, Icon, tint, filled }) => (
    <button type="button" key={id} onClick={() => setTab(id)} className={`${cls}${tab === id ? ` ${activeCls}` : ""}`}>
      <span className="ico" style={{ color: tint }}>
        <Icon size={cls === "tab" ? 22 : 18} fill={filled ? "currentColor" : "none"} strokeWidth={filled ? 0 : 2} />
      </span>
      {label}
    </button>
  ));

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="app-icon"><Heart size={16} fill="currentColor" strokeWidth={0} /></div>
          <span className="brand-name">Pulseware</span>
        </div>
        <nav className="nav">{renderNav("nav-item", "is-active")}</nav>
        <div className="sidebar-spacer" />
        <div className="me">
          <div className="avatar">{initials}</div>
          <div className="me-text">
            <div className="me-name">{displayName}</div>
            <div className="me-mail">{user.email}</div>
          </div>
          <button type="button" className="icon-btn" title="Log out" onClick={onLogout}><LogOut size={16} /></button>
        </div>
      </aside>

      <main className="main">
        <div className="page">
          {/* ---------------- TIER 1 ---------------- */}
          {tab === "tier1" && (
            <>
              <header className="page-head">
                <div>
                  <div className="eyebrow">{todayEyebrow()}</div>
                  <h1 className="large-title">Monitoring</h1>
                  <p className="page-sub">Hi {displayName}, here's how you're trending. {sourceLine}.</p>
                </div>
                <div className="page-actions">
                  <div className="avatar">{initials}</div>
                  <button type="button" className="link is-small" onClick={onLogout}>Log Out</button>
                </div>
              </header>

              <Section
                title="Add a reading"
                footer={
                  manualError ? manualError
                    : tier1ImgError ? tier1ImgError
                    : screenshotBatchNote ? screenshotBatchNote
                    : tier1ImgNote ? tier1ImgNote
                    : justAdded ? "Added to your history."
                    : "Heart rate is required. Upload a smartwatch screenshot or enter readings manually."
                }
                footerTone={manualError || tier1ImgError ? "error" : justAdded || screenshotBatchNote ? "ok" : undefined}
              >
                <div className="group">
                  <FileRow label="Read from Screenshot…" sub="Gemini Vision AI" icon={ImageIcon} accept="image/*" onFile={handleTier1Screenshot} loading={tier1ImgLoading} />
                  <NumberRow label="Heart Rate" unit="BPM" value={manualReading.bpm} placeholder="e.g. 68" onChange={(v) => setManualField("bpm", v)} />
                  <NumberRow label="HRV (RMSSD)" unit="ms" value={manualReading.hrv_rmssd} placeholder="Optional" onChange={(v) => setManualField("hrv_rmssd", v)} />
                  <NumberRow label="HRV (SDNN)" unit="ms" value={manualReading.hrv_sdnn} placeholder="Optional" onChange={(v) => setManualField("hrv_sdnn", v)} />
                  <div className="row">
                    <span className="field-label">Activity</span>
                    <span className="row-main">{manualReading.activity_state ? null : <span className="row-sub">Choose one</span>}</span>
                    <Segmented value={manualReading.activity_state} onChange={(v) => setManualField("activity_state", v)} options={[["resting", "Resting"], ["exercise", "Exercise"]]} />
                  </div>
                  <DisclosureRow
                    open={showMoreWearableFields}
                    onToggle={() => setShowMoreWearableFields((o) => !o)}
                    label={showMoreWearableFields ? "Hide optional metrics" : "More metrics (optional)"}
                    sub={showMoreWearableFields ? undefined : "SpO2, respiratory rate, skin temp, VO2 max, readiness"}
                  />
                  {showMoreWearableFields && WEARABLE_FIELD_DEFS.filter((f) => WEARABLE_OPTIONAL_KEYS.includes(f.key)).map((f) => (
                    <NumberRow key={f.key} label={f.label} unit={f.unit} value={manualReading[f.key]} placeholder="Optional" onChange={(v) => setManualField(f.key, v)} />
                  ))}
                </div>
                <div className="section-h" style={{ marginTop: 16 }}><h2>Context</h2></div>
                <div className="group">
                  <div className="row">
                    <span className="field-label">Exercised today?</span>
                    <span className="row-main" />
                    <Segmented value={exercised} onChange={setExercised} options={[["no", "No"], ["yes", "Yes"], ["unsure", "Not sure"]]} />
                  </div>
                  {exercised === "yes" && (
                    <div className="row">
                      <span className="row-main">
                        <span className="field-label">Workout ended</span>
                        {workoutDetected && <span className="row-sub">From the screenshot ({workoutDetected.label}, approximate) — adjust if wrong.</span>}
                      </span>
                      <input type="time" step="900" value={workoutEnd} onChange={(e) => setWorkoutEnd(e.target.value)} className="time-input" />
                    </div>
                  )}
                  {exercised !== "yes" && workoutDetected && (
                    <div className="row"><span className="row-sub">This screenshot seems to show a {workoutDetected.label} — choose Yes if that's right.</span></div>
                  )}
                  <DisclosureRow
                    open={showSymptoms}
                    onToggle={() => setShowSymptoms((o) => !o)}
                    label={symptoms.length ? `Symptoms today (${symptoms.length})` : "Any symptoms today?"}
                    sub={showSymptoms ? undefined : symptoms.length ? symptoms.join(", ") : "None reported — symptoms outweigh the numbers"}
                  />
                  {showSymptoms && SYMPTOM_OPTIONS.map((opt) => (
                    <label key={opt} className="row" style={{ cursor: "pointer" }}>
                      <span className="row-main"><span className="row-title">{opt}</span></span>
                      <input type="checkbox" checked={symptoms.includes(opt)} onChange={(e) => setSymptoms((prev) => (e.target.checked ? [...prev, opt] : prev.filter((x) => x !== opt)))} style={{ width: 18, height: 18, accentColor: "var(--blue)" }} />
                    </label>
                  ))}
                </div>
                <div className="btn-wrap">
                  <button type="button" className="btn" onClick={addManualReading} disabled={manualReading.bpm == null}>
                    {justAdded ? <><Check size={18} /> Added</> : screenshotExtraCount > 0 ? `Add all ${screenshotExtraCount + 1} readings from this screenshot` : "Add Reading"}
                  </button>
                </div>
              </Section>

              {screenshotExtras && (() => {
                const listed = screenshotExtras.readings;
                const lo = listed.length ? Math.min(...listed.map((r) => r.bpm)) : null;
                const hi = listed.length ? Math.max(...listed.map((r) => r.bpm)) : null;
                const times = listed.map((r) => r.time).filter(Boolean);
                const span = times.length >= 2 ? `${times[0]} – ${times[times.length - 1]}` : times[0] || "";
                return (
                  <Section
                    title="Also on this screenshot"
                    footer={`All of these are checked together with the value above when you press "Add all ${screenshotExtraCount + 1} readings".`}
                  >
                    <div className="group">
                      {listed.length > 0 && (
                        <div className="row">
                          <span className="ico" style={{ color: "var(--red)" }}><AlertCircle size={20} /></span>
                          <span className="row-main">
                            <span className="row-title">{listed.length} listed {listed[0].label.toLowerCase()} reading{listed.length === 1 ? "" : "s"}</span>
                            <span className="row-sub">{lo === hi ? `${fmt(lo)} bpm` : `${fmt(lo)}–${fmt(hi)} bpm`}{span ? ` · ${span}` : ""}</span>
                          </span>
                        </div>
                      )}
                      {screenshotExtras.bpmHigh != null && exercised === "yes" && (
                        <div className="row">
                          <span className="ico" style={{ color: "var(--blue)" }}><Activity size={20} /></span>
                          <span className="row-main">
                            <span className="row-title">Day high {fmt(screenshotExtras.bpmHigh)} bpm</span>
                            <span className="row-sub">Treated as part of your workout, not as a resting reading.</span>
                          </span>
                        </div>
                      )}
                      {screenshotExtras.bpmHigh != null && exercised !== "yes" && (
                        <label className="row" style={{ cursor: "pointer" }}>
                          <span className="ico" style={{ color: "var(--orange)" }}><Activity size={20} /></span>
                          <span className="row-main">
                            <span className="row-title">Day high {fmt(screenshotExtras.bpmHigh)} bpm</span>
                            <span className="row-sub">
                              {highAboveAlerts
                                ? "Above every listed alert, so the watch didn't record it as an inactive reading — it most likely happened during activity. Tick to check it as a resting reading anyway."
                                : "Activity at that moment isn't shown, so it's checked as a resting reading. Untick to leave it out."}
                            </span>
                          </span>
                          <input type="checkbox" checked={includeDayHigh} onChange={(e) => setIncludeDayHigh(e.target.checked)} style={{ width: 18, height: 18, accentColor: "var(--blue)" }} />
                        </label>
                      )}
                    </div>
                  </Section>
                );
              })()}

              <Section
                title="Data"
                footer={uploadError || `${history.length} history points · ${checkLog.length} checks logged`}
                footerTone={uploadError ? "error" : undefined}
              >
                <div className="group">
                  <button type="button" className="row is-action" style={{ width: "100%", background: "none", border: "none", textAlign: "left" }} onClick={useSampleData}>
                    <span className="ico"><Activity size={18} /></span>
                    <span className="row-main"><span className="row-title">Use Sample Data</span></span>
                    {mode === "dummy" && <Check size={18} />}
                  </button>
                  <FileRow label="Upload History CSV…" sub="Replaces your baseline history" icon={Upload} accept=".csv" onFile={handleHistoryUpload} />
                  <FileRow label="Upload New Readings CSV…" sub="Checked against your current baseline" icon={FileText} accept=".csv" onFile={handleNewUpload} />
                </div>
              </Section>

              <Section
                title="Vision AI"
                footer="Screenshots and PDFs are read with Google Gemini. Get a free key at aistudio.google.com — it stays in this browser only."
              >
                <div className="group">
                  <ApiKeyRow value={geminiKey} onChange={setGeminiKey} />
                </div>
              </Section>

              <Section title="About these measurements">
                <Glossary items={TIER1_GLOSSARY} />
              </Section>
            </>
          )}

          {/* ---------------- HISTORY ---------------- */}
          {tab === "history" && (
            <>
              <header className="page-head">
                <div>
                  <div className="eyebrow">{todayEyebrow()}</div>
                  <h1 className="large-title">History</h1>
                  <p className="page-sub">Your past heart rate, HRV and recent checks.</p>
                </div>
                <div className="page-actions">
                  <div className="avatar">{initials}</div>
                  <button type="button" className="link is-small" onClick={onLogout}>Log Out</button>
                </div>
              </header>

              <Section>
                <HeartRateCard history={history} />
              </Section>

              {counts.flag_for_review > 0 && (
                <Section>
                  <div className="group">
                    <button type="button" className="row" style={{ width: "100%", background: "none", border: "none", textAlign: "left", cursor: "pointer" }} onClick={() => setTab("tier2")}>
                      <span className="ico" style={{ color: "var(--red)" }}><AlertCircle size={20} /></span>
                      <span className="row-main">
                        <span className="row-title" style={{ fontWeight: 600 }}>{counts.flag_for_review} reading(s) flagged for review</span>
                        <span className="row-sub">Head to Rhythm Check if you'd like a closer look.</span>
                      </span>
                      <ChevronRight size={16} className="chev" />
                    </button>
                  </div>
                </Section>
              )}

              <Section title="Recent checks">
                <div className="table-wrap">
                  <div className="legend">
                    <span className="status normal">{counts.normal} Normal</span>
                    <span className="status monitor">{counts.monitor} Monitor</span>
                    <span className="status flag_for_review">{counts.flag_for_review} Flag for review</span>
                    {counts.expected_recovery > 0 && <span className="status expected_recovery">{counts.expected_recovery} Expected recovery</span>}
                    {counts.during_exercise > 0 && <span className="status during_exercise">{counts.during_exercise} During exercise</span>}
                    {counts.insufficient_baseline > 0 && <span className="status insufficient_baseline">{counts.insufficient_baseline} Still learning</span>}
                  </div>
                  {recent.length === 0 ? (
                    <div className="empty">No checks yet. Add a reading below or import new readings.</div>
                  ) : (
                    <>
                      <table className="data checks-table">
                        <thead>
                          <tr>
                            <th>Time</th><th>BPM</th><th>HRV (RMSSD)</th><th>HRV (SDNN)</th>
                            {visibleOptionalCols.map((f) => <th key={f.key}>{f.label}</th>)}
                            <th>Status</th><th>Why</th>
                          </tr>
                        </thead>
                        <tbody>
                          {recent.map((r, i) => (
                            <tr key={i}>
                              <td style={{ minWidth: 150 }}>
                                <span style={{ whiteSpace: "nowrap" }}>{fmtTime(new Date(r.timestamp).getTime())}</span>
                                <div className="row-sub"><span className="cap">{r.activity_state}</span>{r.source ? ` · ${r.source}` : ""}</div>
                              </td>
                              <td className="num" style={{ fontWeight: 600 }}>{fmt(r.bpm)}</td>
                              <td className="num">{fmt(r.hrv_rmssd)}</td>
                              <td className="num">{fmt(r.hrv_sdnn)}</td>
                              {visibleOptionalCols.map((f) => <td key={f.key} className="num">{fmt(r[f.key])}</td>)}
                              <td><span className={`status ${r.risk_bucket}`}>{BUCKET_LABELS[r.risk_bucket] || r.risk_bucket}</span></td>
                              <td className="why">{r.explanation}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {/* Narrow screens get the same data as a list instead of a scrolling table. */}
                      <div className="checks-list">
                        {recent.map((r, i) => (
                          <div className="row" key={i} style={{ alignItems: "flex-start" }}>
                            <span className="row-main">
                              <span className="row-title">
                                <b className="num">{fmt(r.bpm)}</b> BPM
                                <span className="dim"> · HRV {fmt(r.hrv_rmssd)}/{fmt(r.hrv_sdnn)} ms · </span>
                                <span className="dim cap">{r.activity_state}</span>
                              </span>
                              <span className="row-sub">{fmtTime(new Date(r.timestamp).getTime())}{r.source ? ` · ${r.source}` : ""}</span>
                              <span className="row-sub">{r.explanation}</span>
                            </span>
                            <span className={`status ${r.risk_bucket}`} style={{ marginTop: 3 }}>{BUCKET_LABELS[r.risk_bucket] || r.risk_bucket}</span>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              </Section>

            </>
          )}

          {/* ---------------- TIER 2 ---------------- */}

          {tab === "tier2" && (
            <>
              <header className="page-head">
                <div>
                  <div className="eyebrow">{todayEyebrow()}</div>
                  <h1 className="large-title">Rhythm Check</h1>
                  <p className="page-sub">Upload a reading or enter the numbers yourself.</p>
                </div>
                <div className="page-actions">
                  <div className="avatar">{initials}</div>
                  <button type="button" className="link is-small" onClick={onLogout}>Log Out</button>
                </div>
              </header>

              <Section title="Sample readings">
                <div className="group">
                  {SAMPLE_ECG_READINGS.map((snap, idx) => (
                    <div className="cell" key={idx}>
                      <div className="cell-head">
                        <span className="row-main">
                          <span className="row-title">{snap.hint}</span>
                          <span className="row-sub num">
                            Mean RR {snap.mean_rr_ms} ms · RR variation {snap.std_rr_ms} ms · pNN50 {snap.pnn50}% · QRS {snap.qrs_width_ms} ms
                          </span>
                        </span>
                        <button type="button" className="link" onClick={() => classifySnapshot(idx)}>
                          {snapshotResult[idx] ? "Check Again" : "Check"}
                        </button>
                      </div>
                      <Verdict result={snapshotResult[idx]} />
                    </div>
                  ))}
                </div>
              </Section>

              <Section
                title="Your reading"
                footer={
                  customError ? customError
                    : tier2FileError ? tier2FileError
                    : tier2FileNote ? tier2FileNote
                    : "Uses Google Gemini Vision AI to extract parameters from reports, rhythm strips, and ECG printouts automatically."
                }
                footerTone={customError || tier2FileError ? "error" : undefined}
              >
                <div className="group">
                  <FileRow label="Upload ECG…" sub="Image, PDF, or CSV" icon={Upload} accept=".csv,.pdf,image/*" onFile={handleTier2FileUpload} loading={tier2FileLoading} />
                  {[["mean_rr_ms", "Mean RR", "ms", "e.g. 850"], ["std_rr_ms", "RR variation", "ms", "e.g. 35"], ["pnn50", "pNN50", "%", "e.g. 18"], ["qrs_width_ms", "QRS width", "ms", "e.g. 90"]].map(([key, label, unit, placeholder]) => (
                    <NumberRow
                      key={key} label={label} unit={unit} placeholder={placeholder} value={customFeatures[key]}
                      onChange={(v) => { setCustomError(""); setCustomFeatures((prev) => ({ ...prev, [key]: v })); }}
                    />
                  ))}
                  <DisclosureRow
                    open={showMoreEcgFields}
                    onToggle={() => setShowMoreEcgFields((o) => !o)}
                    label={showMoreEcgFields ? "Hide extra measurements" : "More measurements (optional)"}
                    sub={showMoreEcgFields ? "QTc, QRS width & P wave feed the safety checks." : "PR/QT interval, axis, voltage — used for guardrails"}
                  />
                  {showMoreEcgFields && ECG_FIELD_DEFS.filter((f) => ECG_EXTRA_KEYS.includes(f.key)).map((f) => (
                    <NumberRow
                      key={f.key} label={f.label} unit={f.unit} placeholder="Optional" value={customExtraFeatures[f.key]}
                      onChange={(v) => { setCustomError(""); setCustomExtraFeatures((prev) => ({ ...prev, [f.key]: v })); }}
                    />
                  ))}
                  <ApiKeyRow value={geminiKey} onChange={setGeminiKey} />
                </div>
                <div className="btn-wrap">
                  <button type="button" className="btn" onClick={classifyCustom}>Check This Reading</button>
                </div>
                <Verdict result={customResult} />
              </Section>

              <Section title="About these measurements">
                <Glossary items={TIER2_GLOSSARY} />
              </Section>
            </>
          )}

          {/* ---------------- ABOUT ---------------- */}
          {tab === "about" && (
            <>
              <header className="page-head">
                <div>
                  <div className="eyebrow">Pulseware</div>
                  <h1 className="large-title">About</h1>
                  <p className="page-sub">How this all works, in plain language.</p>
                </div>
                <div className="page-actions">
                  <div className="avatar">{initials}</div>
                  <button type="button" className="link is-small" onClick={onLogout}>Log Out</button>
                </div>
              </header>
              <AboutPage />
            </>
          )}
        </div>
      </main>

      <nav className="tabbar">{renderNav("tab", "is-active")}</nav>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* App root                                                            */
/* ------------------------------------------------------------------ */
export default function App() {
  const [user, setUser] = useState(null);
  const [justSignedOut, setJustSignedOut] = useState(false);
  const doAuthed = (u) => { setUser(u); setJustSignedOut(false); };
  const doLogout = () => { setUser(null); setJustSignedOut(true); };
  if (!user) {
    if (justSignedOut) return <SignedOutPage onSignIn={() => setJustSignedOut(false)} />;
    return <SignIn onAuthed={doAuthed} />;
  }
  return <Dashboard user={user} onLogout={doLogout} />;
}
