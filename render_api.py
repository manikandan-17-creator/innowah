"""
INNOWAH 2026 — Render ML Prediction API
────────────────────────────────────────
A lightweight Flask API that loads the trained INNOWAH model
and serves predictions. Deployed on Render as a Web Service.

Endpoints:
  POST /predict        — accepts a 31-dim feature vector, returns risk prediction
  POST /predict_raw    — accepts raw software parameters, builds features internally
  GET  /health         — health check for Render monitoring
  GET  /               — landing page / status
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your website

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH  = os.environ.get("MODEL_PATH",  "model/innowah_model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "model/innowah_scaler.pkl")

innowah_model  = None
innowah_scaler = None

try:
    innowah_model  = joblib.load(MODEL_PATH)
    innowah_scaler = joblib.load(SCALER_PATH)
    logger.info(f"✅ INNOWAH model loaded from {MODEL_PATH}")
    logger.info(f"✅ INNOWAH scaler loaded from {SCALER_PATH}")
except Exception as e:
    logger.error(f"❌ Could not load model: {e}")
    logger.info("The API will use rule-based fallback until model files are available.")


# ──────────────────────────────────────────────────────────────────────────────
# NORMALIZATION PARAMETERS (same as app.py)
# ──────────────────────────────────────────────────────────────────────────────

NORM_PARAMS = {
    "gait_speed":          (0.0,    1.6,   True),
    "stride_variability":  (0.0,    10.0,  False),
    "turning_velocity":    (0.0,    200.0, True),
    "postural_sway":       (0.0,    10.0,  False),
    "rmssd":               (0.0,    80.0,  True),
    "sdnn":                (0.0,    100.0, True),
    "lf_hf_ratio":         (0.0,    5.0,   False),
    "spo2":                (85.0,   100.0, True),
    "desat_events":        (0.0,    10.0,  False),
    "alpha_power":         (0.0,    50.0,  True),
    "theta_power":         (0.0,    50.0,  False),
    "delta_power":         (0.0,    40.0,  False),
    "theta_alpha_ratio":   (0.0,    3.0,   False),
    "dominant_frequency":  (5.0,    12.0,  True),
    "daily_steps":         (0.0,    10000, True),
    "reaction_time":       (200.0,  2000.0, False),
    "verbal_fluency":      (0.0,    30.0,  True),
    "iadl_impairments":    (0.0,    8.0,   False),
}


def _norm(value, key):
    """Normalize raw sensor/cognitive value → [0,1] where 1 = healthy."""
    if key not in NORM_PARAMS:
        return float(np.clip(value, 0.0, 1.0))
    lo, hi, higher_is_better = NORM_PARAMS[key]
    n = float(np.clip((float(value) - lo) / (hi - lo + 1e-9), 0.0, 1.0))
    return n if higher_is_better else (1.0 - n)


def _get(d, key, default=0.5):
    v = d.get(key, default)
    return 0.5 if v is None else float(v)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (same as app.py — builds 31-dim vector from raw params)
# ──────────────────────────────────────────────────────────────────────────────

def extract_features_from_raw(software_data: dict, hardware_data: dict = None) -> np.ndarray:
    """
    Build a 31-dim feature vector from raw software parameters.
    Hardware data defaults to midpoint (0.5) when not provided.
    """
    sw = software_data or {}
    hw = hardware_data or {}
    imu = hw.get("imu", {})
    ppg = hw.get("ppg", {})
    eeg = hw.get("eeg", {})

    # Software features [0–13]
    sw_vec = np.array([
        _get(sw, "immediate_recall",       0.5),
        _get(sw, "delayed_recall",         0.5),
        _get(sw, "cue_benefit_index",      0.5),
        _get(sw, "retention_ratio",        0.5),
        _get(sw, "orientation_score",      0.5),
        _get(sw, "serial7s_score",         0.5),
        _get(sw, "clock_drawing_score",    0.5),
        _norm(_get(sw, "reaction_time_ms", 700), "reaction_time"),
        1.0 - _get(sw, "error_consistency_norm", 0.2),
        _get(sw, "naming_task_score",      0.5),
        _get(sw, "sentence_repetition_score", 0.5),
        _norm(_get(sw, "verbal_fluency_wpm", 12), "verbal_fluency"),
        _norm(_get(sw, "iadl_impairment_count", 1), "iadl_impairments"),
        _get(sw, "mood_filter",            0.0),
    ], dtype=np.float32)

    # Hardware features [14–28]
    hw_vec = np.array([
        _norm(_get(imu, "gait_speed",          1.1),  "gait_speed"),
        _norm(_get(imu, "stride_variability",  2.5),  "stride_variability"),
        _norm(_get(imu, "turning_velocity",    130),  "turning_velocity"),
        _norm(_get(imu, "postural_sway",       2.0),  "postural_sway"),
        _norm(_get(ppg, "rmssd",               35),   "rmssd"),
        _norm(_get(ppg, "sdnn",                45),   "sdnn"),
        _norm(_get(ppg, "lf_hf_ratio",         1.5),  "lf_hf_ratio"),
        _norm(_get(ppg, "spo2",                97),   "spo2"),
        _norm(_get(ppg, "desat_events",        0),    "desat_events"),
        _norm(_get(eeg, "alpha_power",         30),   "alpha_power"),
        _norm(_get(eeg, "theta_power",         15),   "theta_power"),
        _norm(_get(eeg, "delta_power",         10),   "delta_power"),
        _norm(_get(eeg, "theta_alpha_ratio",   0.7),  "theta_alpha_ratio"),
        _norm(_get(eeg, "dominant_frequency",  10.0), "dominant_frequency"),
        _norm(_get(imu, "step_count",          5000), "daily_steps"),
    ], dtype=np.float32)

    # Aggregate scores [29–30]
    sensor_score    = float(np.mean(hw_vec))
    cognitive_score = float(np.mean(sw_vec))

    return np.concatenate([sw_vec, hw_vec, np.array([sensor_score, cognitive_score], dtype=np.float32)])


# ──────────────────────────────────────────────────────────────────────────────
# CLINICAL RULES (fallback when model unavailable)
# ──────────────────────────────────────────────────────────────────────────────

CLINICAL_RULES = [
    (14, "gait_speed",          0.57, 0.71, "lower_worse"),
    (15, "stride_variability",  0.70, 0.50, "higher_worse"),
    (16, "turning_velocity",    0.55, 0.40, "lower_worse"),
    (17, "postural_sway",       0.70, 0.50, "higher_worse"),
    (18, "rmssd",               0.31, 0.19, "lower_worse"),
    (19, "sdnn",                0.30, 0.20, "lower_worse"),
    (20, "lf_hf_ratio",         0.80, 0.60, "higher_worse"),
    (21, "spo2",                0.67, 0.47, "lower_worse"),
    (23, "alpha_power",         0.50, 0.40, "lower_worse"),
    (24, "theta_power",         0.60, 0.50, "higher_worse"),
    (25, "delta_power",         0.63, 0.50, "higher_worse"),
    (26, "theta_alpha_ratio",   0.47, 0.57, "higher_worse"),
    (27, "dominant_frequency",  0.57, 0.43, "lower_worse"),
    (0,  "immediate_recall",    0.70, 0.50, "lower_worse"),
    (1,  "delayed_recall",      0.70, 0.50, "lower_worse"),
    (3,  "retention_ratio",     0.70, 0.50, "lower_worse"),
    (5,  "serial7s_score",      0.60, 0.40, "lower_worse"),
    (9,  "naming_task",         0.60, 0.40, "lower_worse"),
]

DOMAIN_INDICES = {
    "memory":       [0, 1, 2, 3, 4, 18, 19, 23, 24, 25],
    "reasoning":    [5, 6, 7, 8, 14, 15, 16, 26, 27],
    "visuospatial": [17, 26],
    "language":     [9, 10, 11, 27],
    "behavior":     [12, 13, 20, 21, 22, 28],
}


def run_inference(feature_vector: np.ndarray) -> dict:
    """Run ML model or fall back to rule-based clinical engine."""
    fv = np.array(feature_vector, dtype=np.float32).flatten()

    # ── ML model path ─────────────────────────────────────────────────────
    if innowah_model is not None and innowah_scaler is not None:
        fv_scaled  = innowah_scaler.transform(fv.reshape(1, -1))
        prob       = innowah_model.predict_proba(fv_scaled)[0]
        risk_score = float(prob[1] * 50 + prob[2] * 100)
        risk_level = ["Normal", "Mild Risk", "High Risk"][int(np.argmax(prob))]
        method     = "ml_model"
    else:
        # ── Rule-based fallback ───────────────────────────────────────────
        mild_flags, high_flags = [], []
        for idx, name, normal_t, mild_t, direction in CLINICAL_RULES:
            if idx >= len(fv):
                continue
            val = fv[idx]
            if direction == "lower_worse":
                if val < mild_t:    high_flags.append(name)
                elif val < normal_t: mild_flags.append(name)
            else:
                if val > mild_t:    high_flags.append(name)
                elif val > normal_t: mild_flags.append(name)

        sensor_health  = float(fv[29]) if len(fv) > 29 else 0.5
        cog_health     = float(fv[30]) if len(fv) > 30 else 0.5
        risk_score     = min(100.0, (1 - sensor_health) * 0.60 * 100 +
                                    (1 - cog_health)    * 0.40 * 100)
        risk_level     = ("High Risk" if risk_score >= 50 else
                          "Mild Risk" if risk_score >= 25 else "Normal")
        method         = "rule_based_clinical"

    # ── Domain scores ─────────────────────────────────────────────────────
    domain_scores = {}
    for domain, indices in DOMAIN_INDICES.items():
        valid = [fv[i] for i in indices if i < len(fv)]
        domain_scores[domain] = round((1.0 - float(np.mean(valid))) * 100, 1) if valid else 50.0

    # ── Feature flags ─────────────────────────────────────────────────────
    mild_flags, high_flags = [], []
    for idx, name, normal_t, mild_t, direction in CLINICAL_RULES:
        if idx >= len(fv):
            continue
        val = fv[idx]
        if direction == "lower_worse":
            if val < mild_t:     high_flags.append(name)
            elif val < normal_t: mild_flags.append(name)
        else:
            if val > mild_t:     high_flags.append(name)
            elif val > normal_t: mild_flags.append(name)

    # ── Recommendation ────────────────────────────────────────────────────
    if risk_level == "High Risk":
        recommendation = (
            "⚠️ High risk indicators detected. Recommend neurological consultation. "
            f"Key concerns: {', '.join(high_flags[:3]) or 'multiple domains'}. "
            "Schedule follow-up within 2 weeks."
        )
    elif risk_level == "Mild Risk":
        recommendation = (
            "⚡ Mild risk indicators present. Monitor trends over time. "
            f"Areas of concern: {', '.join(mild_flags[:3]) or 'minor fluctuations'}. "
            "Repeat assessment in 30 days."
        )
    else:
        recommendation = "✅ Parameters within normal range. Continue routine monitoring monthly."

    return {
        "risk_score":      round(float(risk_score), 1),
        "risk_level":      risk_level,
        "sensor_score":    round(float(fv[29]) * 100, 1) if len(fv) > 29 else 50.0,
        "cognitive_score": round(float(fv[30]) * 100, 1) if len(fv) > 30 else 50.0,
        "domain_scores":   domain_scores,
        "feature_flags":   {"mild": mild_flags, "high": high_flags},
        "recommendation":  recommendation,
        "method":          method,
    }


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Landing page / status."""
    return jsonify({
        "service":    "INNOWAH ML Prediction API",
        "version":    "1.0.0",
        "status":     "running",
        "model_loaded": innowah_model is not None,
        "endpoints": {
            "POST /predict":     "Send a 31-dim feature vector for prediction",
            "POST /predict_raw": "Send raw software parameters (auto-builds features)",
            "GET  /health":      "Health check",
        },
        "timestamp":  datetime.utcnow().isoformat()
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a pre-computed 31-dim feature vector and returns risk prediction.

    Expected JSON:
    {
      "features": [0.67, 0.33, 0.50, ... ]   // 31 float values
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400

    features = data.get("features")
    if not features or not isinstance(features, list):
        return jsonify({"status": "error", "message": "'features' must be a list of 31 floats"}), 400

    if len(features) != 31:
        return jsonify({
            "status": "error",
            "message": f"Expected 31 features, got {len(features)}"
        }), 400

    fv = np.array(features, dtype=np.float32)
    result = run_inference(fv)

    logger.info(f"[/predict] → {result['risk_level']} (score={result['risk_score']}, method={result['method']})")

    return jsonify({
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        **result
    }), 200


@app.route("/predict_raw", methods=["POST"])
def predict_raw():
    """
    Accepts raw software/cognitive parameters, builds the 31-dim feature
    vector internally (hardware features default to healthy midpoints),
    and returns the risk prediction.

    Expected JSON:
    {
      "immediate_recall":       0.67,
      "delayed_recall":         0.33,
      "cue_benefit_index":      0.50,
      "retention_ratio":        0.50,
      "orientation_score":      0.80,
      "serial7s_score":         0.60,
      "clock_drawing_score":    0.60,
      "reaction_time_ms":       700,
      "error_consistency_norm": 0.20,
      "naming_task_score":      0.50,
      "sentence_repetition_score": 0.0,
      "verbal_fluency_wpm":     8.4,
      "iadl_impairment_count":  2,
      "mood_filter":            0,

      // Optional hardware data (if ESP32 is connected)
      "hardware": {
        "imu":  { "gait_speed": 1.1, ... },
        "ppg":  { "rmssd": 35, ... },
        "eeg":  { "alpha_power": 30, ... },
        "temperature": { "skin_temp": 35.5 }
      }
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400

    hardware = data.pop("hardware", None)

    fv = extract_features_from_raw(software_data=data, hardware_data=hardware)
    fv_list = fv.tolist()

    result = run_inference(fv)

    logger.info(f"[/predict_raw] → {result['risk_level']} (score={result['risk_score']}, method={result['method']})")

    return jsonify({
        "status":         "ok",
        "timestamp":      datetime.utcnow().isoformat(),
        "feature_vector": fv_list,
        **result
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render monitoring."""
    return jsonify({
        "status":       "healthy",
        "model_loaded": innowah_model is not None,
        "timestamp":    datetime.utcnow().isoformat()
    })


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
