from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import joblib
import numpy as np
import base64
import io
import cv2
import os
import requests
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.secret_key = 'neurobandplus_secret_key_change_in_production'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Render ML API URL (replace with actual URL when deployed) ─────────────────
RENDER_API_URL = os.environ.get("RENDER_API_URL", "https://your-app-name.onrender.com/predict")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

# ── Existing cognitive game model ─────────────────────────────────────────────
model = joblib.load("model/alz_model.joblib")

# ── INNOWAH hardware ML model (loads if trained, else uses rule-based engine) ─
INNOWAH_MODEL_PATH  = "model/innowah_model.pkl"
INNOWAH_SCALER_PATH = "model/innowah_scaler.pkl"
innowah_model  = None
innowah_scaler = None

if os.path.exists(INNOWAH_MODEL_PATH) and os.path.exists(INNOWAH_SCALER_PATH):
    try:
        innowah_model  = joblib.load(INNOWAH_MODEL_PATH)
        innowah_scaler = joblib.load(INNOWAH_SCALER_PATH)
        logger.info("INNOWAH ML model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load INNOWAH model: {e}. Using rule-based engine.")
else:
    logger.info("No INNOWAH trained model found — using rule-based clinical engine.")

# ══════════════════════════════════════════════════════════════════════════════
# STORE SCORES
# ══════════════════════════════════════════════════════════════════════════════

# ── Original cognitive game scores ────────────────────────────────────────────
cognitive_scores = {
    "memory":        None,
    "nback":         None,
    "final":         None,
    "ml_prediction": None,
    "questionnaire": None
}

# ── INNOWAH hardware session store (device_id → list of ESP32 readings) ───────
hardware_sessions = {}   # { "ESP32_INNOWAH_001": [ {...}, {...} ] }


# ══════════════════════════════════════════════════════════════════════════════
# INNOWAH FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

NORM_PARAMS = {
    # (min, max, higher_is_better)
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
    "beta_power":          (0.0,    40.0,  True),
    "theta_alpha_ratio":   (0.0,    3.0,   False),
    "dominant_frequency":  (5.0,    12.0,  True),
    "signal_entropy":      (0.0,    3.0,   True),
    "posterior_alpha":     (0.0,    50.0,  True),
    "daily_steps":         (0.0,    10000, True),
    "skin_temp":           (32.0,   38.0,  True),
    "reaction_time":       (200.0,  2000.0,False),
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

def extract_innowah_features(software_data: dict, hardware_data: dict) -> np.ndarray:
    """
    Build a 35-dim feature vector from cognitive test + ESP32 sensor data.
    All values normalized to [0,1] where 1 = healthy end of spectrum.

    Layout:
      [0–13]  Software / cognitive (14 features)
      [14–17] IMU (4 features)
      [18–22] PPG/HRV (5 features)
      [23–30] EEG (8 features)
      [31–32] Activity + temperature (2 features)
      [33–34] Aggregate sensor/cognitive scores (2 features)
    """
    sw  = software_data  or {}
    hw  = hardware_data  or {}
    imu = hw.get("imu",  {})
    ppg = hw.get("ppg",  {})
    eeg = hw.get("eeg",  {})
    tmp = hw.get("temperature", {})

    # ── Software features [0–13] ─────────────────────────────────────────────
    # Map existing cognitive game scores into the vector when available
    memory_raw = cognitive_scores["memory"]
    nback_raw  = cognitive_scores["nback"]

    immediate_recall  = _get(sw, "immediate_recall",  memory_raw if memory_raw is not None else 0.5)
    delayed_recall    = _get(sw, "delayed_recall",    memory_raw if memory_raw is not None else 0.5)
    cue_benefit       = _get(sw, "cue_benefit_index", 0.5)
    retention_ratio   = _get(sw, "retention_ratio",   0.5)
    orientation       = _get(sw, "orientation_score", 0.5)
    serial7s          = _get(sw, "serial7s_score",    nback_raw  if nback_raw  is not None else 0.5)
    clock_drawing     = _get(sw, "clock_drawing_score",0.5)
    reaction_time     = _norm(_get(sw, "reaction_time_ms", 700), "reaction_time")
    error_consistency = 1.0 - _get(sw, "error_consistency_norm", 0.2)
    naming_task       = _get(sw, "naming_task_score", 0.5)
    sentence_rep      = _get(sw, "sentence_repetition_score", 0.5)
    verbal_fluency    = _norm(_get(sw, "verbal_fluency_wpm", 12), "verbal_fluency")
    iadl              = _norm(_get(sw, "iadl_impairment_count", 1), "iadl_impairments")
    mood_filter       = _get(sw, "mood_filter", 0.0)

    sw_vec = np.array([
        immediate_recall, delayed_recall, cue_benefit, retention_ratio,
        orientation, serial7s, clock_drawing, reaction_time,
        error_consistency, naming_task, sentence_rep, verbal_fluency,
        iadl, mood_filter
    ], dtype=np.float32)

    # ── Hardware features [14–32] ────────────────────────────────────────────
    hw_vec = np.array([
        # IMU [14–17]
        _norm(_get(imu, "gait_speed",          1.1),   "gait_speed"),
        _norm(_get(imu, "stride_variability",  2.5),   "stride_variability"),
        _norm(_get(imu, "turning_velocity",    130),   "turning_velocity"),
        _norm(_get(imu, "postural_sway",       2.0),   "postural_sway"),
        # PPG/HRV [18–22]
        _norm(_get(ppg, "rmssd",               35),    "rmssd"),
        _norm(_get(ppg, "sdnn",                45),    "sdnn"),
        _norm(_get(ppg, "lf_hf_ratio",         1.5),   "lf_hf_ratio"),
        _norm(_get(ppg, "spo2",                97),    "spo2"),
        _norm(_get(ppg, "desat_events",        0),     "desat_events"),
        # EEG [23–30]
        _norm(_get(eeg, "alpha_power",         30),    "alpha_power"),
        _norm(_get(eeg, "theta_power",         15),    "theta_power"),
        _norm(_get(eeg, "delta_power",         10),    "delta_power"),
        _norm(_get(eeg, "beta_power",          20),    "beta_power"),
        _norm(_get(eeg, "theta_alpha_ratio",   0.7),   "theta_alpha_ratio"),
        _norm(_get(eeg, "dominant_frequency",  10.0),  "dominant_frequency"),
        _norm(_get(eeg, "signal_entropy",      1.8),   "signal_entropy"),
        _norm(_get(eeg, "posterior_alpha",     30),    "posterior_alpha"),
        # Activity + Temp [31–32]
        _norm(_get(imu, "step_count",          5000),  "daily_steps"),
        _norm(_get(tmp, "skin_temp",           35.5),  "skin_temp"),
    ], dtype=np.float32)

    # ── Aggregate scores [33–34] ─────────────────────────────────────────────
    sensor_score    = float(np.mean(hw_vec))
    cognitive_score = float(np.mean(sw_vec))
    agg_vec = np.array([sensor_score, cognitive_score], dtype=np.float32)

    return np.concatenate([sw_vec, hw_vec, agg_vec])


# ══════════════════════════════════════════════════════════════════════════════
# INNOWAH INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Clinical risk rules: (feature_idx, name, normal_threshold, mild_threshold, direction)
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
    (26, "beta_power",          0.30, 0.25, "lower_worse"),
    (27, "theta_alpha_ratio",   0.47, 0.57, "higher_worse"),
    (28, "dominant_frequency",  0.57, 0.43, "lower_worse"),
    (0,  "immediate_recall",    0.70, 0.50, "lower_worse"),
    (1,  "delayed_recall",      0.70, 0.50, "lower_worse"),
    (3,  "retention_ratio",     0.70, 0.50, "lower_worse"),
    (5,  "serial7s_score",      0.60, 0.40, "lower_worse"),
    (9,  "naming_task",         0.60, 0.40, "lower_worse"),
]

DOMAIN_INDICES = {
    "memory":       [0, 1, 2, 3, 4, 18, 19, 23, 24, 25],
    "reasoning":    [5, 6, 7, 8, 14, 15, 16, 26],
    "visuospatial": [17, 30],
    "language":     [9, 10, 11, 28, 29],
    "behavior":     [12, 13, 20, 21, 22, 31],
}

def run_innowah_inference(feature_vector: np.ndarray) -> dict:
    """Run ML model or fall back to rule-based clinical engine."""
    fv = np.array(feature_vector, dtype=np.float32).flatten()

    # ── ML model path ─────────────────────────────────────────────────────────
    if innowah_model is not None and innowah_scaler is not None:
        fv_scaled = innowah_scaler.transform(fv.reshape(1, -1))
        prob      = innowah_model.predict_proba(fv_scaled)[0]
        risk_score = float(prob[1] * 50 + prob[2] * 100)
        risk_level = ["Normal", "Mild Risk", "High Risk"][int(np.argmax(prob))]
        method     = "ml_model"
    else:
        # ── Rule-based fallback ────────────────────────────────────────────────
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

        n = len(CLINICAL_RULES)
        raw = (len(mild_flags) * 30 + len(high_flags) * 60) / n
        # Apply 60:40 weighting: sensor 60%, cognitive 40%
        sensor_health  = float(fv[33])
        cog_health     = float(fv[34])
        risk_score     = min(100.0, (1 - sensor_health) * 0.60 * 100 +
                                    (1 - cog_health)    * 0.40 * 100)
        risk_level     = ("High Risk" if risk_score >= 50 else
                          "Mild Risk" if risk_score >= 25 else "Normal")
        method         = "rule_based_clinical"

    # ── Domain scores ─────────────────────────────────────────────────────────
    domain_scores = {}
    for domain, indices in DOMAIN_INDICES.items():
        valid = [fv[i] for i in indices if i < len(fv)]
        domain_scores[domain] = round((1.0 - float(np.mean(valid))) * 100, 1) if valid else 50.0

    # ── Feature flags ─────────────────────────────────────────────────────────
    mild_flags, high_flags = [], []
    for idx, name, normal_t, mild_t, direction in CLINICAL_RULES:
        if idx >= len(fv): continue
        val = fv[idx]
        if direction == "lower_worse":
            if val < mild_t:     high_flags.append(name)
            elif val < normal_t: mild_flags.append(name)
        else:
            if val > mild_t:     high_flags.append(name)
            elif val > normal_t: mild_flags.append(name)

    # ── Recommendation ────────────────────────────────────────────────────────
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
        "risk_score":       round(float(risk_score), 1),
        "risk_level":       risk_level,
        "sensor_score":     round(float(fv[33]) * 100, 1),
        "cognitive_score":  round(float(fv[34]) * 100, 1),
        "domain_scores":    domain_scores,
        "feature_flags":    {"mild": mild_flags, "high": high_flags},
        "recommendation":   recommendation,
        "method":           method,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING ROUTES (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/memory")
def memory_card():
    return render_template("memory.html")


@app.route("/n-back")
def n_back():
    return render_template("n-back.html")


@app.route("/questionnaire")
def questionnaire():
    return render_template("questionnaire.html")


@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/update_scores", methods=["POST"])
def update_scores():
    global cognitive_scores
    data = request.get_json(force=True)
    print("Received scores:", data)

    if "memory_score" in data:
        cognitive_scores["memory"] = data["memory_score"]
    if "nback_score" in data:
        cognitive_scores["nback"] = data["nback_score"]
    if "questionnaire" in data:
        cognitive_scores["questionnaire"] = data["questionnaire"]

    response_data = {"status": "ok"}

    # Only attempt ML prediction when both game scores are present
    if cognitive_scores["memory"] is not None and cognitive_scores["nback"] is not None:
        try:
            cognitive_scores["final"] = round(
                (cognitive_scores["memory"] + cognitive_scores["nback"]) / 2, 2
            )
            features = np.array([[
                cognitive_scores["memory"],
                cognitive_scores["nback"],
                cognitive_scores["final"]
            ]])
            prediction = model.predict(features)
            cognitive_scores["ml_prediction"] = int(prediction[0])
            response_data["mlPrediction"] = int(prediction[0])
        except Exception as e:
            logger.warning(f"[update_scores] model.predict failed: {e}. Skipping ML prediction.")
            # Still return ok so frontend Firebase save always fires

    print("Updated cognitive_scores:", cognitive_scores)
    return jsonify(response_data)


@app.route("/save_firebase_scores", methods=["POST"])
def save_firebase_scores():
    global cognitive_scores
    data   = request.get_json(force=True)
    scores = data.get("scores", {})

    if scores.get("memory")       is not None: cognitive_scores["memory"]        = scores["memory"]
    if scores.get("nback")        is not None: cognitive_scores["nback"]         = scores["nback"]
    if scores.get("questionnaire") is not None: cognitive_scores["questionnaire"] = scores["questionnaire"]
    if scores.get("mlPrediction") is not None: cognitive_scores["ml_prediction"] = scores["mlPrediction"]

    session["user_uid"]  = data.get("uid",  "")
    session["user_name"] = data.get("name", "")

    print("[Firebase sync] Scores received:", cognitive_scores)
    return jsonify({"status": "ok"})


@app.route("/get_data")
def get_data():
    return jsonify({"cognitive": cognitive_scores})


@app.route("/evaluate-fold", methods=["POST"])
def evaluate_fold():
    """Evaluate whether the uploaded paper image shows a properly folded paper."""
    try:
        data       = request.get_json(force=True)
        image_data = data.get("image", "")

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Could not read image."}), 400

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150)
        lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                  threshold=80,
                                  minLineLength=img.shape[1] * 0.2,
                                  maxLineGap=20)

        fold_detected = False
        fold_score    = 0
        h, w = img.shape[:2]

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle  = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                is_horizontal = angle < 15 or angle > 165
                is_vertical   = 75 < angle < 105

                if is_horizontal and length > w * 0.3:
                    mid_y = (y1 + y2) / 2
                    if 0.2 * h < mid_y < 0.8 * h:
                        fold_score += 2
                elif is_vertical and length > h * 0.3:
                    mid_x = (x1 + x2) / 2
                    if 0.2 * w < mid_x < 0.8 * w:
                        fold_score += 2
                elif length > min(w, h) * 0.25:
                    fold_score += 1

            fold_detected = fold_score >= 2

        top_half    = gray[:h // 2, :]
        bottom_half = cv2.flip(gray[h // 2:, :], 0)
        min_h = min(top_half.shape[0], bottom_half.shape[0])
        min_w = min(top_half.shape[1], bottom_half.shape[1])
        diff  = cv2.absdiff(top_half[:min_h, :min_w], bottom_half[:min_h, :min_w])
        symmetry = 1.0 - (np.mean(diff) / 255.0)

        if symmetry > 0.6:
            fold_score   += 2
            fold_detected = True

        if fold_detected or fold_score >= 2:
            result     = "properly_folded"
            message    = "✅ The paper appears to be properly folded."
            confidence = min(fold_score * 20, 100)
        else:
            result     = "not_folded"
            message    = "⚠️ The paper does not appear to be clearly folded. Please try again."
            confidence = max(20, fold_score * 15)

        return jsonify({
            "status": "ok", "result": result, "message": message,
            "confidence": confidence, "fold_score": fold_score
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# NEW: INNOWAH HARDWARE + ML ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/hardware_data", methods=["POST"])
def receive_hardware_data():
    """
    Called by ESP32 every 10 seconds with computed sensor features.
    Stores the reading in hardware_sessions keyed by device_id.

    Expected JSON:
    {
      "device_id": "ESP32_INNOWAH_001",
      "timestamp": 12345,
      "imu":  { "gait_speed": 1.1, "stride_variability": 2.4, "turning_velocity": 130,
                "postural_sway": 2.0, "step_count": 3200, "daily_activity_min": 24.5 },
      "ppg":  { "heart_rate": 72, "spo2": 97.5, "rmssd": 35.2, "sdnn": 44.1,
                "lf_hf_ratio": 1.6, "avg_spo2": 97.2, "desat_events": 0 },
      "eeg":  { "alpha_power": 30.2, "theta_power": 14.1, "delta_power": 9.8,
                "beta_power": 19.5, "theta_alpha_ratio": 0.47, "dominant_frequency": 10.1,
                "signal_entropy": 1.85, "posterior_alpha": 29.8 },
      "temperature": { "skin_temp": 35.5, "ambient_temp": 28.0 }
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400

    device_id = data.get("device_id", "unknown")
    if not device_id or device_id == "unknown":
        return jsonify({"status": "error", "message": "device_id is required"}), 400

    reading = {
        "timestamp": datetime.utcnow().isoformat(),
        "hardware":  data
    }

    if device_id not in hardware_sessions:
        hardware_sessions[device_id] = []
    hardware_sessions[device_id].append(reading)

    # Keep last 6 readings (~1 minute of history)
    if len(hardware_sessions[device_id]) > 6:
        hardware_sessions[device_id] = hardware_sessions[device_id][-6:]

    logger.info(
        f"[ESP32] device={device_id} — "
        f"HR={data.get('ppg',{}).get('heart_rate','?')} bpm, "
        f"SpO2={data.get('ppg',{}).get('spo2','?')}%, "
        f"Alpha={data.get('eeg',{}).get('alpha_power','?')}%"
    )

    return jsonify({
        "status":          "ok",
        "device_id":       device_id,
        "timestamp":       reading["timestamp"],
        "readings_stored": len(hardware_sessions[device_id])
    }), 200


@app.route("/api/software_data", methods=["POST"])
def receive_software_data():
    """
    Called by the website after a cognitive test session completes.
    Pulls the latest ESP32 hardware reading for this device,
    builds the 35-dim feature vector, runs inference, returns risk result.

    Expected JSON (all fields optional — missing ones use defaults / game scores):
    {
      "device_id":                  "ESP32_INNOWAH_001",
      "immediate_recall":           0.85,   // 0–1
      "delayed_recall":             0.67,
      "retention_ratio":            0.79,
      "orientation_score":          0.90,
      "serial7s_score":             0.80,
      "clock_drawing_score":        0.75,
      "reaction_time_ms":           620,
      "naming_task_score":          0.85,
      "verbal_fluency_wpm":         14.5,
      "iadl_impairment_count":      1,
      "mood_filter":                0
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400

    device_id = data.get("device_id", "unknown")

    # Get latest hardware reading for this device (if any)
    hw_readings = hardware_sessions.get(device_id, [])
    latest_hw   = hw_readings[-1]["hardware"] if hw_readings else {}

    # Build feature vector
    fv = extract_innowah_features(software_data=data, hardware_data=latest_hw)

    # Run inference
    result = run_innowah_inference(fv)

    response = {
        "status":        "ok",
        "device_id":     device_id,
        "timestamp":     datetime.utcnow().isoformat(),
        "hardware_used": bool(hw_readings),
        **result
    }

    logger.info(
        f"[Inference] {device_id} → {result['risk_level']} "
        f"(score={result['risk_score']}, method={result['method']})"
    )
    return jsonify(response), 200


@app.route("/api/predict", methods=["POST"])
def innowah_predict():
    """
    Direct combined prediction endpoint — accepts hardware + software in one call.
    Useful for testing or sending a single batch payload.

    Expected JSON:
    {
      "software": { ... cognitive fields ... },
      "hardware": { "imu": {...}, "ppg": {...}, "eeg": {...}, "temperature": {...} }
    }
    """
    data     = request.get_json(force=True)
    software = data.get("software", {})
    hardware = data.get("hardware", {})

    fv     = extract_innowah_features(software_data=software, hardware_data=hardware)
    result = run_innowah_inference(fv)

    return jsonify({"status": "ok", **result}), 200


@app.route("/api/full_assessment", methods=["POST"])
def full_assessment():
    """
    Combines the original game-based ML prediction with the INNOWAH hardware
    inference into one unified assessment response.
    Returns both scores so the frontend can display a comprehensive report.

    Expected JSON:
    {
      "device_id": "ESP32_INNOWAH_001",
      "memory_score":  0.75,
      "nback_score":   0.68,
      ...any other software fields...
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400

    # ── Update game scores ─────────────────────────────────────────────────
    if "memory_score" in data: cognitive_scores["memory"] = data["memory_score"]
    if "nback_score"  in data: cognitive_scores["nback"]  = data["nback_score"]

    game_prediction = None
    if cognitive_scores["memory"] is not None and cognitive_scores["nback"] is not None:
        final = round((cognitive_scores["memory"] + cognitive_scores["nback"]) / 2, 2)
        cognitive_scores["final"] = final
        features    = np.array([[cognitive_scores["memory"], cognitive_scores["nback"], final]])
        game_pred   = int(model.predict(features)[0])
        cognitive_scores["ml_prediction"] = game_pred
        game_prediction = game_pred

    # ── INNOWAH hardware + full cognitive inference ────────────────────────
    device_id   = data.get("device_id", "unknown")
    hw_readings = hardware_sessions.get(device_id, [])
    latest_hw   = hw_readings[-1]["hardware"] if hw_readings else {}

    fv      = extract_innowah_features(software_data=data, hardware_data=latest_hw)
    innowah = run_innowah_inference(fv)

    return jsonify({
        "status":           "ok",
        "timestamp":        datetime.utcnow().isoformat(),
        "device_id":        device_id,
        "hardware_used":    bool(hw_readings),
        # Original game model output
        "game_ml_prediction": game_prediction,
        "game_scores": {
            "memory": cognitive_scores["memory"],
            "nback":  cognitive_scores["nback"],
            "final":  cognitive_scores["final"],
        },
        # INNOWAH comprehensive output
        "innowah": innowah
    }), 200


@app.route("/api/hardware_status", methods=["GET"])
def hardware_status():
    """Check which ESP32 devices are connected and their latest reading time."""
    status = {}
    for device_id, readings in hardware_sessions.items():
        latest_hw = readings[-1]["hardware"] if readings else {}
        status[device_id] = {
            "readings_count":  len(readings),
            "last_seen":       readings[-1]["timestamp"] if readings else None,
            "latest_hr":       latest_hw.get("ppg", {}).get("heart_rate"),
            "latest_spo2":     latest_hw.get("ppg", {}).get("spo2"),
            "latest_alpha":    latest_hw.get("eeg", {}).get("alpha_power"),
        }
    return jsonify({
        "status":          "ok",
        "innowah_model":   innowah_model is not None,
        "active_devices":  status,
        "timestamp":       datetime.utcnow().isoformat()
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":             "healthy",
        "innowah_model":      innowah_model is not None,
        "game_model":         model is not None,
        "active_devices":     list(hardware_sessions.keys()),
        "cognitive_scores":   cognitive_scores,
        "timestamp":          datetime.utcnow().isoformat()
    })


# ══════════════════════════════════════════════════════════════════════════════
# RENDER ML API ROUTE
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/render_predict", methods=["POST"])
def render_predict():
    """
    Receives the 14 software/cognitive features computed by the questionnaire,
    builds the full 35-dim feature vector (hardware features default to 0.5),
    POSTs to the Render ML model, and returns the prediction.
    Falls back to local rule-based inference if Render API is unreachable.

    Expected JSON from frontend:
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
      "mood_filter":            0
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400

    logger.info(f"[Render Predict] Received software features for device='{data.get('device_id','?')}'")

    # Look up hardware session by device_id (linked to user at signup)
    device_id   = data.get("device_id", "")
    hw_readings = hardware_sessions.get(device_id, [])
    latest_hw   = hw_readings[-1]["hardware"] if hw_readings else {}

    if latest_hw:
        logger.info(f"[Render Predict] Using hardware data for device='{device_id}' ({len(hw_readings)} readings)")
    else:
        logger.info(f"[Render Predict] No hardware data for device='{device_id}'. Using sensor defaults (0.5 mid-range).")

    # Build 35-dim feature vector combining software + hardware
    fv = extract_innowah_features(software_data=data, hardware_data=latest_hw)
    fv_list = fv.tolist()

    logger.info(f"[Render Predict] Feature vector ({len(fv_list)} dims): {[round(v,3) for v in fv_list[:14]]}... (showing first 14)")

    # ── Try Render ML API ──────────────────────────────────────────────
    render_result = None
    try:
        resp = requests.post(
            RENDER_API_URL,
            json={"features": fv_list},
            timeout=15
        )
        if resp.status_code == 200:
            render_result = resp.json()
            logger.info(f"[Render Predict] Render API response: {render_result}")
        else:
            logger.warning(f"[Render Predict] Render API returned {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"[Render Predict] Render API unreachable: {e}. Using local fallback.")

    # ── Fallback: local inference ──────────────────────────────────────
    local_result = run_innowah_inference(fv)

    return jsonify({
        "status":          "ok",
        "render_result":   render_result,
        "local_result":    local_result,
        "feature_vector":  fv_list,
        "used_render":     render_result is not None,
        "hardware_used":   bool(latest_hw),
        "device_id":       device_id,
        "timestamp":       datetime.utcnow().isoformat()
    }), 200


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)