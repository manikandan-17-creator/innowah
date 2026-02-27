"""
INNOWAH 2026 - ML Inference Engine
Loads a trained model and runs predictions on the 35-feature vector.

Falls back to a rule-based clinical engine if no trained model exists yet.
This lets the system work immediately and improve with data over time.
"""

import numpy as np
import os
import joblib
import logging
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class InnowahInferenceEngine:

    # ─── Clinical risk thresholds (from INNOWAH_2026_Parameters.pdf) ─────
    # Each tuple: (feature_index, name, normal_max, mild_max, direction)
    # direction: "higher_worse" or "lower_worse"
    CLINICAL_RULES = [
        # IMU
        (14, "gait_speed",          0.57, 0.71, "lower_worse"),   # 1.0 normalized to 0.625
        (15, "stride_variability",  0.70, 0.50, "higher_worse"),
        (16, "turning_velocity",    0.55, 0.40, "lower_worse"),
        (17, "postural_sway",       0.70, 0.50, "higher_worse"),
        # PPG
        (18, "rmssd",               0.31, 0.19, "lower_worse"),
        (19, "sdnn",                0.30, 0.20, "lower_worse"),
        (20, "lf_hf_ratio",         0.80, 0.60, "higher_worse"),
        (21, "spo2",                0.67, 0.47, "lower_worse"),
        # EEG
        (23, "alpha_power",         0.50, 0.40, "lower_worse"),
        (24, "theta_power",         0.60, 0.50, "higher_worse"),
        (25, "delta_power",         0.625,0.50, "higher_worse"),
        (26, "beta_power",          0.30, 0.25, "lower_worse"),
        (27, "theta_alpha_ratio",   0.467,0.567,"higher_worse"),
        (28, "dominant_frequency",  0.571,0.429,"lower_worse"),
        # Cognitive
        (0,  "immediate_recall",    0.7,  0.5,  "lower_worse"),
        (1,  "delayed_recall",      0.7,  0.5,  "lower_worse"),
        (3,  "retention_ratio",     0.7,  0.5,  "lower_worse"),
        (5,  "serial7s_score",      0.6,  0.4,  "lower_worse"),
        (9,  "naming_task",         0.6,  0.4,  "lower_worse"),
    ]

    # Domain weights for final score (60% sensor, 40% cognitive)
    DOMAIN_WEIGHTS = {
        "memory":       0.25,
        "reasoning":    0.20,
        "visuospatial": 0.15,
        "language":     0.15,
        "behavior":     0.25,
    }

    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        self.model    = None
        self.scaler   = None
        self.extractor = FeatureExtractor()
        self._load_model()

    def _load_model(self):
        """Try to load a trained sklearn model; fall back to rule-based engine."""
        model_path  = os.path.join(self.model_dir, "innowah_model.pkl")
        scaler_path = os.path.join(self.model_dir, "innowah_scaler.pkl")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model  = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Trained ML model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using rule-based engine.")
                self.model  = None
                self.scaler = None
        else:
            logger.info("No trained model found. Using rule-based clinical engine.")

    def is_loaded(self):
        return self.model is not None

    def predict(self, features: np.ndarray) -> dict:
        """Run prediction on a 35-dim feature vector."""
        fv = np.array(features, dtype=np.float32).flatten()

        if self.model is not None:
            return self._ml_predict(fv)
        else:
            return self._rule_based_predict(fv)

    def _ml_predict(self, fv: np.ndarray) -> dict:
        """ML model inference path."""
        fv_scaled = self.scaler.transform(fv.reshape(1, -1))
        prob = self.model.predict_proba(fv_scaled)[0]

        # Classes: 0=Normal, 1=Mild Risk, 2=High Risk
        risk_score = float(prob[1] * 50 + prob[2] * 100)
        risk_level = ["Normal", "Mild Risk", "High Risk"][np.argmax(prob)]

        sensor_score   = float(fv[33]) * 100
        cognitive_score = float(fv[34]) * 100

        domain_scores  = self._compute_domain_scores(fv)
        feature_flags  = self._get_feature_flags(fv)
        recommendation = self._get_recommendation(risk_level, feature_flags)

        return {
            "risk_score":      round(risk_score, 1),
            "risk_level":      risk_level,
            "domain_scores":   domain_scores,
            "sensor_score":    round(sensor_score, 1),
            "cognitive_score": round(cognitive_score, 1),
            "feature_flags":   feature_flags,
            "recommendation":  recommendation,
            "method":          "ml_model"
        }

    def _rule_based_predict(self, fv: np.ndarray) -> dict:
        """
        Rule-based clinical engine using thresholds from INNOWAH document.
        Returns a risk score 0–100 based on how many features fall into
        mild/high risk zones.
        """
        feature_names = self.extractor.get_feature_names()
        mild_flags    = []
        high_flags    = []

        for idx, name, normal_threshold, mild_threshold, direction in self.CLINICAL_RULES:
            if idx >= len(fv):
                continue
            val = fv[idx]

            if direction == "lower_worse":
                if val < mild_threshold:
                    high_flags.append(name)
                elif val < normal_threshold:
                    mild_flags.append(name)
            else:  # higher_worse
                if val > mild_threshold:
                    high_flags.append(name)
                elif val > normal_threshold:
                    mild_flags.append(name)

        n_rules = len(self.CLINICAL_RULES)
        # Risk score: weighted sum of triggered rules
        risk_score = min(100.0, (len(mild_flags) * 30 + len(high_flags) * 60) / n_rules)

        if risk_score >= 50:
            risk_level = "High Risk"
        elif risk_score >= 25:
            risk_level = "Mild Risk"
        else:
            risk_level = "Normal"

        sensor_score    = float(fv[33]) * 100
        cognitive_score = float(fv[34]) * 100
        domain_scores   = self._compute_domain_scores(fv)

        # Weighted final (60% sensor, 40% cognitive from document)
        final_risk = (1.0 - fv[33]) * 0.60 * 100 + (1.0 - fv[34]) * 0.40 * 100

        feature_flags  = {"mild": mild_flags, "high": high_flags}
        recommendation = self._get_recommendation(risk_level, feature_flags)

        return {
            "risk_score":      round(float(final_risk), 1),
            "risk_level":      risk_level,
            "domain_scores":   domain_scores,
            "sensor_score":    round(sensor_score, 1),
            "cognitive_score": round(cognitive_score, 1),
            "feature_flags":   feature_flags,
            "recommendation":  recommendation,
            "method":          "rule_based_clinical"
        }

    def _compute_domain_scores(self, fv: np.ndarray) -> dict:
        """Compute per-domain risk scores (0–100, higher = more risk)."""
        # Domain feature indices
        domains = {
            "memory":       [0, 1, 2, 3, 4, 18, 19, 23, 24, 25],
            "reasoning":    [5, 6, 7, 8, 14, 15, 16, 26],
            "visuospatial": [17, 30],
            "language":     [9, 10, 11, 28, 29],
            "behavior":     [12, 13, 20, 21, 22, 31],
        }
        scores = {}
        for domain, indices in domains.items():
            valid = [fv[i] for i in indices if i < len(fv)]
            if valid:
                # avg health (0=sick, 1=healthy) → invert → risk 0–100
                avg_health = float(np.mean(valid))
                scores[domain] = round((1.0 - avg_health) * 100, 1)
            else:
                scores[domain] = 50.0
        return scores

    def _get_feature_flags(self, fv: np.ndarray) -> dict:
        """Return lists of feature names in mild/high risk zones."""
        mild_flags = []
        high_flags = []
        for idx, name, normal_thresh, mild_thresh, direction in self.CLINICAL_RULES:
            if idx >= len(fv): continue
            val = fv[idx]
            if direction == "lower_worse":
                if val < mild_thresh:
                    high_flags.append(name)
                elif val < normal_thresh:
                    mild_flags.append(name)
            else:
                if val > mild_thresh:
                    high_flags.append(name)
                elif val > normal_thresh:
                    mild_flags.append(name)
        return {"mild": mild_flags, "high": high_flags}

    def _get_recommendation(self, risk_level: str, flags: dict) -> str:
        high = flags.get("high", [])
        mild = flags.get("mild", [])

        if risk_level == "High Risk":
            return (
                "⚠️ High risk indicators detected. Recommend neurological consultation. "
                f"Key concerns: {', '.join(high[:3]) if high else 'multiple domains'}. "
                "Schedule follow-up assessment within 2 weeks."
            )
        elif risk_level == "Mild Risk":
            return (
                "⚡ Mild risk indicators present. Monitor trends over time. "
                f"Areas of concern: {', '.join(mild[:3]) if mild else 'minor fluctuations'}. "
                "Repeat assessment in 30 days."
            )
        else:
            return "✅ Parameters within normal range. Continue routine monitoring monthly."
