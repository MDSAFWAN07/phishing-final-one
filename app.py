# app.py
import os
import re
import json
import numpy as np
import joblib
import traceback
from flask import Flask, request, jsonify

# Try to import tensorflow/keras (will raise if not installed)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    load_model = None

# -------------------------
# Paths & base dir (Render safe)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model filenames - change names here if your files have different names
CNN_MODEL_FILENAME = "cnn_lstm_feature_extractor.h5"
XGB_MODEL_FILENAME = "xgboost_classifier_v4.pkl"
SCALER_FILENAME = "scaler.pkl"
FEATURES_FILENAME = "feature_names.json"

cnn_model_path = os.path.join(BASE_DIR, CNN_MODEL_FILENAME)
xgb_model_path = os.path.join(BASE_DIR, XGB_MODEL_FILENAME)
scaler_path = os.path.join(BASE_DIR, SCALER_FILENAME)
features_path = os.path.join(BASE_DIR, FEATURES_FILENAME)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load models (with helpful errors)
# -------------------------
def safe_load_models():
    missing = []
    if not os.path.exists(cnn_model_path):
        missing.append(CNN_MODEL_FILENAME)
    if not os.path.exists(xgb_model_path):
        missing.append(XGB_MODEL_FILENAME)
    if not os.path.exists(scaler_path):
        missing.append(SCALER_FILENAME)
    if not os.path.exists(features_path):
        missing.append(FEATURES_FILENAME)

    if missing:
        raise FileNotFoundError(f"Missing files in repo root: {missing}. Please upload them and redeploy.")

    # Load scaler and xgboost (joblib)
    scaler = joblib.load(scaler_path)
    xgb = joblib.load(xgb_model_path)

    # Load CNN/LSTM model (Keras)
    if load_model is None:
        raise RuntimeError("TensorFlow / Keras not available in the environment.")
    cnn = load_model(cnn_model_path, compile=False)

    # Load feature names
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    # ensure it's a list
    if isinstance(feature_names, dict) and "expected_features" in feature_names:
        feature_names = feature_names["expected_features"]
    if not isinstance(feature_names, list):
        raise ValueError("feature_names.json must contain a JSON list of feature names")

    return cnn, xgb, scaler, feature_names

# Wrap loading so errors show in logs but app can start (will return 500 on predict)
try:
    cnn_model, xgb_model, scaler, feature_names = safe_load_models()
    print("✅ Models Loaded Successfully")
    print("Expected Features:", len(feature_names), feature_names)
except Exception as e:
    cnn_model = xgb_model = scaler = feature_names = None
    print("❌ Model loading failed at startup:", str(e))
    traceback.print_exc()

# -------------------------
# Feature extraction - returns 15 features (1 x 15 numpy array)
# Must match the order used during training
# -------------------------
def extract_features(url):
    """
    Returns 15 features (shape (1,15)) in the exact order:
    length_url, nb_dots, nb_hyphens, nb_at, nb_slash,
    nb_www, nb_com, has_https, tld_length, contains_ip,
    subdomain_count, keyword_flag, https_in_domain, length_domain, digit_count
    """
    url = str(url).strip().lower()

    # basic counts and flags
    length_url = len(url)
    nb_dots = url.count('.')
    nb_hyphens = url.count('-')
    nb_at = 1 if '@' in url else 0
    nb_slash = url.count('/')
    nb_www = 1 if 'www' in url else 0
    nb_com = 1 if '.com' in url else 0
    has_https = 1 if url.startswith('https') else 0
    tld_length = len(url.split('.')[-1]) if '.' in url else 0
    contains_ip = 1 if re.match(r'http[s]?://\d', url) else 0
    subdomain_count = max(0, url.count('.') - 1)

    # additional features used in training
    keywords = ['secure', 'login', 'account', 'bank', 'update', 'verify', 'webscr', 'signin', 'payment']
    keyword_flag = 1 if any(k in url for k in keywords) else 0

    # get domain part safely
    domain = ""
    parts = url.split('/')
    if len(parts) > 2:
        domain = parts[2]
    https_in_domain = 1 if 'https' in domain else 0
    length_domain = len(domain)
    digit_count = sum(c.isdigit() for c in url)

    features = [
        length_url, nb_dots, nb_hyphens, nb_at, nb_slash,
        nb_www, nb_com, has_https, tld_length, contains_ip,
        subdomain_count, keyword_flag, https_in_domain, length_domain, digit_count
    ]

    arr = np.array(features, dtype=float).reshape(1, -1)
    return arr

# -------------------------
# Prediction pipeline
# -------------------------
def predict_url(url):
    if scaler is None or cnn_model is None or xgb_model is None:
        raise RuntimeError("Models not loaded. Check server logs and ensure model files are in repo root.")

    feats = extract_features(url)              # shape (1,15)
    # Ensure scaler expects same number of features
    if feats.shape[1] != scaler.n_features_in_:
        raise ValueError(f"X has {feats.shape[1]} features, but StandardScaler is expecting {scaler.n_features_in_} features as input.")

    # scale
    scaled = scaler.transform(feats)           # shape (1,15)

    # reshape for CNN+LSTM: (samples, timesteps, features_per_step) -> here timesteps=15, features_per_step=1
    scaled_r = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))

    # deep model -> get deep features (embedding)
    deep_feats = cnn_model.predict(scaled_r, verbose=0)   # shape (1, N)
    deep_feats = np.nan_to_num(deep_feats)

    # XGBoost predicts on deep features
    prob = float(xgb_model.predict_proba(deep_feats)[0][1])
    label = "Phishing" if prob >= 0.5 else "Legitimate"

    return {
        "url": url,
        "prediction": label,
        "confidence": round(prob, 6)
    }

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ Phishing Detection API is running!",
        "model": "Hybrid CNN + LSTM + XGBoost",
        "expected_features": feature_names if feature_names is not None else []
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    # support both {"url":"..."} and {"u":"..."} forms
    url = data.get("url") or data.get("u") or data.get("input") or None
    if not url:
        return jsonify({"error": "Missing 'url' field in JSON body"}), 400

    try:
        result = predict_url(url)
        # optional: log to server logs
        print(f"🔍 URL: {url} -> {result['prediction']} ({result['confidence']})")
        return jsonify(result)
    except ValueError as e:
        # expected errors like feature mismatch
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# -------------------------
# Health check route (simple)
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    ok = all([cnn_model is not None, xgb_model is not None, scaler is not None])
    return jsonify({"status": "ok" if ok else "models_missing", "models_loaded": ok})

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # Use dynamic port for Render / Heroku
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
