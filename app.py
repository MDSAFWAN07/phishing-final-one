import os
import re
import json
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# =====================================================
# 🔧 PATH SETUP (important for Render)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cnn_model_path = os.path.join(BASE_DIR, "cnn_lstm_feature_extractor.h5")
xgb_model_path = os.path.join(BASE_DIR, "xgboost_classifier_v4.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
features_path = os.path.join(BASE_DIR, "feature_names.json")

# =====================================================
# 🚀 LOAD MODELS AND UTILITIES
# =====================================================
print("🔁 Loading models...")

cnn_model = load_model(cnn_model_path)
xgb_model = joblib.load(xgb_model_path)
scaler = joblib.load(scaler_path)

with open(features_path, "r") as f:
    feature_names = json.load(f)

print("✅ Models Loaded Successfully")
print("Expected Features:", len(feature_names), feature_names)

# =====================================================
# 🌐 FLASK APP SETUP
# =====================================================
app = Flask(__name__)

# =====================================================
# 🧠 FEATURE EXTRACTION FUNCTION (11 Features)
# =====================================================
def extract_features(url):
    url = str(url)
    features = [
        len(url),                      # length_url
        url.count('.'),                # nb_dots
        url.count('-'),                # nb_hyphens
        1 if '@' in url else 0,        # nb_at
        url.count('/'),                # nb_slash
        1 if 'www' in url else 0,      # nb_www
        1 if '.com' in url else 0,     # nb_com
        1 if url.startswith('https') else 0,           # has_https
        len(url.split('.')[-1]) if '.' in url else 0,  # tld_length
        1 if re.match(r'http[s]?://\d', url) else 0,   # contains_ip
        url.count('.') - 1             # subdomain_count
    ]
    return np.array(features).reshape(1, -1)

# =====================================================
# 🔮 PREDICTION FUNCTION
# =====================================================
def predict_url(url):
    feats = extract_features(url)
    scaled_feats = scaler.transform(feats)
    scaled_feats_r = scaled_feats.reshape((scaled_feats.shape[0], scaled_feats.shape[1], 1))
    
    deep_feats = cnn_model.predict(scaled_feats_r, verbose=0)
    deep_feats = np.nan_to_num(deep_feats)

    prob = xgb_model.predict_proba(deep_feats)[0][1]
    label = "Phishing" if prob > 0.5 else "Legitimate"
    
    return {"url": url, "prediction": label, "confidence": float(prob)}

# =====================================================
# 🌍 API ROUTES
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ Phishing Detection API is running!",
        "model": "Hybrid CNN + LSTM + XGBoost",
        "expected_features": feature_names
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field"}), 400
    
    url = data["url"]
    try:
        result = predict_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================
# 🚀 RUN FLASK APP (Dynamic Port for Render)
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
