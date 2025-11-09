from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)

# === Load Models ===
cnn_lstm_model = tf.keras.models.load_model("cnn_bilstm_feature_extractor_v4.h5")
xgb_model = joblib.load("xgboost_classifier_v4.pkl")
scaler = joblib.load("scaler.pkl")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

@app.route('/')
def home():
    return jsonify({
        "message": "🚀 CNN + BiLSTM + XGBoost Phishing URL Detector API is Live!",
        "usage": "Send a POST request to /predict with JSON: {'features': [11 feature values]}"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        input_data = np.array(data["features"]).reshape(1, -1)
        scaled_input = scaler.transform(input_data)

        # Extract deep features using CNN + BiLSTM
        deep_features = cnn_lstm_model.predict(scaled_input)
        
        # XGBoost final prediction
        prediction = xgb_model.predict(deep_features)
        label = "Phishing" if prediction[0] == 1 else "Legitimate"

        return jsonify({
            "prediction": label,
            "raw_output": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
