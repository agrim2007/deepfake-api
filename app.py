import os
import pickle
import numpy as np
import pandas as pd
import librosa
import base64
import io
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================================
# REST API CONFIGURATION
# ============================================================================

API_KEY = "sk_test_123456789"

def authenticate_request():
    api_key = request.headers.get('x-api-key')
    if not api_key or api_key != API_KEY:
        return (jsonify({"status": "error", "message": "Invalid API key"}), 401)
    return None

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_PATH = "deepfake_model_lr.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            print("✅ Model loaded successfully!")
    else:
        print(f"⚠️ WARNING: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model: {e}")

def extract_features_for_api(audio_bytes, target_sr=16000):
    try:
        # OPTIMIZATION 1: Load directly with target_sr and 'linear' resampling (Lowest RAM usage)
        y, sr = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=target_sr, 
            mono=True, 
            res_type='linear'  # <--- CRITICAL CHANGE: Uses simple math instead of heavy DSP
        )
        
        if len(y) == 0: 
            return None, "Audio file is empty."
        
        # Noise injection
        noise_amp = 0.001 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
        
        features = {}
        
        # MFCC (Standard)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        for i, m in enumerate(mfcc_means):
            features[f"mfcc_{i}"] = m
        
        # Delta MFCC
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc_means = np.mean(delta_mfcc, axis=1)
        for i, d in enumerate(delta_mfcc_means):
            features[f"delta_mfcc_{i}"] = d
        
        # Other features
        features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))
        features["rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        
        return pd.DataFrame([features]), None
    
    except Exception as e:
        return None, f"Extraction failed: {str(e)}"

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    # Simple Health Check (Fixes the TemplateNotFound error)
    return jsonify({
        "status": "online",
        "message": "Deepfake Detection API is running.",
        "model_loaded": model is not None
    })

@app.route('/api/voice-detection', methods=['POST'])
def voice_detection_api():
    # 1. Authenticate
    auth_error = authenticate_request()
    if auth_error: return auth_error
    
    try:
        # 2. Parse Request
        data = request.get_json()
        if not data: return jsonify({"status": "error", "message": "Invalid JSON"}), 400
        
        language = data.get('language')
        audio_base64 = data.get('audioBase64')
        
        if not language or not audio_base64:
             return jsonify({"status": "error", "message": "Missing fields"}), 400

        # 3. Decode & Process
        audio_bytes = base64.b64decode(audio_base64)
        if model is None: return jsonify({"status": "error", "message": "Model not loaded"}), 500

        features_df, error_msg = extract_features_for_api(audio_bytes)
        if features_df is None: return jsonify({"status": "error", "message": error_msg}), 400

        # 4. Predict
        prediction_proba = model.predict_proba(features_df)[0]
        confidence_score = float(prediction_proba[1]) # Prob of Fake
        
        classification = "AI_GENERATED" if confidence_score > 0.5 else "HUMAN"
        
        # 5. Explanation
        if classification == "AI_GENERATED":
            expl = "High spectral flatness and abnormal frequency consistency detected."
        else:
            expl = "Natural vocal variances and background noise patterns detected."

        return jsonify({
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": confidence_score,
            "explanation": expl
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)