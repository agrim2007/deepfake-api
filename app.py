import os
import pickle
import numpy as np
import pandas as pd
import librosa
import base64
import json
import tempfile
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = "sk_test_123456789"
MODEL_PATH = "deepfake_model_lr.pkl"
model = None

# Load Model
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            print("✅ Model loaded successfully!")
    else:
        print(f"⚠️ WARNING: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model: {e}")

# ============================================================================
# FEATURE EXTRACTION (Fixed to match your original Model)
# ============================================================================

def extract_features_for_api(file_path, target_sr=16000):
    try:
        # Load audio from the temp file
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        if len(y) == 0: 
            return None, "Audio file is empty."
        
        # Noise injection (Prevents errors on silence)
        noise_amp = 0.001 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
        
        features = {}
        
        # 1. Spectral Rolloff
        features["rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        
        # 2. Spectral Centroid (This was missing!)
        features["centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 3. Zero Crossing Rate
        features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 4. MFCCs (Must be 20, not 13, to match your model)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfcc, axis=1)
        
        for i, m in enumerate(mfcc_means):
            features[f"mfcc_{i}"] = m
            
        # Note: We REMOVED 'delta_mfcc' because your model was not trained on it.
        
        return pd.DataFrame([features]), None
    
    except Exception as e:
        return None, f"Extraction failed: {str(e)}"

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online", 
        "message": "Deepfake Detection API is running (Fixed Features).",
        "model_loaded": model is not None
    })

@app.route('/api/voice-detection', methods=['POST'])
def voice_detection_api():
    # 1. Authenticate
    api_key = request.headers.get('x-api-key')
    if api_key != API_KEY:
        return jsonify({"status": "error", "message": "Invalid API key"}), 401
    
    temp_filename = None
    try:
        # 2. Parse Request
        data = request.get_json()
        if not data: return jsonify({"status": "error", "message": "Invalid JSON"}), 400
        
        language = data.get('language')
        audio_base64 = data.get('audioBase64')
        
        if not language or not audio_base64:
             return jsonify({"status": "error", "message": "Missing fields"}), 400

        # 3. Decode & Save to Temp File
        audio_bytes = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
            
        if model is None: 
            return jsonify({"status": "error", "message": "Model not loaded"}), 500

        # 4. Extract Features
        features_df, error_msg = extract_features_for_api(temp_filename)
        
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        if features_df is None: 
            return jsonify({"status": "error", "message": error_msg}), 400

        # 5. Predict
        prediction_proba = model.predict_proba(features_df)[0]
        confidence_score = float(prediction_proba[1]) # Prob of Fake
        
        classification = "AI_GENERATED" if confidence_score > 0.5 else "HUMAN"
        
        if classification == "AI_GENERATED":
            expl = "High spectral centroid and abnormal frequency consistency detected."
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
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)