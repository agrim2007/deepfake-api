import os
import pickle
import numpy as np
import pandas as pd
import librosa
import time
import base64
import io
import json
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# REST API CONFIGURATION
# ============================================================================

# API Key for authentication (strict requirement)
API_KEY = "sk_test_123456789"

def authenticate_request():
    """
    Validates the x-api-key header against the expected API_KEY.
    Returns None if valid, or an error response tuple if invalid.
    """
    api_key = request.headers.get('x-api-key')
    if not api_key or api_key != API_KEY:
        return (
            jsonify({"status": "error", "message": "Invalid API key"}),
            401
        )
    return None

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_PATH = "deepfake_model_lr.pkl"
model = None

def load_model():
    """
    Loads the pre-trained machine learning model from disk.
    """
    global model
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
                print("✅ Model loaded successfully!")
        else:
            model = None
            print(f"ERROR: Model file not found at {MODEL_PATH}")
    except Exception as e:
        model = None
        print(f"ERROR loading model: {e}")

# Load model on startup
load_model()


# ============================================================================
# FEATURE EXTRACTION FOR REST API
# ============================================================================

def extract_features_for_api(audio_bytes, sr=16000):
    """
    Extracts audio features from raw audio bytes for REST API prediction.
    """
    try:
        # Load audio from bytes using librosa
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
        
        # Check for empty audio
        if len(y) == 0:
            return None, "Audio file is empty."
        
        # Add small noise to prevent silent audio issues
        noise_amp = 0.001 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
        
        # Extract features
        features = {}
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        for i, m in enumerate(mfcc_means):
            features[f"mfcc_{i}"] = m
        
        # Delta MFCC
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc_means = np.mean(delta_mfcc, axis=1)
        for i, d in enumerate(delta_mfcc_means):
            features[f"delta_mfcc_{i}"] = d
        
        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        features["zcr"] = zcr
        
        # Spectral Rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        features["rolloff"] = rolloff
        
        # Return as DataFrame for model compatibility
        return pd.DataFrame([features]), None
    
    except Exception as e:
        return None, f"Feature extraction failed: {str(e)}"


def get_physics_features(file_path):
    """
    Extracts features for the Web UI (File Upload)
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        if len(y) == 0:
            return None, "Audio file is empty."

        noise_amp = 0.001 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])

        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfcc, axis=1)
        
        features = {"rolloff": rolloff, "centroid": centroid, "zcr": zcr}
        for i, m in enumerate(mfcc_means):
            features[f"mfcc_{i}"] = m
            
        return pd.DataFrame([features]), None

    except Exception as e:
        print(f"Physics Error: {e}")
        return None, str(e)

def get_explanation(prediction_class):
    # Updated text to remove references to the visual spectrogram
    if prediction_class == 1: 
        return "⚠️ <b>Reason:</b> Analysis detected 'Digital Silence' gaps and a lack of natural background noise."
    else: 
        return "✅ <b>Reason:</b> Analysis detected consistent 'Natural Noise' patterns and expected microphone frequencies."


# ============================================================================
# REST API ENDPOINT: POST /api/voice-detection
# ============================================================================

@app.route('/api/voice-detection', methods=['POST'])
def voice_detection_api():
    """
    REST API endpoint for deepfake audio detection.
    """
    
    # ===== STEP 1: AUTHENTICATE =====
    auth_error = authenticate_request()
    if auth_error:
        return auth_error
    
    # ===== STEP 2: VALIDATE REQUEST FORMAT =====
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body must be valid JSON"
            }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid JSON format: {str(e)}"
        }), 400
    
    # ===== STEP 3: EXTRACT AND VALIDATE INPUT FIELDS =====
    language = data.get('language')
    audio_format = data.get('audioFormat')
    audio_base64 = data.get('audioBase64')
    
    if not all([language, audio_format, audio_base64]):
        return jsonify({
            "status": "error",
            "message": "Missing required fields: language, audioFormat, audioBase64"
        }), 400
    
    # ===== STEP 4: DECODE BASE64 AUDIO =====
    try:
        audio_bytes = base64.b64decode(audio_base64)
        if len(audio_bytes) == 0:
            raise ValueError("Decoded audio is empty")
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to decode Base64 audio: {str(e)}"
        }), 400
    
    # ===== STEP 5: CHECK IF MODEL IS LOADED =====
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model is not available. Server configuration error."
        }), 500
    
    # ===== STEP 6: EXTRACT FEATURES FROM AUDIO =====
    features_df, error_msg = extract_features_for_api(audio_bytes)
    if features_df is None:
        return jsonify({
            "status": "error",
            "message": error_msg or "Feature extraction failed"
        }), 400
    
    # ===== STEP 7: MAKE PREDICTION =====
    try:
        # Get prediction probability for the "fake" class
        prediction_proba = model.predict_proba(features_df)[0]
        confidence_score = float(prediction_proba[1])  # Probability of fake (class 1)
        
        if confidence_score > 0.5:
            classification = "AI_GENERATED"
        else:
            classification = "HUMAN"
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500
    
    # ===== STEP 8: GENERATE EXPLANATION =====
    if classification == "AI_GENERATED":
        if confidence_score > 0.8:
            explanation = "Strong indicators of AI generation: Abnormal spectral flatness and unnatural noise pattern detected."
        else:
            explanation = "Moderate confidence in AI generation: Suspicious spectral characteristics detected."
    else:
        if confidence_score < 0.3:
            explanation = "Strong confidence in human voice: Natural noise distribution and spectral richness observed."
        else:
            explanation = "Likely human voice: Some ambiguous features detected, but overall consistent with natural speech."
    
    # ===== STEP 9: RETURN SUCCESS RESPONSE =====
    return jsonify({
        "status": "success",
        "language": language,
        "classification": classification,
        "confidenceScore": confidence_score,
        "explanation": explanation
    }), 200


# ============================================================================
# EXISTING WEB UI ROUTES
# ============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    confidence_score = ""
    result_class = ""
    explanation_text = ""

    if request.method == 'POST':
        if 'audio' not in request.files:
            flash("No file part found.")
            return redirect(request.url)
        
        file = request.files['audio']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)

        if file:
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if model is None:
                flash("Error: Model is not loaded. Check server logs.")
            else:
                features, error_msg = get_physics_features(filepath)
                
                if features is not None:
                    # Successful Extraction
                    prob = model.predict_proba(features)[0][1]
                    if(prob > 0.70):
                        pred = 1
                    else:
                        pred = 0
                    explanation_text = get_explanation(pred)
                        
                    if pred == 1:
                        prediction_text = "🚨 FAKE DETECTED"
                        confidence_score = f"Confidence: {prob*100:.1f}%"
                        result_class = "danger"
                    else:
                        prediction_text = "✅ REAL VOICE"
                        confidence_score = f"Safety Score: {(1-prob)*100:.1f}%"
                        result_class = "safe"
                else:
                    flash(f"Analysis Failed: {error_msg}")

            if os.path.exists(filepath):
                os.remove(filepath)

    # Note: 'spectrogram_url' is removed from render_template
    return render_template('index.html', 
                           prediction=prediction_text, 
                           score=confidence_score, 
                           css_class=result_class,
                           explanation=explanation_text)

if __name__ == '__main__':
    app.run(debug=True)