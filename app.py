from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load the pre-trained model
model = load_model(r"app/static/model.h5") #r"static\model.h5
scaler = joblib.load(r"app/static/fitted_scaler.pkl") #r"static\fitted_scaler.pkl
encoder = joblib.load(r"app/static/fitted_encoder.pkl") #r"static\fitted_encoder.pkl

# Function to extract features from audio data
def extract_features(data, sample_rate):

    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

# Function to preprocess the input audio data
def preprocess_audio(data, sample_rate):
    features = extract_features(data, sample_rate)
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    features_scaled = np.expand_dims(features_scaled, axis=2)
    return features_scaled

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling audio file upload and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Load and preprocess the audio file
        data, sample_rate = librosa.load(file, duration=2.5, offset=0.6)
        processed_data = preprocess_audio(data, sample_rate)

        # Make prediction
        prediction = model.predict(processed_data)
        predicted_label = encoder.inverse_transform(prediction)

        return jsonify({'emotion': predicted_label[0][0]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
