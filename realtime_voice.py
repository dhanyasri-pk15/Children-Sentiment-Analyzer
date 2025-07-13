import sounddevice as sd
import numpy as np
import librosa
import joblib
import scipy.io.wavfile as wav
import os

# Load the trained model and label encoder
model = joblib.load("voice_emotion_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Audio recording parameters
duration = 5  # seconds
fs = 22050  # sampling rate

print("Recording...")

# Record audio
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Recording complete.")

# Save temporary WAV file
temp_wav = "temp.wav"
wav.write(temp_wav, fs, (recording * 32767).astype(np.int16))

# Extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

features = extract_features(temp_wav).reshape(1, -1)

# Predict emotion
prediction = model.predict(features)
emotion = encoder.inverse_transform(prediction)

print(f"Predicted Emotion: {emotion[0]}")

# Cleanup
os.remove(temp_wav)
