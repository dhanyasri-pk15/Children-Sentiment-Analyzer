import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Set path to your dataset
data_path = "Audio_dataset\TESS Toronto emotional speech set data"
# Check if the directory exists
if not os.path.exists(data_path):
    print(f"❌ The specified path does not exist: {data_path}")
    exit()  # Exit the program if the path is invalid

# Extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Initialize lists to store features and labels
features_list = []
labels_list = []

# Traverse each emotion folder
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if not os.path.isdir(folder_path):
        continue

    # Extract emotion from folder name (e.g., OAF_angry → angry)
    try:
        emotion = folder.split('_', 1)[-1].lower()
    except IndexError:
        continue

    print(f"\nProcessing folder: {folder} | Emotion: {emotion}")

    # Process each .wav file in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            try:
                features = extract_features(file_path)
                features_list.append(features)
                labels_list.append(emotion)
                print(f"✔ Processed: {file}")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

# Convert feature and label lists to numpy arrays
X = np.array(features_list)

# Check if data was collected
if len(X) == 0:
    print("❗No features extracted. Please check your dataset path or files.")
    exit()

# Encode string labels to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(labels_list)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save model and encoder
joblib.dump(model, "voice_emotion_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("\n✅ Model and encoder saved successfully!")
