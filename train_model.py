import pandas as pd
import re
import nltk
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

# Download stopwords from NLTK
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        stop_words = set(stopwords.words('english'))
        text = ' '.join(word for word in text.split() if word not in stop_words)
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

try:
    # Path to the uploaded dataset
    data_path = "emotion_sentimen_dataset.csv"
    print("Looking for dataset at:", data_path)

    # Load the dataset
    data = pd.read_csv(data_path)
    print("Dataset loaded successfully!\n\nColumns:", data.columns)

    # Preview first few rows to infer structure
    print("\nFirst few rows:\n", data.head())

    # Ensure necessary columns exist
    if 'text' not in data.columns or 'Emotion' not in data.columns:
        raise KeyError("Dataset must contain 'text' and 'Emotion' columns.")

    # Drop missing rows
    data.dropna(subset=['text', 'Emotion'], inplace=True)

    # Clean the comment column
    data['text'] = data['text'].apply(clean_text)

    # Encode labels
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['Emotion'])

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save model components
    model_dir = os.path.dirname(data_path)
    joblib.dump(model, os.path.join(model_dir, 'emotion_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

    print("\n✅ Model, vectorizer, and label encoder saved successfully!")

except FileNotFoundError as fnf_error:
    print(f"File error: {fnf_error}")
except KeyError as key_error:
    print(f"Column error: {key_error}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
