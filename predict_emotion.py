import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords', quiet=True)

# Load the saved model components (assumes they are in the same directory)
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Clean text function
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

# Function to classify emotion
def classify_emotion(user_input):
    cleaned_input = clean_text(user_input)
    if not cleaned_input.strip():
        return "Input too short or contains only stopwords."
    try:
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)
        emotion = label_encoder.inverse_transform(prediction)[0]
        return emotion
    except Exception as e:
        return f"Prediction error: {e}"

# Interactive loop
if __name__ == "__main__":
    print("🔍 Emotion classifier is ready. Type 'exit' to quit.\n")
    while True:
        user_text = input("Enter text: ").strip()
        if user_text.lower() == "exit":
            print("Exiting... Goodbye!")
            break
        predicted_emotion = classify_emotion(user_text)
        print("Predicted Emotion:", predicted_emotion, "\n")
