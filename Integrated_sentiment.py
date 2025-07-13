from predict_emotion import predict_text_emotion
from realtime_voice import detect_voice_emotion
from realtime_emotion import detect_video_emotion

def main():
    print("\n🎭 Children Sentiment Analyzer 🎭")
    print("Select Input Type:")
    print("1. Text")
    print("2. Voice")
    print("3. Video")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        text = input("Enter text to analyze: ")
        emotion = predict_text_emotion(text)
        print(f"\n📝 Text Emotion: {emotion}")

    elif choice == "2":
        print("\n🎙️ Recording audio...")
        emotion = detect_voice_emotion()
        print(f"\n🔊 Voice Emotion: {emotion}")

    elif choice == "3":
        print("\n📹 Launching webcam for real-time video emotion detection...")
        emotion = detect_video_emotion()
        print(f"\n👁️ Video Emotion: {emotion}")

    else:
        print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
