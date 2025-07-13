from model import create_model  # Only if you moved the model to another file
from load_data import train_data, test_data  # Only if load_data is a separate file

# Create model
model = create_model()

# Train the model
model.fit(train_data, epochs=10, validation_data=test_data)

# Save the model
model.save("emotion_model.h5")
print("Model saved successfully.")
