import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json

# Load your trained model
model = load_model('mobilenet_handwritten_model.keras')

# Load class indices to map prediction to text
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    class_indices = {v: k for k, v in class_indices.items()}  # Reverse mapping

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Test your model on custom image
image_path = input(r"C:\Users\mailb\Downloads\2.png")
image = Image.open(image_path)
image.show()  # Show the image for verification

# Preprocess and make prediction
processed_image = preprocess_image(image)
prediction = model.predict(processed_image)
label = np.argmax(prediction)
predicted_text = class_indices[label]

# Display the prediction
print("Predicted Text: ", predicted_text)
