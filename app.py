import streamlit as st
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

# Streamlit app interface
st.title("Handwritten Text Recognition")
st.write("Upload an image of handwritten text and get the predicted text.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    label = np.argmax(prediction)
    predicted_text = class_indices[label]

    st.subheader("Predicted Text")
    st.write(predicted_text)
