import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.h5")  # Change to your model file
    return model

model = load_model()

# Define class labels (Updated)
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight"]  # Updated class labels

def predict(image):
    img = image.resize((224, 224))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Streamlit UI
st.title("🌿 Plant Disease Detection System")
st.write("Upload an image of a plant leaf to detect diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing... 🧪"):
            label, confidence = predict(image)
            st.success(f"Prediction: *{label}*")
            st.info(f"Confidence: {confidence:.2%}")
