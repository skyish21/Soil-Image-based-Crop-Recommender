# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("soil_classifier.h5")
class_names = ["Alluvial soil", "Black soil", "Clay soil", "Red soil"]

# Crop recommendations
crop_map = {
    "Alluvial soil": ["Wheat", "Rice", "Sugarcane"],
    "Black soil": ["Cotton", "Soybean", "Groundnut"],
    "Clay soil": ["Rice", "Mustard"],
    "Red soil": ["Millets", "Pulses", "Potato"]
}

# Preprocessing
def preprocess(img):
    image = Image.open(img).convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# App
st.title("🌱 Soil-Based Crop Recommendation")
st.markdown("Upload an image of your soil to get crop suggestions.")

uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    image = preprocess(uploaded_file)
    prediction = model.predict(image)
    predicted_label = class_names[np.argmax(prediction)]
    
    st.success(f"🧪 Detected Soil Type: **{predicted_label}**")
    st.markdown("### 🌾 Recommended Crops:")
    for crop in crop_map[predicted_label]:
        st.markdown(f"- {crop}")
