import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("dr_mobilenetv2.h5")

# Define class names
classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

st.title("🩺 Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg","png"])
if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.expand_dims(np.array(image)/255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    st.write("Prediction:", classes[np.argmax(prediction)])