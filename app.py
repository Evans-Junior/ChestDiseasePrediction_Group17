import subprocess
subprocess.run(["pip", "install", "opencv-python"])
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model_path = "tuberculosisModel.h5"
model = load_model(model_path)

# Function to preprocess the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app
st.title("Tuberculosis Prediction App")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(uploaded_file)

    # Make prediction
    prediction = model.predict(processed_image)

    # Display the prediction
    st.subheader("Prediction:")
    if prediction[0][0] > 0.5:
        st.write("Normal")
    else:
        st.write("Tuberculosis")
