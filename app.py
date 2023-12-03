import subprocess
subprocess.run(["pip", "install", "opencv-python"])
import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained models
tuberculosis_model_path = "tuberculosisModel.h5"
pneumonia_model_path = "PneumoniaModel.h5"
tuberculosis_model = load_model(tuberculosis_model_path)
pneumonia_model = load_model(pneumonia_model_path)

# Function to preprocess the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app
page = st.sidebar.radio("Select Disease", ["Tuberculosis", "Pneumonia"])

if page == "Tuberculosis":
    st.title("Tuberculosis Prediction App")
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)

        # Make prediction
        prediction = tuberculosis_model.predict(processed_image)

        # Display the prediction
        st.subheader("Prediction:")
        if prediction[0][0] > 0.5:
            st.write("Normal")
        else:
            st.write("Tuberculosis")

if page == "Pneumonia":
    st.title("Pneumonia Prediction App")
    # Rest of the pneumonia prediction code goes here
    uploaded_file = st.file_uploader("Choose an X-ray image for Pneumonia", type=["jpg", "png","jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)

        # Make prediction for Pneumonia
        pneumonia_prediction = pneumonia_model.predict(processed_image)

        # Display the prediction
        st.subheader("Pneumonia Prediction:")
        if pneumonia_prediction[0][0] > 0.5:
            st.write("Normal")
        else:
            st.write("Pneumonia")


