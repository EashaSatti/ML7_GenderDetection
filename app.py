import streamlit as st
import numpy as np
from PIL import Image
from gender_model import gender_detector  
import cv2


# Streamlit app code
st.title("Gender Detection using Pretrained OpenCV DNN Models")

# File uploader for user image input
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = np.array(Image.open(uploaded_file))

    # Detect gender on the uploaded image using the imported function
    output_image = gender_detector(image)

    # Convert OpenCV BGR format to RGB format for displaying in Streamlit
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Display the result
    st.image(output_image_rgb, caption="Processed Image", use_column_width=True)
