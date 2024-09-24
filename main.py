import streamlit as st
from PIL import Image
import torch
import os

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/path/to/best.pt', force_reload=True)

# Streamlit layout according to your wireframe
st.title("Cervical Cancer Detection")

# Inserting three images about Cervical Cancer
col1, col2, col3 = st.columns(3)
with col1:
    img1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
    if img1:
        image1 = Image.open(img1)
        st.image(image1, caption='Cervical Cancer Image 1')

with col2:
    img2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])
    if img2:
        image2 = Image.open(img2)
        st.image(image2, caption='Cervical Cancer Image 2')

with col3:
    img3 = st.file_uploader("Upload Image 3", type=["jpg", "png", "jpeg"])
    if img3:
        image3 = Image.open(img3)
        st.image(image3, caption='Cervical Cancer Image 3')

# A brief about Cervical Cancer
st.subheader("A brief about Cervical Cancer")
st.write("""
Cervical cancer is a type of cancer that occurs in the cells of the cervix — the lower part of the uterus that connects to the vagina.
Various strains of the human papillomavirus (HPV), a sexually transmitted infection, play a role in causing most cervical cancer.
""")

# File uploader to upload the input file for the model
uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Uploaded Image for Prediction', use_column_width=True)

    # Save uploaded image temporarily
    with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform detection using YOLO model
    results = model(os.path.join("tempDir", uploaded_file.name))

    # Display results
    st.image(results.render()[0], caption="Model Prediction Results", use_column_width=True)

    # Print out YOLOv8 model's predictions
    st.subheader("Detected Classes:")
    for result in results.xyxy[0]:
        st.write(f"Class: {model.names[int(result[5])]}, Confidence: {result[4]:.2f}")

# Background or extra space for design as per wireframe
st.write("")
st.write("")
st.write("Background Placeholder")