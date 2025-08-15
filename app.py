import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

# Set page configuration
st.set_page_config(page_title="YOLOv11 Military Object and vehicle Detection", layout="wide")

# Load the trained YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Replace with path to your trained model weights

model = load_model()

# Function to process image
def process_image(image):
    # Convert PIL image to numpy array
    img = np.array(image)
    # Perform inference
    results = model(img)
    # Get annotated image
    annotated_img = results[0].plot()
    return annotated_img

# Function to process video
def process_video(video_path):
    # Create temporary file for output
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Perform inference
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    return output_path

# Streamlit UI
st.title("YOLOv11 military Object and Vehicle Detection App")
st.write("Upload an image or video to perform object detection using your trained YOLOv11 model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if file_type in ["image/jpeg", "image/png"]:
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Processing image..."):
            annotated_img = process_image(image)
            st.image(annotated_img, caption="Detection Results", use_column_width=True)
    
    elif file_type == "video/mp4":
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        with st.spinner("Processing video..."):
            output_video_path = process_video(tfile.name)
            
            # Display input and output videos
            col1, col2 = st.columns(2)
            with col1:
                st.video(tfile.name)
                st.caption("Uploaded Video")
            with col2:
                st.video(output_video_path)
                st.caption("Detection Results")
            
            # Clean up temporary files
            os.remove(tfile.name)
            os.remove(output_video_path)
else:
    st.info("Please upload an image or video file to proceed.")

# Display model information
st.sidebar.header("Model Information")
st.sidebar.write("Model: YOLOv11")
st.sidebar.write("Trained on: Military-2 dataset")
st.sidebar.write("Image size: 640x640")
st.sidebar.write("Epochs: 100")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload an image (JPG, JPEG, PNG) or video (MP4).
2. Wait for the model to process the input.
3. View the detection results with bounding boxes and labels.
""")
