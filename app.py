import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

# Set page configuration
st.set_page_config(page_title="YOLOv11 Object Detection", layout="wide")

# Load the trained YOLO model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return YOLO(model_path)

model = load_model()

# Function to process image
def process_image(image):
    img = np.array(image)
    results = model(img)
    annotated_img = results[0].plot()
    return annotated_img

# Function to process video
def process_video(video_path):
    st.write("Starting video processing...")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open input video.")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        st.error("Failed to initialize video writer.")
        cap.release()
        return None
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        if results[0].boxes:
            st.write(f"Detections found in frame {frame_count+1}")
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            st.write(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    st.write(f"Video processing complete. Output saved to {output_path}")
    return output_path

# Streamlit UI
st.title("YOLOv11 Object Detection App")
st.write("Upload an image or video to perform object detection using your trained YOLOv11 model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if file_type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Processing image..."):
            annotated_img = process_image(image)
            st.image(annotated_img, caption="Detection Results", use_column_width=True)
    
    elif file_type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        with st.spinner("Processing video..."):
            output_video_path = process_video(tfile.name)
            
            if output_video_path:
                col1, col2 = st.columns(2)
                with col1:
                    st.video(tfile.name)
                    st.caption("Uploaded Video")
                with col2:
                    st.video(output_video_path)
                    st.caption("Detection Results")
                os.remove(tfile.name)
                os.remove(output_video_path)
            else:
                st.error("Video processing failed. Check the logs for details.")
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
