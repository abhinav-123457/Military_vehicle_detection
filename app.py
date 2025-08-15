import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from io import BytesIO

# --- Streamlit Page Config ---
st.set_page_config(page_title="YOLOv11 Object Detection", layout="wide")

# --- Load Model ---
@st.cache_resource
def load_model(model_path="best.pt"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return YOLO(model_path)

model_path = st.sidebar.text_input("Model Path", value="best.pt")
model = load_model(model_path)

# --- Image Processing ---
def process_image(image):
    img = np.array(image)
    results = model(img)
    return results[0].plot()

# --- Video Processing ---
def process_video(video_path):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress = st.progress(0)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 50 == 0:
            progress.progress(frame_count / total_frames)

    cap.release()
    out.release()
    progress.progress(1.0)

    with open(output_path, "rb") as f:
        video_bytes = f.read()
    os.remove(output_path)
    return video_bytes

# --- UI Tabs ---
st.title("YOLOv11 Object Detection")
tabs = st.tabs(["📷 Image Detection", "🎥 Video Detection"])

with tabs[0]:
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Processing image..."):
            annotated_img = process_image(image)
            st.image(annotated_img, caption="Detection Results", use_column_width=True)

with tabs[1]:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        with st.spinner("Processing video..."):
            output_video_bytes = process_video(tfile.name)
        os.remove(tfile.name)

        if output_video_bytes:
            col1, col2 = st.columns(2)
            with col1:
                st.video(uploaded_video)
                st.caption("Uploaded Video")
            with col2:
                st.video(output_video_bytes)
                st.caption("Detection Results")

# --- Sidebar ---
st.sidebar.header("Model Information")
st.sidebar.write("Model: YOLOv11")
st.sidebar.write("Image size: 640x640")
st.sidebar.write("Epochs: 100")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Select model path or keep default.
2. Upload an image/video.
3. View detection results side-by-side.
""")
