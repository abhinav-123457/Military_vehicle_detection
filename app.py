import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Set page configuration
st.set_page_config(page_title="YOLOv11 Object Detection", layout="wide")

# Load the trained YOLO model
@st.cache_resource
def load_model(model_path="best.pt"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise

# Initialize model
try:
    model = load_model()  # Load model with default path "best.pt"
except Exception as e:
    st.error(f"Model initialization failed: {str(e)}")
    st.stop()

# Function to process image
def process_image(image):
    try:
        img = np.array(image)
        results = model(img)
        if not results[0].boxes:
            st.warning("No detections found in the image.")
        annotated_img = results[0].plot()
        return annotated_img
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None

# Video processor for real-time webcam detection
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        try:
            self.model = load_model()  # Use cached model
        except Exception as e:
            st.error(f"Failed to initialize video processor: {str(e)}")
            raise

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 640))  # Resize to match training resolution
            results = self.model(img)
            if results[0].boxes:
                st.write("Detections found in webcam frame!")
            annotated_img = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        except Exception as e:
            st.error(f"Webcam processing failed: {str(e)}")
            return frame  # Return original frame if processing fails

# Streamlit UI
st.title("YOLOv11 Object Detection App")
st.write("Upload an image or use your webcam for real-time object detection using the trained YOLOv11 model.")

# Option to choose between image upload and webcam
option = st.radio("Choose input method:", ("Upload Image", "Real-Time Webcam"))

if option == "Upload Image":
    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Processing image..."):
                annotated_img = process_image(image)
                if annotated_img is not None:
                    st.image(annotated_img, caption="Detection Results", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to process uploaded image: {str(e)}")
    else:
        st.info("Please upload an image to proceed.")

elif option == "Real-Time Webcam":
    # Webcam streaming
    st.write("Click 'Start' to begin real-time detection. Ensure your webcam is enabled.")
    try:
        webrtc_ctx = webrtc_streamer(
            key="yolo-webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if webrtc_ctx.state.playing:
            st.write("Webcam is active. Processing real-time detections...")
        else:
            st.info("Click 'Start' to enable webcam detection.")
    except Exception as e:
        st.error(f"Webcam streaming failed: {str(e)}")

# Display model information
st.sidebar.header("Model Information")
st.sidebar.write("Model: YOLOv11")
st.sidebar.write("Trained on: Military-2 dataset")
st.sidebar.write("Image size: 640x640")
st.sidebar.write("Epochs: 100")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Select 'Upload Image' to upload a JPG, JPEG, or PNG file for detection.
2. Select 'Real-Time Webcam' and click 'Start' to use your webcam for live detection.
3. Ensure 'best.pt' model weights are in the project directory.
4. View the detection results with bounding boxes and labels.
""")
