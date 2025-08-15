import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stButton>button[kind="secondary"] {
        background-color: #ff4b4b;
    }
    .stButton>button[kind="secondary"]:hover {
        background-color: #e04343;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab-panel"] {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        background-color: #0e1117;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #ddd;
    }
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(page_title="YOLOv11 Object Detection", layout="wide")

# Load the trained YOLO model
@st.cache_resource
def load_model(model_path="best.pt"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure 'best.pt' is in the project directory.")
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
def process_image(image, conf_threshold=0.25):
    try:
        img = np.array(image)
        results = model(img, conf=conf_threshold)
        if not results[0].boxes:
            st.warning("No detections found in the image.")
            return None
        annotated_img = results[0].plot()
        return annotated_img
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None

# Video processor for real-time webcam detection
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, conf_threshold):
        try:
            self.model = load_model()  # Use cached model
            self.conf_threshold = conf_threshold
        except Exception as e:
            st.error(f"Failed to initialize video processor: {str(e)}")
            raise

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 640))  # Resize to match training resolution
            results = self.model(img, conf=self.conf_threshold)
            if results[0].boxes:
                st.write("Detections found in webcam frame!")
            annotated_img = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        except Exception as e:
            st.error(f"Webcam processing failed: {str(e)}")
            return frame  # Return original frame if processing fails

# Initialize session state for webcam control
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Streamlit UI
st.title("YOLOv11 Object Detection App")
st.markdown("Detect military vehicles in images or real-time webcam feed using a trained YOLOv11 model.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05,
    help="Adjust the confidence threshold for object detections."
)
st.sidebar.header("ℹ️ Model Information")
st.sidebar.markdown("""
- **Model**: YOLOv11
- **Trained on**: Military-2 dataset
- **Image size**: 640x640
- **Epochs**: 100
""")
st.sidebar.header("📝 Instructions")
st.sidebar.markdown("""
1. Use the tabs to select 'Image Upload' or 'Real-Time Webcam'.
2. For images, upload a JPG, JPEG, or PNG file (max 10MB).
3. For webcam, click 'Start' to begin and 'Stop' to end detection.
4. Adjust the confidence threshold in the settings.
5. View detection results with bounding boxes and labels.
""")

# Tabs for input methods
tab1, tab2 = st.tabs(["📷 Image Upload", "🎥 Real-Time Webcam"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image file (max 10MB)."
    )
    
    if uploaded_file is not None:
        # Validate file size (max 10MB)
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 10:
            st.error("File size exceeds 10MB. Please upload a smaller image.")
        else:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with st.spinner("Processing image..."):
                    progress_bar = st.progress(0)
                    annotated_img = process_image(image, conf_threshold)
                    progress_bar.progress(100)
                    if annotated_img is not None:
                        st.image(annotated_img, caption="Detection Results", use_column_width=True)
                        st.success("Image processed successfully!")
            except Exception as e:
                st.error(f"Failed to process uploaded image: {str(e)}")
    else:
        st.info("Please upload an image to proceed.")

with tab2:
    st.subheader("Real-Time Webcam Detection")
    st.markdown("Click 'Start' to begin real-time detection or 'Stop' to end it. Ensure your webcam is enabled.")
    
    # Webcam control buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Webcam", key="start_webcam"):
            st.session_state.webcam_active = True
    with col2:
        if st.button("Stop Webcam", key="stop_webcam", type="secondary"):
            st.session_state.webcam_active = False
    
    # Render webcam stream only if active
    if st.session_state.webcam_active:
        try:
            webrtc_ctx = webrtc_streamer(
                key="yolo-webcam",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: YOLOVideoProcessor(conf_threshold),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if webrtc_ctx.state.playing:
                st.success("Webcam is active. Processing real-time detections...")
            else:
                st.info("Webcam stream initialized. Waiting for camera access...")
        except Exception as e:
            st.error(f"Webcam streaming failed: {str(e)}")
            st.session_state.webcam_active = False
    else:
        st.info("Click 'Start Webcam' to enable detection.")
