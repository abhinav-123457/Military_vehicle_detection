import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import av
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# =========================
# Page & Sidebar
# =========================
st.set_page_config(page_title="YOLOv11 — Image + Camera", layout="wide")
st.title("YOLOv11 Object Detection — Image + Live Camera")

with st.sidebar:
    st.header("Model Settings")
    model_path = st.text_input("Model path (.pt)", value="best.pt")
    conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_thres = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.01)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    st.markdown("---")
    st.caption("Tip: Use **cuda** if a GPU is available for best live performance.")

# =========================
# Load Model (cached)
# =========================
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    return YOLO(path)

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Common inference kwargs
infer_kwargs = dict(conf=conf_thres, iou=iou_thres, device=None if device == "auto" else device)

# =========================
# TABS: Image | Live Camera
# =========================
tab_image, tab_camera = st.tabs(["📷 Image Detection", "🎥 Live Camera"])

# ---------- IMAGE ----------
with tab_image:
    uploaded_image = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        run_btn = st.button("Run Detection on Image", use_container_width=True)
        if run_btn:
            with st.spinner("Running YOLO on image..."):
                img_np = np.array(image)  # RGB
                # ultralytics expects BGR; convert for consistent colors in plot overlay
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                results = model(img_bgr, verbose=False, **infer_kwargs)
                annotated = results[0].plot()  # BGR with boxes
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Detections", use_column_width=True)
                # Optional: show per-class summary
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                    names = [results[0].names[c] for c in classes]
                    st.success(f"Detected: {', '.join(names)}")

# ---------- LIVE CAMERA ----------
with tab_camera:
    st.markdown("Enable your webcam to see **real-time detections**.")

    # WebRTC ICE servers (Google public STUN)
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # Video frame callback for live inference
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")  # Incoming frame (BGR)
        # Inference
        results = model(img, verbose=False, **infer_kwargs)
        annotated = results[0].plot()  # BGR with boxes/labels
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,  # keeps UI responsive
    )

# =========================
# Footer Info
# =========================
with st.expander("ℹ️ App Info"):
    st.write(
        "This app runs YOLOv11 for object detection on uploaded images and live webcam frames. "
        "Adjust thresholds in the sidebar for precision vs. recall."
    )
    st.code(
        "pip install ultralytics streamlit streamlit-webrtc av opencv-python-headless pillow",
        language="bash",
    )
