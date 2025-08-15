# YOLOv11 Military Object Detection Streamlit App

## Overview
This is a Streamlit web application that uses a pre-trained YOLOv11 model for real-time object detection on images and videos. The model was trained on the `Military-2` dataset for 100 epochs with an image size of 640x640. The app allows users to upload images (JPG, PNG) or videos (MP4) and displays detection results with bounding boxes and labels.

## Features
- Upload images or videos for object detection.
- Display annotated results with bounding boxes and class labels.
- Sidebar with model information and usage instructions.
- Supports both image and video processing using the Ultralytics YOLO library.
- Clean and user-friendly interface built with Streamlit.

## Requirements
- Python 3.8+
- Streamlit
- Ultralytics YOLO
- OpenCV
- Pillow
- NumPy

## Installation
1. **Clone the repository** (if applicable) or create a project directory.
2. **Install dependencies**:
   ```bash
   pip install streamlit ultralytics opencv-python pillow numpy
