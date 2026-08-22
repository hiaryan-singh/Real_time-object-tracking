# Real-Time Object Detection and Multi-Object Tracking (MOT)

## 📌 Project Overview
This project implements an end-to-end real-time computer vision pipeline for object detection and multi-object tracking using OpenCV, YOLOv8, and ByteTrack.

## 🚀 Key Features
- **Real-Time Video Input:** Supports live webcam stream and pre-recorded video files.
- **Deep Learning Object Detection:** Powered by pre-trained YOLOv8 Nano (`yolov8n.pt`) on the COCO dataset (80 object classes).
- **Persistent Multi-Object Tracking:** Assigns unique, persistent tracking IDs (`ID: 1`, `ID: 2`) across frames using ByteTrack.
- **Motion Trajectory Visualization:** Dynamically renders the historical motion path behind active tracked objects.
- **Real-Time Performance:** Displays live frames-per-second (FPS) metrics and exports processed video streams to disk.

## 🛠️ Tech Stack
- Python 3
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)
- NumPy

## ⚙️ Installation & Usage

### 1. Set Up Environment
```bash
python -m venv cv_tracker_env
.\cv_tracker_env\Scripts\activate
pip install -r requirements.txt