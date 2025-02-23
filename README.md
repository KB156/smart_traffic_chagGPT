# Smart Traffic Perception

## Overview

Smart Traffic Perception is a deep learning-powered traffic monitoring system designed for real-time detection, tracking, and analysis of vehicles and pedestrians in video streams. The system leverages a dual-model YOLO approach combined with DeepSORT tracking and a web-based dashboard for visualizing traffic patterns and generating structured insights for city planning.

## Features

### **1. Advanced Object Detection with YOLOv8**

- **YOLOv8n (Nano Model)**: Lightweight, high-speed detection on every frame for real-time processing.
- **YOLOv8m (Medium Model)**: More detailed analysis, running on every 5th frame for improved accuracy.
- **IoU Filtering**: Reduces duplicate detections to enhance computational efficiency and precision.

### **2. Real-Time Object Tracking with DeepSORT**

- Maintains unique object identities across frames.
- Assigns unique tracking IDs to detected objects for continuity.
- Handles object occlusion and re-identification effectively.

### **3. Optimized Performance with Multi-Threading**

- Implements threading for parallel video frame processing, reducing latency and improving real-time performance.
- Ensures smooth operation even with high-resolution videos.

### **4. Class-Specific Labeling**

- Preserves original object class names instead of generic labels (e.g., "car" instead of "vehicle").
- Facilitates detailed traffic analysis by distinguishing between different object types.

### **5. Structured Data Export & Analysis**

- Saves detected objects, timestamps, and tracking data into **CSV format** for city planning analysis.
- Enables long-term traffic pattern monitoring for smarter infrastructure decisions.

### **6. Web-Based Dashboard for Visualization**

- **Live visualization** of detected objects with tracking overlays.
- **Graphical analytics** including real-time vehicle and pedestrian count graphs.
- **User-uploaded video processing** to analyze any traffic footage.
- **Download processed results** including CSV traffic data and processed video.

## Architecture

### **1. Video Processing Pipeline**

- Extracts frames from the input video.
- **YOLOv8n** runs on every frame for real-time speed.
- **YOLOv8m** runs every 5th frame for a more detailed analysis.
- IoU filtering removes redundant detections for efficiency.
- DeepSORT assigns unique tracking IDs and tracks objects across frames.

### **2. Data Structuring & Export**

- Categorizes detected objects and assigns timestamps.
- Saves data into structured CSV format for offline analysis.
- Provides object tracking history for long-term traffic pattern evaluation.

### **3. Web Interface & Visualization**

- Flask-based web application for real-time monitoring.
- Upload videos or provide live stream links for analysis.
- View detection overlays and download structured traffic data.
- Graphs update dynamically, showing vehicle and pedestrian flow over time.

## Repository Structure

```sh
smart_traffic_perception/
│-- weights/                  # YOLO model weights (.pt files)
│-- static/                   # Static assets (CSS, JavaScript, images)
│-- templates/                # HTML templates for the web dashboard
│   ├── upload.html           # File upload interface
│-- uploads/                  # Stores uploaded videos for processing
│-- app.py                    # Main Flask application
│-- README.md                 # Project documentation
│-- requirements.txt           # Dependencies and package requirements
│-- yolov8m.pt                 # YOLOv8m model weight (detailed detection)
│-- yolov8n.pt                 # YOLOv8n model weight (fast detection)
```

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/KB156/smart_traffic_perception_chadGPT.git
cd smart_traffic_perception_chadGPT
```

### 2. Create a Virtual Environment (Recommended)

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Download and Install Model Weights

The model weights are required for YOLO object detection. Download them from Google Drive and place them in the `weights/` directory.

#### **Option 1: Manually Download from Google Drive**

1. Download weights from the following links:
   - [YOLOv8m Weights](https://drive.google.com/file/d/1zyd14BoA5tEVSBRDsn7YlNmAaAbHCo2O/view?usp=sharing)
   - [YOLOv8n Weights](https://drive.google.com/file/d/1Ed-eSd1H8xAxQ0ab94VyhSdYRRDWz4HK/view?usp=sharing)
2. Create a `weights` directory inside the project:
   ```sh
   mkdir -p weights
   ```
3. Move the downloaded `.pt` files into the `weights/` directory.

#### **Option 2: Download via Command Line**

Use `wget` or `Invoke-WebRequest` to download weights directly:

```sh
wget -P weights <your-google-drive-direct-link>
```

Or, on Windows PowerShell:

```powershell
Invoke-WebRequest -Uri "https://drive.google.com/file/d/1zyd14BoA5tEVSBRDsn7YlNmAaAbHCo2O/view?usp=sharing" -OutFile "weights/yolov8m.pt"
Invoke-WebRequest -Uri "https://drive.google.com/file/d/1Ed-eSd1H8xAxQ0ab94VyhSdYRRDWz4HK/view?usp=sharing" -OutFile "weights/yolov8n.pt"
```

### 5. Run the Application

```sh
python app.py
```

## Web Interface Usage

- Open a browser and navigate to `http://127.0.0.1:5000`
- Upload a video or provide a YouTube link for analysis.
- View live detections with tracking overlays.
- Download processed results, including structured CSV traffic data.

## Contributing

If you want to contribute, feel free to fork the repository, make changes, and submit a pull request!

created by team chadGPT

