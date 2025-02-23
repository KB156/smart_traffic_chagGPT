from flask import Flask, Response, render_template, request, send_from_directory
from flask_socketio import SocketIO
import cv2
import torch
import pandas as pd
import os
import yt_dlp
from ultralytics import YOLO
import numpy as np

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Define directories
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# File paths
CSV_FILE_PATH = os.path.join(STATIC_FOLDER, "output.csv")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO models
device = "cuda" if torch.cuda.is_available() else "cpu"
primary_model = YOLO("yolov8n.pt").to(device)
secondary_model = YOLO("yolov8m.pt").to(device)

# Class names for YOLOv8 (COCO dataset)
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Class IDs for persons and vehicles
PERSON_VEHICLE_CLASS_IDS = {0, 1, 2, 3, 5, 7}  # Person, bicycle, car, motorcycle, bus, truck

# Class colors for bounding boxes
CLASS_COLORS = {
    0: (0, 255, 0),  # Person (green)
    1: (255, 0, 0),  # Bicycle (blue)
    2: (0, 0, 255),  # Car (red)
    3: (255, 255, 0),  # Motorcycle (cyan)
    5: (255, 0, 255),  # Bus (magenta)
    7: (0, 255, 255)  # Truck (yellow)
}

# Initialize an empty DataFrame to store object data
object_data = pd.DataFrame(columns=["Frame", "Class", "Count", "Positions"])

def download_youtube_video(youtube_url):
    """Download video from YouTube."""
    try:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': os.path.join(UPLOAD_FOLDER, '%(title)s.%(ext)s'),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            return os.path.join(UPLOAD_FOLDER, f"{info['title']}.mp4")
    except Exception as e:
        print("Error downloading YouTube video:", e)
        return None

def calculate_road_region(vehicle_boxes, frame_width, frame_height):
    """
    Calculate the road region as a trapezium based on vehicle positions.
    """
    if not vehicle_boxes:
        return None

    # Get the bottom points of all vehicle bounding boxes
    bottom_points = [((x1 + x2) // 2, y2) for (x1, y1, x2, y2, _) in vehicle_boxes]

    # Define the road region as a trapezium
    left_bottom = (0, frame_height)
    right_bottom = (frame_width, frame_height)
    left_top = (min([x for x, y in bottom_points]), min([y for x, y in bottom_points]))
    right_top = (max([x for x, y in bottom_points]), min([y for x, y in bottom_points]))

    road_polygon = np.array([left_bottom, right_bottom, right_top, left_top], dtype=np.int32)
    return road_polygon

def is_person_near_road(box, road_polygon):
    """
    Check if a person is near the road using the road polygon.
    """
    x1, y1, x2, y2 = box[:4]
    # Use the bottom center of the bounding box to check if it's near the road polygon
    bottom_center = ((x1 + x2) // 2, y2)
    return cv2.pointPolygonTest(road_polygon, bottom_center, False) >= 0

def process_frame(frame, frame_count, previous_boxes):
    global object_data  # Use the global DataFrame to store data
    vehicle_count = 0
    pedestrian_count = 0
    primary_boxes, secondary_boxes = [], []

    # Run primary model every 2nd frame
    if frame_count % 2 == 0:
        results_primary = primary_model(frame, device=device)
        primary_boxes = [
            (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), int(box.cls[0].item())) 
            for r in results_primary for box in r.boxes if int(box.cls[0].item()) in PERSON_VEHICLE_CLASS_IDS
        ]

    # Run secondary model every 5th frame
    if frame_count % 5 == 0:
        results_secondary = secondary_model(frame, device=device)
        secondary_boxes = [
            (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), int(box.cls[0].item())) 
            for r in results_secondary for box in r.boxes if int(box.cls[0].item()) in PERSON_VEHICLE_CLASS_IDS
        ]

    # Combine detections, prioritizing primary model
    final_boxes = primary_boxes + [box for box in secondary_boxes if box not in primary_boxes]

    # Ensure previous detections persist if no new ones are found
    if not final_boxes:
        final_boxes = previous_boxes
    else:
        previous_boxes = final_boxes

    # Separate vehicles and persons
    vehicle_boxes = [box for box in final_boxes if box[4] in {2, 5, 7}]  # Cars, buses, trucks
    person_boxes = [box for box in final_boxes if box[4] == 0]  # Persons

    # Calculate road region based on vehicle positions
    frame_height, frame_width = frame.shape[:2]
    road_polygon = calculate_road_region(vehicle_boxes, frame_width, frame_height)

    # Log object data for the current frame
    frame_data = {}
    for i, (x1, y1, x2, y2, cls) in enumerate(final_boxes):
        class_name = CLASS_NAMES[cls]
        if class_name not in frame_data:
            frame_data[class_name] = {"Count": 0, "Positions": []}
        frame_data[class_name]["Count"] += 1
        frame_data[class_name]["Positions"].append((x1, y1, x2, y2))

        if cls == 0:
            pedestrian_count += 1
        else:
            vehicle_count += 1

        # **Draw Bounding Boxes** on the frame
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        if cls == 0 and road_polygon is not None and is_person_near_road((x1, y1, x2, y2), road_polygon):
            # Highlight person near road with red box and warning message
            color = (0, 0, 255)  # Red
            cv2.putText(frame, "Person Near Road", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Append frame data to the global DataFrame
    for class_name, data in frame_data.items():
        new_row = pd.DataFrame({
            "Frame": [frame_count],
            "Class": [class_name],
            "Count": [data["Count"]],
            "Positions": [data["Positions"]]
        })
        object_data = pd.concat([object_data, new_row], ignore_index=True)

    # Emit data to update live graph
    socketio.emit('update_chart', {'vehicles': vehicle_count, 'pedestrians': pedestrian_count})
    
    return frame, previous_boxes

csv_ready = False  # Global flag to track CSV readiness

@app.route('/is_csv_ready')
def is_csv_ready():
    global csv_ready
    return {"ready": csv_ready}

def generate_frames(video_path):
    global object_data, csv_ready
    object_data = pd.DataFrame(columns=["Frame", "Class", "Count", "Positions"])
    csv_ready = False  # Reset flag at the start

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    previous_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, previous_boxes = process_frame(frame, frame_count, previous_boxes)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        frame_count += 1

    cap.release()

    # Save the DataFrame to a CSV file after processing the video
    object_data.to_csv(CSV_FILE_PATH, index=False)
    csv_ready = True  # Set flag to True when CSV is ready
    print(f"CSV file saved to {CSV_FILE_PATH}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')
        file = request.files.get('file')
        video_path = None
        filename = None

        if file and file.filename:
            filename = file.filename
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
        elif youtube_url:
            video_path = download_youtube_video(youtube_url)
            if video_path:
                filename = os.path.basename(video_path)

        if video_path:
            return render_template('upload.html', filename=filename)

    return render_template('upload.html', filename=None)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    """
    Stream video frames with object detection.
    """
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_csv')
def download_csv():
    if os.path.exists(CSV_FILE_PATH):
        return send_from_directory(STATIC_FOLDER, "output.csv", as_attachment=True)
    return "CSV File not found", 404

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)