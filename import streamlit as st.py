import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO
import tempfile
from pathlib import Path

# Load YOLO model (Ensure this points to your model file)
model = YOLO('yolov8l.pt')  # Replace with the correct path to your YOLO model

# Simple Tracker to manage car tracking and crossing status
class SimpleTracker:
    def __init__(self, max_distance=50, max_lost=30):
        self.objects = {}  # ID: (centroid, crossed_mid, crossed_out, lost_frames)
        self.max_distance = max_distance
        self.max_lost = max_lost  # Maximum number of frames an object can be lost before being removed
        self.next_id = 0
        self.bounding_boxes = {}  # Store bounding boxes for each object

    def register(self, centroid, bounding_box):
        self.objects[self.next_id] = (centroid, False, [False] * len(out_lanes), 0)
        self.bounding_boxes[self.next_id] = bounding_box
        self.next_id += 1

    def update(self, detections, bounding_boxes):
        updated_objects = {}
        updated_bounding_boxes = {}
        for obj_id, (old_centroid, crossed_mid, crossed_out_lanes, lost_frames) in self.objects.items():
            found_match = False
            for i, det in enumerate(detections):
                new_centroid = det
                if self._is_same_object(old_centroid, new_centroid):
                    updated_objects[obj_id] = (new_centroid, crossed_mid, crossed_out_lanes, 0)
                    updated_bounding_boxes[obj_id] = bounding_boxes[i]
                    detections.pop(i)
                    bounding_boxes.pop(i)
                    found_match = True
                    break
            if not found_match:
                lost_frames += 1
                if lost_frames < self.max_lost:
                    updated_objects[obj_id] = (old_centroid, crossed_mid, crossed_out_lanes, lost_frames)
                    updated_bounding_boxes[obj_id] = self.bounding_boxes[obj_id]

        for i, det in enumerate(detections):
            self.register(det, bounding_boxes[i])
            updated_objects[self.next_id - 1] = (det, False, [False] * len(out_lanes), 0)
            updated_bounding_boxes[self.next_id - 1] = bounding_boxes[i]

        self.objects = updated_objects
        self.bounding_boxes = updated_bounding_boxes

    def _is_same_object(self, old, new):
        distance = np.linalg.norm(np.array(old) - np.array(new))
        return distance < self.max_distance

# Function to check if a point is within the lane boundaries
def is_crossing_lane(centroid, lane):
    (x, y) = centroid
    (x1, y1), (x2, y2) = lane
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

# Function to check if a vehicle has fully crossed the lane
def fully_crossed(centroid, lane, previous_position):
    (x, y) = centroid
    (x1, y1), (x2, y2) = lane
    if previous_position == "before" and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        return True
    return False

# Main detection function with progress bar
def detect_and_count(video_path, output_path, in_lane, out_lanes, cropped_folder):
    tracker = SimpleTracker(max_lost=30)
    in_count = 0
    out_counts = [0] * len(out_lanes)
    previous_positions = {}

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize Streamlit progress bar
    progress = st.progress(0)

    frame_number = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection with a confidence threshold to filter weak detections
        results = model(frame)
        detections = results[0].boxes

        centroids = []
        bounding_boxes = []
        confidence_threshold = 0.5
        if detections is not None and len(detections) > 0:
            for box in detections:
                cls_id = int(box.cls)
                conf = float(box.conf[0])
                if cls_id == 2 and conf > confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    centroids.append(centroid)
                    bounding_boxes.append((x1, y1, x2, y2))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        tracker.update(centroids, bounding_boxes)

        for obj_id, (centroid, crossed_mid, crossed_out_lanes, lost_frames) in tracker.objects.items():
            if not crossed_mid and is_crossing_lane(centroid, in_lane):
                crossed_mid = True
                in_count += 1
                tracker.objects[obj_id] = (centroid, crossed_mid, crossed_out_lanes, lost_frames)

            for i, out_lane in enumerate(out_lanes):
                if crossed_mid and not crossed_out_lanes[i] and is_crossing_lane(centroid, out_lane):
                    crossed_out_lanes[i] = True
                    out_counts[i] += 1
                    tracker.objects[obj_id] = (centroid, crossed_mid, crossed_out_lanes, lost_frames)

                    x1, y1, x2, y2 = tracker.bounding_boxes[obj_id]
                    cropped_car = frame[y1:y2, x1:x2]
                    if cropped_car.size > 0:
                        cv2.imwrite(os.path.join(cropped_folder, f'car_{obj_id}_outlane{i+1}.jpg'), cropped_car)

        out.write(frame)

        # Update the progress bar
        frame_number += 1
        progress.progress(frame_number / total_frames)

    cap.release()
    out.release()

    return in_count, out_counts

# Streamlit app
st.title("Vehicle Detection and Lane Crossing Tracker")

# Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Create a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Display the original video
    st.header("Original Video")
    st.video(tfile.name)

    # Set output video and folder paths
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    cropped_folder = tempfile.mkdtemp()

    # Define IN and OUT lanes
    in_lane = [(550, 420), (630, 370)]  # Example IN lane coordinates
    out_lanes = [
        [(580, 600), (950, 580)],
        [(390, 430), (420, 510)],
        [(1100, 450), (1300, 550)],
        [(650, 350), (840, 350)]
    ]

    # Run the detection and counting function on the uploaded video
    st.write("Processing video, please wait...")
    in_count, out_counts = detect_and_count(tfile.name, output_video_path, in_lane, out_lanes, cropped_folder)
    st.write("Video processing complete!")

    # Display the processed video
    st.header("Processed Video with Vehicle Detection")
    st.video(output_video_path)

    # Optionally, create a collapsible section for cropped car images
    with st.expander("Show Cropped Images of Cars"):
        cropped_images = os.listdir(cropped_folder)
        for img_file in cropped_images:
            img_path = os.path.join(cropped_folder, img_file)
            st.image(img_path, caption=f"Cropped Car: {img_file}", use_column_width=True)