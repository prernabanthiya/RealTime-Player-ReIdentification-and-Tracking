"""
Player Re-Identification in a Single Feed

This script performs object detection and real-time multi-object tracking using YOLOv11 and DeepSORT.
It tracks players consistently even if they go out of frame and return later.

Usage:
- The output will include a tracked video (with bounding boxes and IDs) and a player appearance log.

Author: Prerna (for Liat.ai Internship Task)
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import time

# === CONFIGURATION ===
MODEL_PATH = r'C:\Users\Prerna\PycharmProjects\PlayerReID-Tracking\best.pt'  # Update to your YOLOv11 model path
VIDEO_PATH = r'C:\Users\Prerna\PycharmProjects\PlayerReID-Tracking\15sec_input_720p.mp4'  # Input video path

# Output video path with timestamp
OUTPUT_VIDEO_PATH = rf'output_tracked_{int(time.time())}.mp4'

# DeepSort tracker params
MAX_AGE = 60        # How long to keep lost tracks before deletion
N_INIT = 3          # Frames needed to confirm a track
MAX_COSINE_DISTANCE = 0.2  # Appearance similarity threshold

# Detection confidence threshold for YOLO
YOLO_CONF_THRESH = 0.4

# Device to run model on ('cpu' or 'cuda')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# === INITIALIZATION ===
print("[INFO] Loading YOLOv11 model...")
model = YOLO(MODEL_PATH)

print(f"[INFO] Initializing DeepSort tracker (max_age={MAX_AGE}, n_init={N_INIT}, max_cosine_distance={MAX_COSINE_DISTANCE})...")
tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT, max_cosine_distance=MAX_COSINE_DISTANCE)

print(f"[INFO] Opening video file: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video file {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[INFO] Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {frame_count}")

# Prepare video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

frame_idx = 0
player_log = {}

print("[INFO] Starting player re-identification and tracking...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video reached.")
        break
    frame_idx += 1

    # Run detection with YOLOv11 on current frame
    results = model.predict(source=frame, conf=YOLO_CONF_THRESH, device=DEVICE)
    detections = results[0].boxes.data.cpu().numpy()

    dets_for_tracker = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        class_id = int(cls)
        class_name = model.names[class_id].lower()
        if class_name == 'player':
            bbox = [x1, y1, x2 - x1, y2 - y1]
            dets_for_tracker.append((bbox, conf, class_name))

    # Update tracker with filtered detections
    tracks = tracker.update_tracks(dets_for_tracker, frame=frame)

    # Annotate frame with bounding boxes and consistent IDs
    for track in tracks:
        if not track.is_confirmed() or track.is_deleted():
            continue
        if track.time_since_update > 0:

            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Draw bounding box and player ID label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Player {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Log the appearance of this player at current frame
        if track_id not in player_log:
            player_log[track_id] = []
        player_log[track_id].append(frame_idx)

    # Write annotated frame to output video
    out.write(frame)

    cv2.imshow("Player Re-ID & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Early exit requested by user.")
        break

    # Print progress every 50 frames
    if frame_idx % 50 == 0:
        print(f"[INFO] Processed {frame_idx} / {frame_count} frames")

# Cleanup resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Output video saved as: {os.path.abspath(OUTPUT_VIDEO_PATH)}")

# Save detailed player appearance log to file
log_path = os.path.join(os.getcwd(), 'player_log.txt')
with open(log_path, 'w') as f:
    for pid, frames_seen in player_log.items():
        frames_str = ','.join(map(str, frames_seen))
        f.write(f"Player {pid}: Frames {frames_str}\n")

print(f"[INFO] Player appearance log saved at: {log_path}")
print("[INFO] Processing complete.")