# RealTime-Player-ReIdentification-and-Tracking

This project performs player detection and real-time re-identification in a single video feed using YOLOv11 for object detection and DeepSORT for tracking. It assigns consistent IDs to players even if they leave and re-enter the frame, simulating a real-time tracking scenario in sports footage.

---

## Features

- Detects players in each frame using a fine-tuned YOLOv11 model.
- Tracks multiple players consistently using DeepSORT tracker.
- Maintains player IDs even when players are temporarily out of the frame.
- Outputs a video annotated with bounding boxes and player IDs.
- Generates a player appearance log recording frames where each player appears.

---


### Requirements

- Python 3.8+
- CUDA-enabled GPU (recommended for faster inference; fallback to CPU if unavailable)

## Approach and Methodology
- Detection: Used a fine-tuned YOLOv11 model trained to detect players and the ball.

- Tracking: Used DeepSORT to assign persistent IDs by combining motion and appearance features.

- Re-Identification: DeepSORT appearance features help maintain ID consistency even when players temporarily leave the frame.

- Optimization: Tuned tracker parameters (max_age, n_init, and max_cosine_distance) to balance accuracy and robustness.

## Challenges and Future Work
- Handling occlusions and overlapping players remains challenging.

- Tracking accuracy depends on the quality of the detection model.

- Future improvements could include integrating pose estimation or temporal smoothing.

- Real-time deployment optimization for live broadcast scenarios.


# Contact: 
Created by Prerna — for the Lial.ai internship task.
[LinkedIn- prerna-banthiya](https://www.linkedin.com/in/prerna-banthiya/ )

