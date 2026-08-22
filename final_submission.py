import cv2
import time
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

# 1. Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 2. Input source: Change to "sample_video.mp4" or keep 0 for webcam
VIDEO_SOURCE = 0  
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Get video properties for saving output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30

# 3. Define Video Writer to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_tracked.mp4", fourcc, fps, (frame_width, frame_height))

# Dictionary to store trajectory path points for each ID
track_history = defaultdict(lambda: [])

prev_time = 0

print("Running pipeline. Saving output to 'output_tracked.mp4'. Press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate real-time FPS
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Run detection and tracking (ByteTrack)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
            x1, y1, x2, y2 = box
            class_name = model.names[cls_id]
            label = f"ID:{track_id} {class_name} {conf:.2f}"

            # Calculate bottom center of bounding box for trajectory
            center_point = (int((x1 + x2) / 2), int(y2))
            track_history[track_id].append(center_point)
            if len(track_history[track_id]) > 30:  # Keep last 30 points
                track_history[track_id].pop(0)

            # Draw trajectory path line
            points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw Label Header
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0] + 6, y1), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display real-time FPS overlay
    cv2.putText(frame, f"FPS: {int(fps_display)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write frame to output video file
    out.write(frame)

    # Show live preview
    cv2.imshow("Internship Task 4 - Complete Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Process finished. Video saved as 'output_tracked.mp4'.")