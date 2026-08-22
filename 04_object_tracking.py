import cv2
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# 2. Initialize video stream (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access video feed.")
    exit()

print("Object detection and tracking running. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # 3. Detect and Track objects
    # persist=True maintains tracking IDs across consecutive frames
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml", verbose=False
    )

    # 4. Extract boxes, persistent IDs, and class labels
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(
            boxes, track_ids, class_ids, confidences
        ):
            x1, y1, x2, y2 = box
            class_name = model.names[cls_id]
            label = f"ID:{track_id} {class_name} {conf:.2f}"

            # Draw bounding box (Blue color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw background banner for readable text
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )[0]
            cv2.rectangle(
                frame,
                (x1, y1 - 25),
                (x1 + text_size[0] + 6, y1),
                (255, 0, 0),
                -1,
            )

            # Draw text label with Tracking ID and Class Name
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    # 5. Display real-time output
    cv2.imshow("Internship Task 4 - Object Detection and Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up hardware resources
cap.release()
cv2.destroyAllWindows()