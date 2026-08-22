import cv2
from ultralytics import YOLO

# 1. Load pre-trained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# 2. Initialize webcam feed (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Object detection started. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # 3. Perform inference on the current frame
    results = model(frame, verbose=False)

    # 4. Extract bounding boxes, class names, and confidence scores
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Filter out low-confidence detections
        if confidence > 0.5:
            label = f"{class_name} {confidence:.2f}"

            # Draw green bounding box around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw text label above the bounding box
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # 5. Display real-time output
    cv2.imshow("Step 3 - Real-Time YOLO Detection", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 6. Release hardware resources
cap.release()
cv2.destroyAllWindows()