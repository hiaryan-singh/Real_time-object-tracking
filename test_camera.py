import cv2

# 1. Connect to the camera (0 = Default built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the camera initialized successfully
if not cap.isOpened():
    print("Error: Could not initialize camera feed.")
    exit()

print("Camera is active. Press 'q' on your keyboard to exit.")

# 2. Continuous loop to read video frames
while True:
    # Read a single frame
    success, frame = cap.read()

    # Break the loop if the frame could not be retrieved
    if not success:
        print("Error: Failed to grab frame from camera.")
        break

    # Display the current frame in a window
    cv2.imshow("Internship Task - Step 2: Camera Feed", frame)

    # Wait for 1 millisecond; break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 3. Release hardware resources and close display windows
cap.release()
cv2.destroyAllWindows()