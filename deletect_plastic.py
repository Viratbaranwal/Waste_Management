from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Load the trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# Initialize video capture (0 for webcam)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set webcam width
cap.set(4, 720)  # Set webcam height

# Class names (only plastic waste)
classNames = ["bottle"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()  # Read frame from webcam
    results = model(img, stream=True)  # Perform detection on the frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Apply a confidence threshold (e.g., 0.5)
            if conf > 0.5:  # Adjust threshold as needed
                cvzone.putTextRect(img, f'{classNames[0]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    # Display the frame with detections
    cv2.imshow("Plastic Waste Detection", img)
    cv2.waitKey(1)