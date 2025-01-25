import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO('potholes.pt')

# Function to process the frame and count potholes
def process_frame(frame):
    results = model(frame)
    detections = results[0].boxes  # Access detection boxes

    pothole_count = 0
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class index

        if cls == 0:  # Assuming class 0 corresponds to potholes
            pothole_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Pothole: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame, pothole_count

# Input video file path
video_path = 'C:\Affan\Affan\__S.I.H. 2k24__\HackRev 2025\WasteManagement Predictor\input\potholes_input.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
output_path = 'output_video2.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    frame, pothole_count = process_frame(frame)
    frame = cv2.resize(frame, (1920, 1080))

    # Display total potholes on the top-right corner
    cv2.putText(frame, f'Total Potholes: {pothole_count}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write frame to output file
    out.write(frame)

    # Display the frame
    cv2.imshow('Pothole Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

