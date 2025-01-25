# import cv2
# from ultralytics import YOLO
# import os
# import numpy as np

# # Load the YOLO model
# model = YOLO('best.pt')

# # Define label names
# LABELS = {0: "Garbage", 1: "Person"}  # Update as per your labels

# # Define the region coordinates
# REGION = [(645, 665), (808, 665), (808, 803), (645, 803)]

# # Input and output paths
# INPUT_FOLDER = 'C:\\Affan\\Affan\\__S.I.H. 2k24__\\HackRev 2025\\WasteManagement Predictor\\input\\'
# OUTPUT_FOLDER = 'C:\\Affan\\Affan\\__S.I.H. 2k24__\\HackRev 2025\\WasteManagement Predictor\\output\\'
# CAPTURE_FOLDER = os.path.join(INPUT_FOLDER, 'captures')
# os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# # Function to process the frame and detect objects
# def process_frame(frame, frame_count):
#     results = model(frame)
#     detections = results[0].boxes  # Access detection boxes
#     captured = False

#     for box in detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
#         conf = box.conf[0].item()  # Confidence score
#         cls = int(box.cls[0].item())  # Class index

#         if cls in LABELS:  # Check if the detected class is in the defined labels
#             label = LABELS[cls]
#             color = (0, 255, 0) if cls == 0 else (255, 0, 0) if cls == 1 else (0, 0, 255)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, f'{label}: {conf:.2f}', (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#             # Check if garbage is within the defined region
#             if label == "Garbage":
#                 center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
#                 if cv2.pointPolygonTest(np.array(REGION, dtype=np.int32), (center_x, center_y), False) >= 0:
#                     # Capture the frame
#                     if not captured:
#                         capture_path = os.path.join(CAPTURE_FOLDER, f'capture_frame_{frame_count}.jpg')
#                         cv2.imwrite(capture_path, frame)
#                         print(f"Frame captured: {capture_path}")
#                         captured = True
#     return frame

# # Input video file path
# video_path = os.path.join(INPUT_FOLDER, 'Garbage_Throwing.mp4')  # Replace with your video file path
# cap = cv2.VideoCapture(video_path)

# # Check if the video file opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Get video properties
# frame_width = 1920
# frame_height = 1080
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Output video writer
# output_path = os.path.join(OUTPUT_FOLDER, 'garbage_throwing3.mp4')
# out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize frame to 1920x1080
#     frame = cv2.resize(frame, (1920, 1080))

#     # Process the frame
#     frame = process_frame(frame, frame_count)
#     frame_count += 1

#     # Display the frame
#     cv2.imshow('Video Detection', frame)

#     # Write the frame to the output video file
#     out.write(frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import os
import numpy as np
import time

# Load the YOLO model
model = YOLO('best.pt')

# Define label names
LABELS = {0: "Garbage", 1: "Person"}  # Update as per your labels

# Define the region coordinates
REGION = [(645, 665), (808, 665), (808, 803), (645, 803)]

# Input and output paths
INPUT_FOLDER = 'C:\\Affan\\Affan\\__S.I.H. 2k24__\\HackRev 2025\\WasteManagement Predictor\\input\\'
OUTPUT_FOLDER = 'C:\\Affan\\Affan\\__S.I.H. 2k24__\\HackRev 2025\\WasteManagement Predictor\\output\\'
CAPTURE_FOLDER = os.path.join(INPUT_FOLDER, 'captures')
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# Global variables to manage capture timing
garbage_detected_time = None
captured = False

def process_frame(frame, frame_count):
    global garbage_detected_time, captured

    results = model(frame)
    detections = results[0].boxes  # Access detection boxes

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class index

        if cls in LABELS:  # Check if the detected class is in the defined labels
            label = LABELS[cls]
            color = (0, 255, 0) if cls == 0 else (255, 0, 0) if cls == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label}: {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Check if garbage is within the defined region
            if label == "Garbage":
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if cv2.pointPolygonTest(np.array(REGION, dtype=np.int32), (center_x, center_y), False) >= 0:
                    if garbage_detected_time is None:
                        garbage_detected_time = time.time()  # Record the time of detection

                    # Capture the frame if 5 seconds have passed and not already captured
                    if not captured and time.time() - garbage_detected_time >= 20:
                        capture_path = os.path.join(CAPTURE_FOLDER, f'capture_frame_{frame_count}.jpg')
                        cv2.imwrite(capture_path, frame)
                        print(f"Frame captured: {capture_path}")
                        captured = True
    return frame

# Input video file path
video_path = os.path.join(INPUT_FOLDER, 'Garbage_Throwing.mp4')  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = 1920
frame_height = 1080
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
output_path = os.path.join(OUTPUT_FOLDER, 'garbage_throwing3.mp4')
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 1920x1080
    frame = cv2.resize(frame, (1920, 1080))

    # Process the frame
    frame = process_frame(frame, frame_count)
    frame_count += 1

    # Display the frame
    cv2.imshow('Video Detection', frame)

    # Write the frame to the output video file
    out.write(frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
