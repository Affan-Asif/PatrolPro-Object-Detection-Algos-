import cv2
from ultralytics import YOLO
from twilio.rest import Client

# Twilio credentials
TWILIO_ACCOUNT_SID = 'AC589c729c679a8f7307a2fc46ec90ba90'  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = 'a47c041d25202486399dcf1302b64d25'  # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = '+12185066690'  # Replace with your Twilio phone number
TARGET_PHONE_NUMBER = '+919121795950'  # Replace with the target phone number

# Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms():
    try:
        message = client.messages.create(
            body="Alert: High activity detected! A woman is being harassed...Please take necessary action!!",
            from_=TWILIO_PHONE_NUMBER,
            to=TARGET_PHONE_NUMBER
        )
        print(f"SMS sent: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")

# Load the YOLO model
model = YOLO('women_safety3.pt')

# Define label names
LABELS = {0: 'Man', 1: 'Woman'}  # Update as per your labels

# Function to check if two bounding boxes overlap
def check_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Check for overlap
    if x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2:
        return True
    return False

# Function to process the frame and detect objects
def process_frame(frame, activity_state):
    results = model(frame)
    detections = results[0].boxes  # Access detection boxes

    man_boxes = []
    woman_boxes = []

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

            # Store bounding boxes for men and women
            if cls == 0:  # Man
                man_boxes.append((x1, y1, x2, y2))
            elif cls == 1:  # Woman
                woman_boxes.append((x1, y1, x2, y2))

    # Check for overlap between men and women
    if activity_state["activity"] == "low":
        for man_box in man_boxes:
            for woman_box in woman_boxes:
                if check_overlap(man_box, woman_box):
                    activity_state["activity"] = "high"
                    send_sms()  # Send SMS alert
                    break

    # Display activity level on the frame
    cv2.putText(frame, f'Activity: {activity_state["activity"]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Input video file path
video_path = 'C:\\Affan\\Affan\\__S.I.H. 2k24__\\HackRev 2025\\WasteManagement Predictor\\input\\Women_Safety.mp4'  # Replace with your video file path
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
output_path = 'C:\\Affan\\Affan\\__S.I.H. 2k24__\\HackRev 2025\\WasteManagement Predictor\\output\\women_safety3.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Activity state dictionary
activity_state = {"activity": "low"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 1920x1080
    frame = cv2.resize(frame, (1920, 1080))

    # Process the frame
    frame = process_frame(frame, activity_state)

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
