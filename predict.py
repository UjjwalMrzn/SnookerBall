import os
import cv2
from ultralytics import YOLO

# Define the directory where the video is located
VIDEOS_DIR = os.path.join(r'C:\Users\Ujjwal\Downloads', 'New folder')

# Define the path to the input video file
video_path = os.path.join(VIDEOS_DIR, 'footage5.mp4')

# Define the path to the output video file
video_path_out = '{}_out.mp4'.format(video_path)

# Open the input video file
cap = cv2.VideoCapture(video_path)

# Read the first frame from the video
ret, frame = cap.read()

# Get the height (H) and width (W) of the frame
H, W, _ = frame.shape

# Initialize the video writer to write the output video
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Define the path to the YOLO model weights
model_path = os.path.join('.', 'models', r'D:\2024\pythonProject1', 'runs', 'detect', 'train11', 'weights', 'last.pt')

# Load the YOLO model
model = YOLO(model_path)

# Define the confidence threshold for detecting objects
threshold = 0.69

# Define a dictionary to map class IDs to class names
class_name_dict = {0: 'triangle'}

# Variable to track if the triangle is detected
triangle_detected = False
frames_without_triangle = 0
max_frames_without_triangle = 7  # Reduced to detect disappearance quicker

# Loop through the video frames
while ret:
    # Get the model predictions for the current frame
    results = model(frame)

    # Create a copy of the current frame to draw the detections on
    new_frame = frame.copy()

    # Assume triangle is not detected in this frame
    current_frame_has_triangle = False

    # Process the model results
    for result in results:
        for box in result.boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]  # x1, y1, x2, y2
            # Get the confidence score of the detection
            conf = box.conf[0]  # confidence
            # Get the class ID of the detected object
            cls_id = box.cls[0]  # class id

            # If the confidence score is above the threshold, draw the bounding box
            if conf > threshold:
                label = class_name_dict.get(int(cls_id), 'Unknown')
                if label == 'triangle':
                    current_frame_has_triangle = True
                color = (0, 255, 0)
                cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(new_frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Check if the triangle was detected in the current frame
    if current_frame_has_triangle:
        triangle_detected = True
        frames_without_triangle = 0
    else:
        if triangle_detected:
            frames_without_triangle += 1
            if frames_without_triangle >= max_frames_without_triangle:
                triangle_detected = False
                frames_without_triangle = 0

    # If the triangle has scattered/disappeared, display "Play"
    if not triangle_detected:
        cv2.putText(new_frame, 'Play', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Write the processed frame to the output video
    out.write(new_frame)
    # Read the next frame from the video
    ret, frame = cap.read()

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
