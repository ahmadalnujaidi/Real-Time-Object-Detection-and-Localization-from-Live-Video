# SPEED MEASUREMENT # Average Speed: 40.71 ms
import os
os.system('pip install opencv-python ultralytics')
import cv2
import time
from ultralytics import YOLO

# Load your YOLO model
model = YOLO("model.onnx")

# Open the input video with OpenCV
cap = cv2.VideoCapture('testing_video.mp4')

# Get the video's properties (width, height, frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save the output video
out = cv2.VideoWriter('testing_video_DETECTIONS.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

total_time = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model(frame)  # Run detection
    end = time.time()

    total_time += (end - start) * 1000  # Convert to milliseconds
    frame_count += 1

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

cap.release()
out.release()

avg_speed = total_time / frame_count  # Average time per frame in milliseconds
print(f'Average Speed: {avg_speed:.2f} ms')