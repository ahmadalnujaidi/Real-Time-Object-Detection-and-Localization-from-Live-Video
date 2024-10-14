import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
onnx_model_path = "model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Get input and output names from the model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Open the testing video
video_path = "testing_video_DETECTIONS.mp4"
cap = cv2.VideoCapture(video_path)

# Get total number of frames (for progress tracking)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total Frames in Video: {total_frames}")

pred_boxes = []  # List to store predicted boxes
pred_labels = []  # List to store predicted labels

# Track the number of processed frames
processed_frames = 0

# Process the video and run inference on each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when the video ends

    # Prepare the input frame (resize and reshape to match model's input size)
    input_tensor = cv2.resize(frame, (640, 640)).astype('float32')
    input_tensor = input_tensor.transpose(2, 0, 1)  # Change shape to (C, H, W)
    input_tensor = input_tensor[np.newaxis, ...]  # Add batch dimension

    # Run inference on the input frame
    outputs = session.run([output_name], {input_name: input_tensor})

    # Extract bounding boxes and labels from the model's output
    for detection in outputs[0]:  # Adjust based on your model's output structure
        x1, y1, x2, y2, conf, class_id = detection[:6]

        # Ensure class_id is a scalar (handle arrays)
        if isinstance(class_id, (list, np.ndarray)):
            class_id = int(class_id[0])  # Extract scalar from array
        else:
            class_id = int(class_id)

        # Store the predictions
        pred_boxes.append((x1, y1, x2, y2))
        pred_labels.append(class_id)

    # Update progress
    processed_frames += 1
    if processed_frames % 10 == 0 or processed_frames == total_frames:
        print(f"Processed {processed_frames}/{total_frames} frames ({(processed_frames / total_frames) * 100:.2f}%)")

cap.release()  # Release the video capture object

# Print predictions to verify OPTIONAL! uncomment to see.
# print("Predicted Boxes:", pred_boxes)
# print("Predicted Labels:", pred_labels)

print("Predictions Completed!")
print("now finding ground truth...")
#---------------------------------------------------------------------------------------------  

# GROUND TRUTH !!!

# DOWNLOAD VALIDATION DATASET ZIP FROM GOOGLE DRIVE
import gdown
import zipfile
import os

# Replace with your own Google Drive file ID and output name
file_id = "1UGnBqZQT0amOFeKVTMNM9briM43Vwxhd"  # Change this to your file's ID
output = "val_data.zip"  # Name of the downloaded file
# https://drive.google.com/file/d/1UGnBqZQT0amOFeKVTMNM9briM43Vwxhd/view?usp=drive_link

# Download the zip file from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Extract the zip file to a specific directory
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("data")  # Extract into 'coco8' folder

# Adjust path to the validation labels folder
labels_path = "data/val"  # Path within the extracted zip

gt_boxes = []  # List to store ground truth boxes
gt_labels = []  # List to store ground truth labels

# Get the total number of label files for progress tracking
total_files = len(os.listdir(labels_path))
print(f"Total Label Files: {total_files}")

# Track processed files
processed_files = 0

# Iterate through the label files and extract information
for label_file in os.listdir(labels_path):
    with open(os.path.join(labels_path, label_file), 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Convert normalized coordinates to (x1, y1, x2, y2)
            x1 = (x_center - width / 2) * 640  # Assuming 640x640 input size
            y1 = (y_center - height / 2) * 640
            x2 = (x_center + width / 2) * 640
            y2 = (y_center + height / 2) * 640

            # Store the ground truth boxes and labels
            gt_boxes.append((x1, y1, x2, y2))
            gt_labels.append(int(class_id))

    # Update and display progress
    processed_files += 1
    if processed_files % 10 == 0 or processed_files == total_files:
        print(f"Processed {processed_files}/{total_files} files ({(processed_files / total_files) * 100:.2f}%)")

# OPTIONAL: Uncomment to print the ground truth data for verification
# print("Ground Truth Boxes:", gt_boxes)
# print("Ground Truth Labels:", gt_labels)


#---------------------------------------------------------------------------------------------


# CALCULATE ACCURACY FUNCTION

from shapely.geometry import box

# Function to safely format bounding boxes by flattening nested lists/arrays
def format_box(bbox):
    """Convert bounding box elements to floats, flattening any nested arrays/lists."""
    return tuple(float(coord) if not isinstance(coord, (list, np.ndarray)) else float(coord[0]) for coord in bbox)

# Function to compute IoU (Intersection over Union)
def compute_iou(pred_box, gt_box):
    # Ensure boxes are properly formatted as (x1, y1, x2, y2) with floats
    pred_box = format_box(pred_box)
    gt_box = format_box(gt_box)

    pred = box(*pred_box)  # Create a Shapely box from (x1, y1, x2, y2)
    gt = box(*gt_box)  # Create a Shapely box from (x1, y1, x2, y2)

    intersection = pred.intersection(gt).area
    union = pred.union(gt).area
    return intersection / union if union != 0 else 0

# Function to calculate the accuracy metric (e) with progress tracking
def compute_e(pred_boxes, gt_boxes, pred_labels, gt_labels):
    e = 0  # Initialize total error
    total_predictions = len(pred_boxes)  # Total number of predicted boxes

    # Loop over each predicted box
    for i, pred_box in enumerate(pred_boxes):
        min_error = float('inf')  # Start with a high error value

        # Compare with all ground truth boxes
        for j, gt_box in enumerate(gt_boxes):
            # Calculate d_ij: 0 if labels match, 1 otherwise
            d = 0 if pred_labels[i] == gt_labels[j] else 1

            # Calculate f_ij: 0 if IoU >= 0.5, 1 otherwise
            iou = compute_iou(pred_box, gt_box)
            f = 0 if iou >= 0.5 else 1

            # Error is the maximum of d_ij and f_ij
            error = max(d, f)
            min_error = min(min_error, error)  # Keep the smallest error

        e += min_error  # Add the smallest error for this prediction

        # Display progress every 10 predictions or at the end
        if (i + 1) % 10 == 0 or (i + 1) == total_predictions:
            print(f"Processed {i + 1}/{total_predictions} predictions "
                  f"({(i + 1) / total_predictions * 100:.2f}%)")

    # Normalize by the number of predictions
    accuracy = e / len(pred_boxes) if pred_boxes else 1.0
    return accuracy

# Example Usage: Calculate the accuracy (e) using predictions and ground truth
accuracy = compute_e(pred_boxes, gt_boxes, pred_labels, gt_labels)
print(f"Accuracy (e): {accuracy:.4f}")


# Accuracy (e): 1.0000