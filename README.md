# Real-Time Object Detection & Localization from Live Video

[![Watch the video](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://youtu.be/X4o2S9-aUVI)

## Project Overview

This project showcases a real-time object detection system powered by a custom-trained YOLO11m model. It demonstrates live detection and localization of objects from video streams, with bounding boxes drawn around detected objects.

## Key Highlights

### 1. **Comprehensive Dataset Management**

- Created **numerous Python scripts** to manipulate, manage, and tweak the dataset for optimal model training.
- Scripts include tasks for image preprocessing, annotation format conversion, data augmentation, and dataset splitting.

### 2. **Training the YOLO11m Model on Vast.ai**

- Trained the YOLO11m model using my dataset through a **Jupyter Notebook** on **Vast.ai**, leveraging multiple rented GPUs to **accelerate training**.
- This setup significantly reduced training time while handling a large volume of data effectively.

### 3. **Evaluating Model Performance**

- Focused on key metrics: **speed, memory usage**, and **accuracy**.
- Speed and memory metrics were measured to ensure the model could operate efficiently in real-time conditions.

### 4. **Accuracy Calculation with Custom Scripts**

- The most challenging part was calculating **accuracy**. I developed a **custom script** that:
  - Extracts **ground truth labels** from the dataset annotations.
  - Extracts **predicted bounding boxes and labels** from a **processed video** where the model detected objects.
  - Compares predictions against the ground truth to compute metrics like precision and recall.

## How to Run the Project

1. setup the scripts in your enviroment.
2. download the model (downloadModel.py) and it will also download a testing_video.mp4
3. run the speedMeasurement.py to get the speed, and video with detections (testing_video_DETECTIOSN.mp4)
4. check the memory usage with memoryMeasurement.py
5. get the accuracy(e) measurement from running accuracyMeasurement.py
6. Finally plug in those values into evaluateModel.py to get a final combined score of the model's performance!
