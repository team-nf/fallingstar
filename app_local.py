import cv2
import numpy as np
import time
import os
import argparse
import supervision as sv
from roboflow import Roboflow

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run FRC 2025 Algae and Coral detection on camera feed')
parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
parser.add_argument('--download', action='store_true', help='Download the model first')
args = parser.parse_args()

# Download model if requested or if it doesn't exist
if args.download or not os.path.exists("best.pt"):
    print("Downloading model...")
    # Hardcoded API key
    ROBOFLOW_API_KEY = "XmQnHuLA3FZgcP19fRD8"
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("frc-team-503-frog-force").project("frc-2025-algae-and-coral")
    
    # Download model directly in current directory
    model = project.version(1).download("yolov8")
    
    print("Model downloaded to current directory")

# Initialize the local model using OpenCV DNN
print("Loading model...")
model_format = ""
if os.path.exists("model.onnx"):
    # Use ONNX model
    net = cv2.dnn.readNetFromONNX("model.onnx")
    model_format = "onnx"
    print("Using ONNX model")
elif os.path.exists("best.pt"):
    # Use PyTorch model
    try:
        import torch
        from ultralytics import YOLO
        yolo_model = YOLO("best.pt")
        model_format = "yolo"
        print("Using YOLOv8 model")
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        exit(1)
else:
    print("Error: No supported model found in current directory")
    exit(1)

# Load class names
classnames = []
with open("classes.txt", "r") as f:
    classnames = [line.strip() for line in f.readlines()]
print(f"Loaded {len(classnames)} classes: {classnames}")

# Initialize camera
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"Error: Could not open camera {args.camera}")
    exit(1)

# Initialize detection visualizer
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

print("Starting detection... Press 'q' to quit")

# Enable OpenCV DNN CUDA backend if available
if model_format == "onnx" and cv2.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("CUDA acceleration enabled")

# For FPS calculation
prev_time = time.time()
frame_count = 0
fps = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Get the frame dimensions
    height, width = frame.shape[:2]
    
    # Prepare detections list
    detections_list = []
    
    if model_format == "onnx":
        # Prepare input blob for ONNX model
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Forward pass
        outputs = net.forward(net.getUnconnectedOutLayersNames())[0]
        
        # Process detections
        for output in outputs:
            confidence = output[4]
            if confidence >= args.confidence:
                classes_scores = output[5:]
                class_id = np.argmax(classes_scores)
                if classes_scores[class_id] > args.confidence:
                    # YOLO format to bounding box coordinates
                    x, y, w, h = output[0], output[1], output[2], output[3]
                    x1 = int((x - w/2) * width)
                    y1 = int((y - h/2) * height)
                    x2 = int((x + w/2) * width)
                    y2 = int((y + h/2) * height)
                    
                    detections_list.append([x1, y1, x2, y2, confidence, class_id])
    
    elif model_format == "yolo":
        # Use YOLOv8 model
        results = yolo_model(frame, conf=args.confidence)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            detections_list.append([x1, y1, x2, y2, confidence, class_id])
    
    # Create Detections object
    if detections_list:
        detections_array = np.array(detections_list)
        detections = sv.Detections(
            xyxy=detections_array[:, :4],
            confidence=detections_array[:, 4],
            class_id=detections_array[:, 5].astype(int)
        )
        
        # Format labels
        labels = [
            f"{classnames[class_id]} {confidence:.2f}"
            for confidence, class_id in zip(detections.confidence, detections.class_id)
        ]
        
        # Annotate and display the frame
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    
    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    if current_time - prev_time >= 1.0:
        fps = frame_count / (current_time - prev_time)
        frame_count = 0
        prev_time = current_time
        print(f"FPS: {fps:.2f}")
    
    # Display FPS on frame
    cv2.putText(
        frame, 
        f"FPS: {fps:.2f}", 
        (10, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Display info
    cv2.putText(
        frame, 
        f"FRC 2025 - Algae and Coral Detector", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Display the resulting frame
    cv2.imshow('FRC 2025 - Algae and Coral Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows() 