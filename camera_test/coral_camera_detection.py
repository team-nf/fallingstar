#!/usr/bin/env python3

"""
Real-time camera feed object detection using Google Coral TPU with SSD MobileNet v2
"""

import os
import argparse
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Import Coral TPU libraries
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file

def load_labels(labels_file):
    """Loads labels from file"""
    with open(labels_file, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def preprocess_frame(frame, input_size):
    """Preprocess the camera frame to meet SSD MobileNet v2 input requirements"""
    # Convert OpenCV BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Resize the image to the required input dimensions
    resized_image = pil_image.resize(input_size, Image.LANCZOS)
    
    return resized_image, rgb_frame

def draw_detection_results(frame, detections, labels, input_size, fps=0, threshold=0.3):
    """Draw detection results on the camera frame"""
    height, width, _ = frame.shape
    
    # Scale factors to map from model input size to original frame size
    x_scale = width / input_size[0]
    y_scale = height / input_size[1]
    
    # Draw bounding boxes and labels
    for i, det in enumerate(detections):
        if det.score < threshold:
            continue
        
        # Get bounding box coordinates
        bbox = det.bbox
        xmin, ymin, xmax, ymax = bbox
        
        # Scale coordinates to match the original frame size
        xmin = max(0, int(xmin * x_scale))
        ymin = max(0, int(ymin * y_scale))
        xmax = min(width, int(xmax * x_scale))
        ymax = min(height, int(ymax * y_scale))
        
        # Ensure valid coordinates
        if xmin >= xmax or ymin >= ymax:
            continue
        
        # Choose a color based on class ID for better distinction
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        color = colors[det.id % len(colors)]
        
        # Draw bounding box with OpenCV
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
        
        # Get class label
        class_id = det.id
        score = det.score
        
        label_text = f"{labels.get(class_id, class_id)}: {score:.2f}"
        
        # Draw text background
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (xmin, ymin - 25), (xmin + text_size[0], ymin), color, -1)
        
        # Draw text
        cv2.putText(frame, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add FPS info
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Run real-time object detection on camera feed using Coral TPU')
    parser.add_argument('--model', type=str, default='models/ll/GOOGLECORAL_coral_and_algae_monochrome.tflite',
                        help='Path to the TFLite model')
    parser.add_argument('--labels', type=str, default='models/ll/labels.txt',
                        help='Path to the labels file')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device number (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera feed width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera feed height (default: 480)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Load labels
    labels = {}
    if args.labels and os.path.exists(args.labels):
        labels = load_labels(args.labels)
    
    # Initialize the TPU with the detection model
    print("Loading SSD MobileNet v2 model on Edge TPU...")
    interpreter = edgetpu.make_interpreter(args.model)
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    _, input_height, input_width, _ = input_shape
    input_size = (input_width, input_height)  # (width, height)
    
    print(f"Model input shape: {input_shape}")
    
    # Initialize video capture
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # FPS calculation variables
    frame_count = 0
    fps = 0
    start_time = time.time()
    
    print("Starting detection. Press 'q' to quit.")
    
    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            # Preprocess the frame
            image, original_frame = preprocess_frame(frame, input_size)
            
            # Run inference
            common.set_input(interpreter, np.expand_dims(np.array(image), axis=0))
            interpreter.invoke()
            
            # Get detection results
            detections = detect.get_objects(interpreter)
            
            # Filter results by threshold
            detections = [det for det in detections if det.score >= args.threshold]
            
            # Update FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Draw detection results on the frame
            result_frame = draw_detection_results(frame, detections, labels, input_size, fps, args.threshold)
            
            # Display the resulting frame
            cv2.imshow('Coral TPU Object Detection', result_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == '__main__':
    main() 