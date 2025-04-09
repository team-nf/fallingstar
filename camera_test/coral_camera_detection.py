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

# Import tracking module
from tracking import Sort, draw_tracks

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

def draw_detection_results(frame, detections, labels, input_size, tracked_objects=None, fps=0, threshold=0.3):
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
        
        # Find matching tracked object if tracking is enabled
        track_id = None
        if tracked_objects is not None:
            # Look for matching tracked object by IoU
            max_iou = 0
            for trk in tracked_objects:
                tx1, ty1, tx2, ty2, tid = trk
                # Calculate IoU with current detection
                xx1 = max(xmin, tx1)
                yy1 = max(ymin, ty1)
                xx2 = min(xmax, tx2)
                yy2 = min(ymax, ty2)
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                intersection = w * h
                det_area = (xmax - xmin) * (ymax - ymin)
                trk_area = (tx2 - tx1) * (ty2 - ty1)
                union = det_area + trk_area - intersection
                iou = intersection / union if union > 0 else 0
                if iou > max_iou and iou > 0.5:
                    max_iou = iou
                    track_id = int(tid)
        
        # Choose a color based on track_id if available, otherwise use class ID
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        if track_id is not None:
            color = (track_id * 50 % 255, track_id * 120 % 255, track_id * 220 % 255)
        else:
            color = colors[det.id % len(colors)]
        
        # Draw bounding box with OpenCV
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
        
        # Draw a circle at the center of the bounding box
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        
        # Get class label
        class_id = det.id
        score = det.score
        
        # Prepare label text
        if track_id is not None:
            label_text = f"{labels.get(class_id, class_id)}: {score:.2f} (ID: {track_id})"
        else:
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
    parser.add_argument('--trail_duration', type=float, default=3.0,
                        help='Duration of object trail in seconds (default: 3.0)')
    
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
    
    # Initialize the tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    color_map = {}  # To store colors for each track ID
    
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
            
            # Convert detections to format for tracking
            if detections:
                tracking_dets = np.array([
                    [
                        det.bbox[0] * frame.shape[1] / input_size[0],  # xmin scaled to frame
                        det.bbox[1] * frame.shape[0] / input_size[1],  # ymin scaled to frame
                        det.bbox[2] * frame.shape[1] / input_size[0],  # xmax scaled to frame
                        det.bbox[3] * frame.shape[0] / input_size[1],  # ymax scaled to frame
                        det.score                                       # detection score
                    ] 
                    for det in detections
                ])
                
                # Update tracker with new detections
                tracked_objects = tracker.update(tracking_dets)
            else:
                tracking_dets = np.empty((0, 5))
                tracked_objects = tracker.update()
            
            # Draw tracking trails first (so they're under the bounding boxes)
            frame, color_map = draw_tracks(frame, tracker, max_age=args.trail_duration, color_map=color_map)
            
            # Update FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Draw detection results on the frame
            result_frame = draw_detection_results(frame, detections, labels, input_size, 
                                                  tracked_objects if len(tracked_objects) > 0 else None, 
                                                  fps, args.threshold)
            
            # Display the resulting frame
            cv2.imshow('Coral TPU Object Detection with Tracking', result_frame)
            
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