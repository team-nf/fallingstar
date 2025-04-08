#!/usr/bin/env python3

"""
Test script for running SSD MobileNet v2 object detection using Google Coral TPU
"""

import os
import argparse
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Import Coral TPU libraries
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file

def load_labels(labels_file):
    """Loads labels from file"""
    with open(labels_file, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def preprocess_image(image_path, input_size):
    """Preprocess the image to meet SSD MobileNet v2 input requirements"""
    image = Image.open(image_path).convert('RGB')
    
    # Get original dimensions for scaling bounding boxes later
    orig_width, orig_height = image.size
    
    # Resize the image to the required input dimensions
    resized_image = image.resize(input_size, Image.LANCZOS)
    
    return resized_image, (orig_width, orig_height)

def visualize_results(image, detections, labels, threshold=0.3):
    """Visualize detection results on the image"""
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    print(f"Drawing {len(detections)} bounding boxes...")
    
    # Draw bounding boxes and labels
    for i, det in enumerate(detections):
        if det.score < threshold:
            continue
            
        # Get bounding box coordinates
        bbox = det.bbox
        xmin, ymin, xmax, ymax = bbox
        
        # Scale coordinates to match the displayed image size
        width, height = image.size
        x_scale = width / 300  # Model input is 300x300
        y_scale = height / 300
        
        xmin = max(0, int(xmin * x_scale))
        ymin = max(0, int(ymin * y_scale))
        xmax = min(width, int(xmax * x_scale))
        ymax = min(height, int(ymax * y_scale))
        
        # Ensure valid coordinates
        if xmin >= xmax or ymin >= ymax:
            print(f"Warning: Invalid bounding box coordinates: {xmin},{ymin},{xmax},{ymax}")
            continue
        
        print(f"Box {i+1}: ({xmin},{ymin}) to ({xmax},{ymax}) - {labels.get(det.id, det.id)}: {det.score:.2f}")
        
        # Choose a color based on class ID for better distinction
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        color = colors[det.id % len(colors)]
        
        # Draw bounding box with thicker line
        for offset in range(3):
            draw.rectangle(
                [(xmin-offset, ymin-offset), (xmax+offset, ymax+offset)], 
                outline=color
            )
        
        # Get class label
        class_id = det.id
        score = det.score
        
        label_text = f"{labels.get(class_id, class_id)}: {score:.2f}"
        
        # Draw background rectangle for text
        text_bg_height = 24
        text_bg_width = len(label_text) * 10  # Rough estimate
        draw.rectangle(
            [(xmin, ymin), (xmin + text_bg_width, ymin + text_bg_height)],
            fill=color
        )
        
        # Draw text
        draw.text((xmin + 2, ymin + 2), label_text, fill='white', font=font)
    
    return image_with_boxes

def run_detection(model_path, image_path, labels_path, threshold=0.3, save_output=True):
    """Run SSD MobileNet v2 detection on an image using Coral TPU"""
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load labels
    labels = {}
    if labels_path and os.path.exists(labels_path):
        labels = load_labels(labels_path)
    
    # Initialize the TPU with the detection model
    print("Loading SSD MobileNet v2 model on Edge TPU...")
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    _, input_height, input_width, _ = input_shape
    input_size = (input_width, input_height)  # (width, height)
    
    print(f"Model input shape: {input_shape}")
    
    # Preprocess the image
    print(f"Preprocessing image: {image_path}")
    image, (orig_width, orig_height) = preprocess_image(image_path, input_size)
    
    # Run inference
    print("Running detection...")
    start_time = time.time()
    common.set_input(interpreter, np.expand_dims(np.array(image), axis=0))
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000
    print(f"Inference time: {inference_time:.2f}ms")
    
    # Get detection results
    detections = detect.get_objects(interpreter)
    
    # Filter results by threshold
    detections = [det for det in detections if det.score >= threshold]
    
    print(f"\nFound {len(detections)} objects.")
    for i, det in enumerate(detections):
        label_name = labels.get(det.id, det.id)
        print(f"  {i+1}. {label_name}: {det.score:.2f} at {det.bbox}")
    
    # Visualize results
    if save_output:
        output_image = visualize_results(image, detections, labels, threshold)
        output_path = os.path.splitext(image_path)[0] + "_detection.jpg"
        output_image.save(output_path)
        print(f"Detection results saved to {output_path}")
    
    return detections

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run SSD MobileNet v2 detection on an image using Coral TPU')
    parser.add_argument('--model', type=str, default='models/ll/GOOGLECORAL_coral_and_algae_monochrome.tflite',
                        help='Path to the TFLite model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--labels', type=str, default='models/ll/labels.txt',
                        help='Path to the labels file')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the output image with detections')
    
    args = parser.parse_args()
    
    try:
        run_detection(
            args.model, 
            args.image, 
            args.labels,
            threshold=args.threshold,
            save_output=not args.no_save
        )
    except Exception as e:
        print(f"Error running detection: {e}")

if __name__ == '__main__':
    main() 