#!/usr/bin/env python3
"""
Object detection test utility for images.
Tests loading models and performing inference on image files.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import importlib.util
from PIL import Image

# Add parent directory to path for importing detection modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import detection utilities if available
try:
    from detection.detector import ObjectDetector
    from detection.utils import draw_bounding_boxes
    HAS_DETECTION = True
except ImportError:
    print("Warning: detection package not found in parent directory")
    HAS_DETECTION = False

def check_dependencies():
    """Check if required packages are installed."""
    dependencies = [
        ("PIL", "Pillow", "PIL.Image"),
        ("numpy", "numpy", "numpy"),
        ("cv2", "opencv-python", "cv2"),
        ("tflite_runtime", "tflite-runtime", "tflite_runtime.interpreter"),
        ("pycoral", "pycoral", "pycoral.utils.edgetpu")
    ]
    
    missing = []
    
    for package_name, pip_name, import_path in dependencies:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            missing.append(pip_name)
            print(f"Missing dependency: {package_name} (install with 'pip install {pip_name}')")
    
    return len(missing) == 0

def load_labels(labels_path):
    """Load labels from a text file."""
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found: {labels_path}")
        return None
    
    try:
        with open(labels_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None

def manual_detection(image_path, model_path, labels_path, threshold=0.5, edgetpu=True):
    """Run detection manually without using the detection package."""
    # Import necessary libraries
    try:
        from tflite_runtime.interpreter import Interpreter
        from tflite_runtime.interpreter import load_delegate
    except ImportError:
        print("Error: TFLite runtime not available")
        return None
    
    # Load TFLite model
    try:
        if edgetpu:
            # Use Edge TPU
            print(f"Loading EdgeTPU model: {model_path}")
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
        else:
            # Use CPU
            print(f"Loading CPU model: {model_path}")
            interpreter = Interpreter(model_path=model_path)
        
        interpreter.allocate_tensors()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = input_details[0]['dtype'] == np.float32
    
    print(f"Model input shape: {input_details[0]['shape']}")
    
    # Load image and prepare it for the model
    try:
        image = Image.open(image_path).convert('RGB')
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        input_data = np.expand_dims(image.resize((width, height)), axis=0)
        
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        elapsed = time.time() - start_time
        print(f"Inference time: {elapsed*1000:.1f}ms")
        
        # Get detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
        
        # Load labels
        labels = load_labels(labels_path) if labels_path else None
        
        # Filter detections based on threshold
        detections = []
        for i in range(num_detections):
            if scores[i] >= threshold:
                # Get class name from labels if available
                class_id = int(classes[i])
                class_name = labels[class_id] if labels and class_id < len(labels) else f"Class {class_id}"
                
                # Get bounding box
                box = boxes[i].tolist()
                ymin, xmin, ymax, xmax = box
                
                # Convert normalized coordinates to pixel values
                h, w, _ = image_rgb.shape
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)
                
                detections.append({
                    'class': class_id,
                    'class_name': class_name,
                    'confidence': float(scores[i]),
                    'bbox': [xmin, ymin, xmax, ymax]
                })
        
        return {
            'image': image_rgb,
            'detections': detections,
            'inference_time': elapsed
        }
    
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def draw_detections(image, detections):
    """Draw bounding boxes and labels on image."""
    result = image.copy()
    
    for detection in detections:
        # Get detection info
        box = detection['bbox']
        score = detection['confidence']
        label = f"{detection['class_name']}: {score:.2f}"
        
        # Draw bounding box
        cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (box[0], box[1] - label_size[1] - 5), 
                     (box[0] + label_size[0], box[1]), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(result, label, (box[0], box[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result

def test_detection_package(image_path, model_path, labels_path, threshold=0.5):
    """Run detection using the detection package."""
    if not HAS_DETECTION:
        print("Error: detection package not available")
        return None
    
    try:
        # Load detector
        print(f"Loading model using detection package: {model_path}")
        detector = ObjectDetector(model_path, labels_path)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(image, threshold)
        elapsed = time.time() - start_time
        
        print(f"Detection completed in {elapsed*1000:.1f}ms")
        print(f"Found {len(detections)} objects")
        
        # Format results
        results = []
        for detection in detections:
            results.append({
                'class': detection.class_id,
                'class_name': detection.class_name,
                'confidence': detection.score,
                'bbox': [detection.bbox.xmin, detection.bbox.ymin, 
                        detection.bbox.xmax, detection.bbox.ymax]
            })
        
        return {
            'image': image,
            'detections': results,
            'inference_time': elapsed
        }
    
    except Exception as e:
        print(f"Error during detection with package: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test object detection on images")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", required=True, help="Path to TFLite model")
    parser.add_argument("--labels", help="Path to labels file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of EdgeTPU")
    parser.add_argument("--no-package", action="store_true", 
                       help="Don't use detection package, run manual detection")
    parser.add_argument("--output", help="Path to save output image")
    parser.add_argument("--show", action="store_true", help="Show detection results")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("Missing required packages. Please install them and try again.")
        return
    
    # Check files
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Run detection
    if args.no_package or not HAS_DETECTION:
        result = manual_detection(args.image, args.model, args.labels, 
                                 args.threshold, not args.cpu)
    else:
        result = test_detection_package(args.image, args.model, args.labels, 
                                       args.threshold)
    
    if not result:
        print("Detection failed")
        return
    
    # Print results
    print("\nDetection Results:")
    for i, detection in enumerate(result['detections']):
        print(f"  {i+1}. {detection['class_name']} (Score: {detection['confidence']:.2f})")
        box = detection['bbox']
        print(f"     BBox: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    
    # Draw detections on image
    output_image = draw_detections(result['image'], result['detections'])
    
    # Add info text
    h, w = output_image.shape[:2]
    info_text = [
        f"Model: {os.path.basename(args.model)}",
        f"Detections: {len(result['detections'])}",
        f"Inference: {result['inference_time']*1000:.1f}ms"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(output_image, text, (10, 30 + 30*i),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Save output image if requested
    if args.output:
        cv2.imwrite(args.output, output_image)
        print(f"Output image saved to: {args.output}")
    
    # Show image if requested
    if args.show:
        cv2.imshow("Detection Results", output_image)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
