import os
import cv2
import numpy as np
import argparse
import time
from PIL import Image
import matplotlib.pyplot as plt

# Import Edge TPU libraries
# The tflite_runtime package is specific to Edge TPU
try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
    from pycoral.adapters import classify
    CORAL_AVAILABLE = True
except ImportError:
    print("Warning: PyCoral library not found. Running without Edge TPU acceleration.")
    import tflite_runtime.interpreter as tflite
    CORAL_AVAILABLE = False

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test TFLite model on Google Coral')
parser.add_argument('--model_path', type=str, default='models/model_int8.tflite', help='Path to TFLite model')
parser.add_argument('--labels_path', type=str, default='models/labels.txt', help='Path to labels file')
parser.add_argument('--image_path', type=str, help='Path to test image')
parser.add_argument('--camera', type=int, default=0, help='Camera index for live testing')
parser.add_argument('--image_size', type=int, default=320, help='Input image size for model')
parser.add_argument('--use_camera', action='store_true', help='Use camera instead of test image')
parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
args = parser.parse_args()

# Load labels
with open(args.labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f"Loaded {len(labels)} labels: {labels}")

# Load model
print(f"Loading model from {args.model_path}")
if CORAL_AVAILABLE:
    print("Using Edge TPU acceleration")
    interpreter = edgetpu.make_interpreter(args.model_path)
else:
    print("Using CPU for inference")
    interpreter = tflite.Interpreter(model_path=args.model_path)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(f"Model input shape: {input_shape}")
print(f"Model output shape: {output_details[0]['shape']}")

# Check if the model is quantized
is_quantized = input_details[0]['dtype'] == np.uint8 or input_details[0]['dtype'] == np.int8
print(f"Model is{'nt' if not is_quantized else ''} quantized")

# Function to preprocess image
def preprocess_image(img, input_size):
    # Resize
    resized = cv2.resize(img, (input_size, input_size))
    
    # Convert to RGB
    if len(resized.shape) == 2:  # If grayscale
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:  # If RGBA
        resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
    elif resized.shape[2] == 3 and np.array_equal(resized[:,:,0], resized[:,:,2]):  # If BGR
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    if not is_quantized:
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    else:
        # Check input type
        input_type = input_details[0]['dtype']
        if input_type == np.int8:
            # Convert to int8 by normalizing and scaling to [-128, 127]
            # First normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            # Then scale to [-1, 1]
            normalized = normalized * 2.0 - 1.0
            # Finally scale to [-128, 127] and convert to int8
            quantized = (normalized * 127).astype(np.int8)
            return quantized
        else:
            # For uint8 quantization
            return resized.astype(np.uint8)

# Function to run inference
def run_inference(img):
    # Get original image dimensions
    original_height, original_width = img.shape[:2]
    
    # Preprocess image
    processed_img = preprocess_image(img, args.image_size)
    
    # Add batch dimension
    input_data = np.expand_dims(processed_img, axis=0)
    
    # Set input tensor
    if CORAL_AVAILABLE:
        common.set_input(interpreter, input_data)
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Get all output details
    all_output_details = interpreter.get_output_details()
    print(f"Number of outputs: {len(all_output_details)}")
    for i, output in enumerate(all_output_details):
        print(f"Output {i}: {output['name']}, shape: {output['shape']}, dtype: {output['dtype']}")
    
    # Get outputs based on TFLite model type
    if len(all_output_details) >= 4:
        # SSD/EfficientDet detection model with 4 outputs:
        # - Locations (bounding boxes)
        # - Classes
        # - Scores
        # - Number of detections
        boxes = interpreter.get_tensor(all_output_details[0]['index'])[0]
        classes = interpreter.get_tensor(all_output_details[1]['index'])[0]
        scores = interpreter.get_tensor(all_output_details[2]['index'])[0]
        num_detections = int(interpreter.get_tensor(all_output_details[3]['index'])[0])
        
        # Convert normalized coordinates [0,1] to pixel values
        height, width = img.shape[:2]
        boxes_scaled = []
        for box in boxes[:num_detections]:
            ymin, xmin, ymax, xmax = box
            boxes_scaled.append([
                int(xmin * width),    # x1
                int(ymin * height),   # y1
                int(xmax * width),    # x2
                int(ymax * height)    # y2
            ])
        
        # Return detection results
        detections = []
        for i in range(num_detections):
            if scores[i] >= args.threshold:
                class_id = int(classes[i])
                if class_id < len(labels):
                    label = labels[class_id]
                else:
                    label = f"Class {class_id}"
                detections.append({
                    'box': boxes_scaled[i],
                    'class': class_id,
                    'label': label,
                    'score': float(scores[i])
                })
        
        return detections, inference_time
    else:
        # Classification model or unknown format
        if len(all_output_details) >= 1:
            # Try to interpret as classification output
            output = interpreter.get_tensor(all_output_details[0]['index'])
            if len(output.shape) >= 2 and output.shape[1] <= len(labels):
                # Sigmoid for multi-label classification
                try:
                    predictions = 1 / (1 + np.exp(-output))
                    return predictions[0], inference_time
                except:
                    print("Warning: Failed to apply sigmoid, using raw outputs")
                    return output[0], inference_time
            else:
                print(f"Warning: Unexpected output shape: {output.shape}")
                return output, inference_time
        else:
            print("Warning: No output tensors found")
            return [], inference_time

# Display results for object detection
def display_results(img, detections, inference_time):
    # Create a copy for display
    display_img = img.copy()
    
    # Print results
    print(f"Inference time: {inference_time:.2f}s ({1/inference_time:.2f} FPS)")
    print(f"Detected {len(detections)} objects:")
    
    # Colors for different classes (BGR format)
    colors = [
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 0, 0),     # Blue
        (0, 255, 255),   # Yellow
        (255, 0, 255),   # Magenta
        (255, 255, 0)    # Cyan
    ]
    
    # Draw detections on image
    for i, detection in enumerate(detections):
        # Get detection info
        box = detection['box']
        label = detection['label']
        score = detection['score']
        class_id = detection['class'] % len(colors)  # Ensure color index is in range
        
        # Print detection details
        print(f"{label} ({score:.2f}): Box {box}")
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        color = colors[class_id]
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"{label}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1] + 5)
        cv2.rectangle(display_img, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - 5), color, -1)
        cv2.putText(display_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show FPS on image
    cv2.putText(display_img, f"FPS: {1/inference_time:.2f}", 
                (10, display_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2)
    
    return display_img

# Run on test image or camera
if args.use_camera:
    # Camera mode
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        exit(1)
    
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Run inference
        detections, inference_time = run_inference(frame)
        
        # Display results
        result_frame = display_results(frame, detections, inference_time)
        
        # Show FPS on image
        cv2.putText(result_frame, f"FPS: {1/inference_time:.2f}", 
                    (10, result_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Coral Edge TPU Inference', result_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
else:
    # Test image mode
    if args.image_path is None:
        print("Error: Please provide an image path with --image_path or use --use_camera")
        exit(1)
    
    # Load image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        exit(1)
    
    # Run inference
    detections, inference_time = run_inference(img)
    
    # Display results
    result_img = display_results(img, detections, inference_time)
    
    # Save image to file
    output_path = "result_image.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"Result image saved to {output_path}")
    
    # Try to show image if possible
    try:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Inference time: {inference_time:.4f}s ({1/inference_time:.2f} FPS)")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Could not display image: {e}")

print("Done!") 