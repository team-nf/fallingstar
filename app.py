import cv2
import supervision as sv
from roboflow import Roboflow
import time
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run FRC 2025 Algae and Coral detection on camera feed')
parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
args = parser.parse_args()

# Initialize Roboflow API
rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
project = rf.workspace("frc-team-503-frog-force").project("frc-2025-algae-and-coral")
model = project.version(1).model

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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Get the frame dimensions
    height, width = frame.shape[:2]
    
    # Make prediction
    result = model.predict(frame, confidence=args.confidence).json()
    
    # Process predictions
    detections = sv.Detections.from_roboflow(result)
    
    # If detections exist, annotate the frame
    if len(detections) > 0:
        # Format labels
        labels = []
        for i, (_, confidence, class_id) in enumerate(zip(
                detections.xyxy, 
                detections.confidence, 
                detections.class_id
            )):
            label = f"{result['predictions'][i]['class']} {confidence:.2f}"
            labels.append(label)
        
        # Annotate and display the frame
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    
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
    
    # Add a small delay to reduce CPU usage
    time.sleep(0.01)

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows() 