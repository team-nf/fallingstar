#!/usr/bin/env python3

"""
Main application for coral and algae detection with tracking
"""

import os
import argparse
import time
from typing import Dict, List, Optional, Tuple, Any

from src.camera import Camera
from src.model import ObjectDetector
from src.tracking import ObjectTracker
from src.objects import DetectedObject, Coral, Algae
from src.pnp import PnPEstimator  # Correct import for the PnP algorithm

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Real-time coral and algae detection with tracking using Google Coral TPU'
    )
    
    parser.add_argument('--model', type=str, 
                       default='models/ll/GOOGLECORAL_coral_and_algae_monochrome.tflite',
                       help='Path to the TFLite model')
    parser.add_argument('--labels', type=str, 
                       default='models/ll/labels.txt',
                       help='Path to the labels file')
    parser.add_argument('--threshold', type=float, 
                       default=0.3,
                       help='Detection confidence threshold')
    parser.add_argument('--camera', type=int, 
                       default=0,
                       help='Camera device number (default: 0)')
    parser.add_argument('--width', type=int, 
                       default=640,
                       help='Camera feed width (default: 640)')
    parser.add_argument('--height', type=int, 
                       default=480,
                       help='Camera feed height (default: 480)')
    parser.add_argument('--trail_duration', type=float, 
                       default=3.0,
                       help='Duration of object trail in seconds (default: 3.0)')
    parser.add_argument('--algae_diameter', type=float, 
                       default=413.0,
                       help='Diameter of algae object in millimeters (default: 413.0)')
    parser.add_argument('--calibration_factor', type=float,
                       default=0.833,
                       help='PnP distance calibration factor (default: 0.833 for 100cm actual vs 120cm measured)')
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize detector
    try:
        detector = ObjectDetector(
            model_path=args.model,
            labels_path=args.labels
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Initialize camera
    camera = Camera(
        camera_id=args.camera,
        width=args.width,
        height=args.height
    )
    
    if not camera.open():
        print("Error: Could not open camera.")
        return
    
    # Initialize tracker
    tracker = ObjectTracker(
        max_age=30,  # Maximum number of frames to keep a track alive
        min_hits=3,  # Minimum number of hits to start tracking
        iou_threshold=0.3  # IoU threshold for matching
    )
    
    print(f"Detection threshold: {args.threshold}")
    print(f"Trail duration: {args.trail_duration} seconds")
    print(f"Algae diameter: {args.algae_diameter} mm")
    print(f"Calibration factor: {args.calibration_factor} (100cm/120cm = 0.833)")
    print("Starting detection. Press 'q' to quit.")
    
    # Main loop
    try:
        while True:
            # Read a frame from the camera
            frame = camera.read_frame()
            if frame is None:
                print("Error: Failed to capture image.")
                break
            
            # Perform detection
            detections = detector.detect(
                image=frame,
                threshold=args.threshold,
                algae_diameter_mm=args.algae_diameter,  # Pass algae diameter to create algae objects
                calibration_factor=args.calibration_factor  # Pass calibration factor to adjust distances
            )
            
            # Update tracker with detections
            tracked_objects = tracker.update(detections)
            
            # Draw tracking trails
            frame = tracker.draw_trails(frame, max_age=args.trail_duration)
            
            # Draw detection results
            frame = camera.draw_detections(frame, tracked_objects)
            
            # Display the result
            camera.display(frame, window_name='Coral and Algae Detection with Tracking')
            
            # Check for key press
            if camera.check_key() == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Close camera and destroy windows
        camera.close()
        print("Camera released and windows closed.")

if __name__ == '__main__':
    main() 