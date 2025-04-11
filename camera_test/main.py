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
from src.contour import ContourDetector  # Import for contour detection

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
                       help='Legacy calibration factor (default: 0.833 for 100cm actual vs 120cm measured)')
    parser.add_argument('--calibration_x', type=float,
                       default=1.25,
                       help='X-axis calibration factor (default: 1.25 to correct 40cm to 50cm)')
    parser.add_argument('--calibration_y', type=float,
                       default=1.25,
                       help='Y-axis calibration factor (default: 1.25 to correct 40cm to 50cm)')
    parser.add_argument('--calibration_z', type=float,
                       default=1.0,
                       help='Z-axis/depth calibration factor (default: 1.0, uses calibration_factor)')
    parser.add_argument('--disable_contours', action='store_true',
                       help='Disable contour detection for objects (enabled by default)')
    parser.add_argument('--canny_low', type=int, 
                       default=50,
                       help='Lower threshold for Canny edge detector (default: 50)')
    parser.add_argument('--canny_high', type=int,
                       default=150,
                       help='Upper threshold for Canny edge detector (default: 150)')
    parser.add_argument('--use_adaptive_threshold', action='store_true',
                       help='Use adaptive thresholding instead of Canny edge detection')
    
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
    
    # Initialize contour detector if enabled
    contour_detector = None
    if not args.disable_contours:
        contour_detector = ContourDetector(
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            use_adaptive_threshold=args.use_adaptive_threshold
        )
        print("Contour detection enabled.")
        if args.use_adaptive_threshold:
            print("Using adaptive thresholding for contour detection.")
        else:
            print(f"Using Canny edge detection with thresholds: {args.canny_low}-{args.canny_high}.")
    
    print(f"Detection threshold: {args.threshold}")
    print(f"Trail duration: {args.trail_duration} seconds")
    print(f"Algae diameter: {args.algae_diameter} mm")
    print(f"Calibration factor (Z): {args.calibration_factor} (100cm/120cm = 0.833)")
    print(f"Calibration X-axis: {args.calibration_x}")
    print(f"Calibration Y-axis: {args.calibration_y}")
    print(f"Calibration Z-axis: {args.calibration_z}")
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
                algae_diameter_mm=args.algae_diameter,
                calibration_factor=args.calibration_factor,
                calibration_x=args.calibration_x,
                calibration_y=args.calibration_y,
                calibration_z=args.calibration_z
            )
            
            # Update tracker with detections
            tracked_objects = tracker.update(detections)
            
            # Draw tracking trails
            frame = tracker.draw_trails(frame, max_age=args.trail_duration)
            
            # Draw detection results
            frame = camera.draw_detections(frame, tracked_objects)
            
            # Detect contours if enabled
            if contour_detector and tracked_objects:
                contour_detector.detect_contours(frame, tracked_objects)
            
            # Display the result
            camera.display(frame, window_name='Coral and Algae Detection with Tracking')
            
            # Check for key press
            if camera.check_key() == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Close camera and destroy windows
        camera.close()
        
        # Close contour detector if initialized
        if contour_detector:
            contour_detector.close()
            
        print("Camera released and windows closed.")

if __name__ == '__main__':
    main() 