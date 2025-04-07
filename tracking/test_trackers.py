#!/usr/bin/env python3
"""
Test script for comparing different tracking algorithms.

This script loads a video (or camera) and demonstrates all the
tracking algorithms side-by-side for visual comparison.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracking import (
    BoundingBox,
    TrackedObject,
    IoUTracker,
    KalmanTracker,
    OpenCVTracker,
    SORTTracker,
    visualize_tracked_objects
)

# Try to import detector if available
try:
    from detection.detector import ObjectDetector
    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False
    print("Warning: detection.detector not found, will use manual detection")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test tracking algorithms")
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Path to video file")
    input_group.add_argument("--camera", type=int, help="Camera device ID")
    
    # Tracking options
    parser.add_argument("--trackers", default="iou,kalman,opencv,sort", 
                      help="Comma-separated list of trackers to test")
    parser.add_argument("--detection-interval", type=int, default=5,
                      help="Run detection every N frames (0 = run on all frames)")
    
    # Detection options
    parser.add_argument("--model", help="Path to detection model (if not using detector package)")
    parser.add_argument("--labels", help="Path to labels file (if not using detector package)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                      help="Detection confidence threshold")
    
    # Display options
    parser.add_argument("--width", type=int, default=1280, help="Display width")
    parser.add_argument("--height", type=int, default=720, help="Display height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--no-display", action="store_true", help="Don't show GUI")
    parser.add_argument("--output", help="Save output to video file")
    
    return parser.parse_args()

def create_trackers(tracker_names: List[str], detection_interval: int) -> Dict[str, Any]:
    """
    Create requested trackers.
    
    Args:
        tracker_names: List of tracker names to create
        detection_interval: Detection interval for trackers that support it
        
    Returns:
        Dictionary of tracker objects
    """
    trackers = {}
    
    for name in tracker_names:
        name = name.strip().lower()
        
        if name == "iou":
            trackers["IoU"] = IoUTracker(max_lost=10, iou_threshold=0.3)
        elif name == "kalman":
            trackers["Kalman"] = KalmanTracker(max_lost=15, iou_threshold=0.3, max_prediction_steps=10)
        elif name == "opencv":
            trackers["OpenCV"] = OpenCVTracker(max_lost=30, tracker_type="CSRT", 
                                           detection_interval=detection_interval, iou_threshold=0.3)
        elif name == "sort":
            trackers["SORT"] = SORTTracker(max_lost=10, iou_threshold=0.3, max_age=15, min_hits=3)
        else:
            print(f"Warning: Unknown tracker '{name}'")
    
    return trackers

def create_detector(model_path: Optional[str] = None, labels_path: Optional[str] = None) -> Any:
    """
    Create detector object.
    
    Args:
        model_path: Path to detection model
        labels_path: Path to labels file
        
    Returns:
        Detector object
    """
    if HAS_DETECTOR:
        from detection.detector import ObjectDetector
        return ObjectDetector(model_path, labels_path)
    else:
        print("Detection package not found, no detector created")
        return None

def manual_detect(frame: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Manually detect objects in a frame using mouse selection.
    
    Args:
        frame: Input frame
        threshold: Detection threshold (not used for manual detection)
        
    Returns:
        List of detection dictionaries
    """
    global manual_detections
    
    # Return existing detections if already created
    if hasattr(manual_detect, "detections"):
        return manual_detect.detections
    
    # Create a copy of the frame for drawing
    display = frame.copy()
    
    # Create a window for displaying the frame
    window_name = "Manual Detection (draw boxes, press Enter when done)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Variables for drawing rectangles
    manual_detections = []
    drawing = False
    start_point = (-1, -1)
    end_point = (-1, -1)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, display
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # Draw rectangle on the display
            display = frame.copy()
            cv2.rectangle(display, start_point, (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            drawing = False
            
            # Create a detection from the drawn rectangle
            xmin = min(start_point[0], end_point[0])
            ymin = min(start_point[1], end_point[1])
            xmax = max(start_point[0], end_point[0])
            ymax = max(start_point[1], end_point[1])
            
            # Only add if rectangle has some size
            if xmax - xmin > 10 and ymax - ymin > 10:
                detection = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "class_id": 0,
                    "class_name": "object",
                    "score": 1.0
                }
                manual_detections.append(detection)
                
                # Draw permanent rectangle with ID
                cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(display, f"Obj {len(manual_detections)}", (xmin, ymin - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Set mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Main loop for manual detection
    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            break
        elif key == 27:  # Escape key
            manual_detections = []
            break
        elif key == ord('c'):  # Clear detections
            manual_detections = []
            display = frame.copy()
    
    cv2.destroyWindow(window_name)
    
    # Store detections for future frames
    manual_detect.detections = manual_detections
    
    return manual_detections

def run_tracking_test(args):
    """
    Run tracking test with the specified parameters.
    
    Args:
        args: Command line arguments
    """
    # Create video capture object
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video source: {frame_width}x{frame_height} @ {fps} FPS")
    
    # Create trackers
    tracker_names = args.trackers.split(",")
    trackers = create_trackers(tracker_names, args.detection_interval)
    
    # Create detector
    detector = create_detector(args.model, args.labels)
    
    # Create output video writer if requested
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    is_paused = False
    
    # Main loop
    while True:
        # Read a frame
        if not is_paused:
            ret, frame = cap.read()
            
            if not ret:
                if args.video:
                    print("End of video")
                    break
                else:
                    print("Error reading from camera")
                    break
            
            frame_count += 1
        
        # Determine if we should run detection on this frame
        run_detection = False
        if args.detection_interval == 0:
            run_detection = True
        elif frame_count % args.detection_interval == 0:
            run_detection = True
        
        # Run detection if needed
        detections = None
        if run_detection:
            if detector:
                # Run detection using detector
                try:
                    detections = detector.detect(frame, threshold=args.threshold)
                    
                    # Convert to dictionary format
                    detections = [
                        {
                            "bbox": [det.bbox.xmin, det.bbox.ymin, det.bbox.xmax, det.bbox.ymax],
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "score": det.score
                        }
                        for det in detections
                    ]
                except Exception as e:
                    print(f"Error during detection: {e}")
                    detections = None
            else:
                # Use manual detection
                detections = manual_detect(frame, args.threshold)
        
        # Update trackers with new frame and detections
        tracked_results = {}
        for name, tracker in trackers.items():
            try:
                tracked_objects = tracker.update(frame, detections)
                tracked_results[name] = tracked_objects
            except Exception as e:
                print(f"Error updating tracker {name}: {e}")
                tracked_results[name] = []
        
        # Visualize results
        if not args.no_display or writer:
            # Create composite image for all trackers
            result_frame = frame.copy()
            
            # Add tracker name and frame info
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add detection info if available
            if detections:
                det_count = len(detections)
                cv2.putText(result_frame, f"Detections: {det_count}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw tracks for each tracker
            y_offset = 110
            for name, objects in tracked_results.items():
                # Draw tracker name
                cv2.putText(result_frame, f"{name} Tracker: {len(objects)} objects", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 40
                
                # Draw tracked objects
                result_frame = visualize_tracked_objects(result_frame, objects, 
                                                      show_id=True, show_velocity=True)
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (frame_width - 150, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Write frame to output video if requested
            if writer:
                writer.write(result_frame)
            
            # Display the result
            if not args.no_display:
                cv2.imshow("Tracking Test", result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # Escape
                    break
                elif key == ord(' '):  # Space
                    is_paused = not is_paused
                elif key == ord('r'):  # Reset trackers
                    for tracker in trackers.values():
                        tracker.clear()
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    args = parse_args()
    run_tracking_test(args) 