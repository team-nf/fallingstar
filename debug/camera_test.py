#!/usr/bin/env python3
"""
Simple camera test utility to verify camera connection and operation.
"""

import cv2
import argparse
import time
import sys
import os

def list_cameras():
    """List all available camera devices."""
    available_cameras = []
    for i in range(10):  # Check camera indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            available_cameras.append((i, width, height, fps))
            cap.release()
    return available_cameras

def test_camera(camera_id, width=640, height=480, fps=30):
    """Test opening and displaying video from a camera."""
    print(f"Opening camera {camera_id} at {width}x{height} @{fps}fps...")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Failed to open camera {camera_id}")
        return False
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Get actual properties
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera properties:")
    print(f"  Resolution: {actual_width}x{actual_height} (requested: {width}x{height})")
    print(f"  FPS: {actual_fps} (requested: {fps})")
    
    # Display camera feed
    print("Displaying camera feed (press 'q' to quit)...")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera")
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                current_fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Camera Test", frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera test completed")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test camera connection and display")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--list", action="store_true", help="List available cameras")
    
    args = parser.parse_args()
    
    if args.list:
        print("Scanning for available cameras...")
        cameras = list_cameras()
        if cameras:
            print(f"Found {len(cameras)} camera(s):")
            for cam_id, width, height, fps in cameras:
                print(f"  Camera {cam_id}: {width}x{height} @{fps}fps")
        else:
            print("No cameras found")
        return
    
    test_camera(args.camera_id, args.width, args.height, args.fps)

if __name__ == "__main__":
    main() 