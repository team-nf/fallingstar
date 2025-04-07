#!/usr/bin/env python3
import json
import cv2
import os
import platform
import subprocess
import numpy as np

def create_mock_camera():
    """Create a mock camera image with a test pattern"""
    # Create a test pattern image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a color gradient background
    for y in range(480):
        for x in range(640):
            img[y, x] = [
                int(255 * (x / 640)),
                int(255 * (y / 480)),
                int(255 * ((640-x) / 640))
            ]
    
    # Draw a white rectangle in the center
    cv2.rectangle(img, (270, 190), (370, 290), (255, 255, 255), 2)
    
    # Add text
    cv2.putText(img, "MOCK CAMERA", (220, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "No actual camera found", (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

def get_available_cameras():
    """
    Detects and lists available camera devices.
    Returns a list of available camera indices and names if possible.
    """
    available_cameras = []
    
    # Platform-specific camera detection
    system = platform.system()
    
    # Debug: Print system information
    print(f"System: {system}")
    
    if system == "Linux":
        # Check for v4l2 devices
        try:
            # Try to use v4l2-ctl if available
            v4l2_output = subprocess.check_output(['v4l2-ctl', '--list-devices'], 
                                                universal_newlines=True)
            print("v4l2-ctl output:", v4l2_output)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("v4l2-ctl not available")
            pass
            
        # For Linux, check /dev/video* devices
        print("Checking /dev for video devices")
        if os.path.exists('/dev'):
            video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
            print(f"Found devices: {video_devices}")
            
            for device in sorted(video_devices):
                index = int(device.replace('video', ''))
                try:
                    print(f"Trying to open camera {index}")
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        # Get camera info if available
                        ret, frame = cap.read()
                        if ret:
                            height, width = frame.shape[:2]
                            available_cameras.append({
                                "id": index,
                                "name": f"Camera {index}",
                                "resolution": f"{width}x{height}"
                            })
                            print(f"Successfully opened camera {index}: {width}x{height}")
                        else:
                            available_cameras.append({
                                "id": index,
                                "name": f"Camera {index} (no frame)",
                                "resolution": "unknown"
                            })
                            print(f"Opened camera {index} but couldn't read frame")
                        cap.release()
                    else:
                        print(f"Failed to open camera {index}")
                except Exception as e:
                    print(f"Error checking camera {index}: {str(e)}")
        
        # If no cameras found, try a fallback approach
        if not available_cameras:
            print("No cameras found via /dev/video*, trying fallback approach")
            for i in range(5):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            height, width = frame.shape[:2]
                            available_cameras.append({
                                "id": i,
                                "name": f"Camera {i}",
                                "resolution": f"{width}x{height}"
                            })
                            print(f"Fallback: Successfully opened camera {i}: {width}x{height}")
                        else:
                            available_cameras.append({
                                "id": i,
                                "name": f"Camera {i} (no frame)",
                                "resolution": "unknown"
                            })
                            print(f"Fallback: Opened camera {i} but couldn't read frame")
                        cap.release()
                except Exception as e:
                    print(f"Fallback: Error checking camera {i}: {str(e)}")
    
    elif system == "Windows":
        # For Windows, try indices 0-9
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        available_cameras.append({
                            "id": i,
                            "name": f"Camera {i}",
                            "resolution": f"{width}x{height}"
                        })
                    else:
                        available_cameras.append({
                            "id": i,
                            "name": f"Camera {i} (no frame)",
                            "resolution": "unknown"
                        })
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
    
    elif system == "Darwin":  # macOS
        # For macOS, try indices 0-9
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        available_cameras.append({
                            "id": i,
                            "name": f"Camera {i}",
                            "resolution": f"{width}x{height}"
                        })
                    else:
                        available_cameras.append({
                            "id": i,
                            "name": f"Camera {i} (no frame)",
                            "resolution": "unknown"
                        })
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
    
    # If no cameras were found, add a dummy entry for testing
    if not available_cameras:
        available_cameras.append({
            "id": 999,  # Using a special ID for the mock camera
            "name": "Mock Camera (no actual camera found)",
            "resolution": "640x480"
        })
        print("No physical cameras found, added mock camera")
    
    return available_cameras

if __name__ == "__main__":
    cameras = get_available_cameras()
    print(json.dumps(cameras, indent=2)) 