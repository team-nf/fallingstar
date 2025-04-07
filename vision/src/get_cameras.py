#!/usr/bin/env python3
import json
import cv2
import os
import platform

def get_available_cameras():
    """
    Detects and lists available camera devices.
    Returns a list of available camera indices and names if possible.
    """
    available_cameras = []
    
    # Platform-specific camera detection
    system = platform.system()
    
    if system == "Linux":
        # For Linux, check /dev/video* devices
        video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
        
        for device in sorted(video_devices):
            index = int(device.replace('video', ''))
            try:
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
                    else:
                        available_cameras.append({
                            "id": index,
                            "name": f"Camera {index} (no frame)",
                            "resolution": "unknown"
                        })
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {index}: {str(e)}")
    
    elif system == "Windows":
        # For Windows, try indices 0-9
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
    
    return available_cameras

if __name__ == "__main__":
    cameras = get_available_cameras()
    print(json.dumps(cameras, indent=2)) 