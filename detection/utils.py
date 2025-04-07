#!/usr/bin/env python3

import os
import cv2
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import threading

def load_calibration(calibration_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load camera calibration from a file.
    
    Args:
        calibration_file: Path to calibration file (JSON or XML)
        
    Returns:
        Tuple of (camera_matrix, dist_coeffs) or (None, None) if loading fails
    """
    if not os.path.exists(calibration_file):
        print(f"Calibration file not found: {calibration_file}")
        return None, None
    
    try:
        # Determine file type
        if calibration_file.endswith('.json'):
            # Load from JSON
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                camera_matrix = np.array(data['camera_matrix'])
                dist_coeffs = np.array(data['dist_coeffs'])
        elif calibration_file.endswith('.xml'):
            # Load from XML
            fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode('camera_matrix').mat()
            dist_coeffs = fs.getNode('dist_coeffs').mat()
            fs.release()
        else:
            print(f"Unsupported calibration file type: {calibration_file}")
            return None, None
        
        print(f"Loaded camera calibration from {calibration_file}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None, None

def undistort_image(img: np.ndarray, camera_matrix: np.ndarray, 
                    dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort an image using camera calibration.
    
    Args:
        img: Input image
        camera_matrix: Camera matrix from calibration
        dist_coeffs: Distortion coefficients from calibration
        
    Returns:
        Undistorted image
    """
    h, w = img.shape[:2]
    
    # Get optimal new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Undistort
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop the image (optional)
    x, y, w, h = roi
    if all(v > 0 for v in [x, y, w, h]):
        undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def draw_detection_overlay(image: np.ndarray, 
                         tracked_objects: Dict[int, Dict[str, Any]],
                         show_labels: bool = True,
                         show_confidence: bool = True,
                         show_fps: bool = True,
                         fps_value: float = 0.0) -> np.ndarray:
    """
    Draw detection results on the image.
    
    Args:
        image: Input image
        tracked_objects: Dictionary of tracked objects
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        show_fps: Whether to show FPS counter
        fps_value: Current FPS value
        
    Returns:
        Image with detection overlay
    """
    # Create a copy of the image
    overlay = image.copy()
    
    # Draw each tracked object
    for track_id, obj in tracked_objects.items():
        # Get bounding box
        bbox = obj.get('bbox', None)
        if not bbox:
            continue
            
        # Get coordinates
        x, y, w, h = bbox.get_coords()
        
        # Get class information
        class_id = bbox.class_id
        class_name = obj.get('class_name', f"Class {class_id}")
        confidence = bbox.confidence
        
        # Choose color based on track_id or class_id
        color_options = [
            (0, 255, 0),   # Green
            (255, 0, 0),   # Blue
            (0, 0, 255),   # Red
            (0, 255, 255), # Yellow
            (255, 0, 255), # Magenta
            (255, 255, 0)  # Cyan
        ]
        color = color_options[track_id % len(color_options)]
        
        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        if show_labels:
            label_text = f"{class_name}"
            if show_confidence:
                label_text += f" ({confidence:.2f})"
            label_text += f" ID:{track_id}"
            
            # Get text size
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for text
            cv2.rectangle(
                overlay, 
                (x, y - text_size[1] - 5), 
                (x + text_size[0], y), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                overlay, 
                label_text, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
    
    # Draw FPS
    if show_fps and fps_value > 0:
        cv2.putText(
            overlay,
            f"FPS: {fps_value:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    
    return overlay

def fps_counter(smoothing: float = 0.9):
    """
    Create an FPS counter with smoothing.
    
    Args:
        smoothing: Smoothing factor (0-1, higher means more smoothing)
        
    Returns:
        Dictionary with update and get_fps functions
    """
    fps_data = {
        'last_time': time.time(),
        'fps': 0.0,
        'smoothing': smoothing
    }
    
    def update():
        """Update FPS calculation."""
        current_time = time.time()
        delta = current_time - fps_data['last_time']
        fps_data['last_time'] = current_time
        
        if delta > 0:
            current_fps = 1.0 / delta
            
            # Apply smoothing
            if fps_data['fps'] > 0:
                fps_data['fps'] = (fps_data['smoothing'] * fps_data['fps'] + 
                                  (1.0 - fps_data['smoothing']) * current_fps)
            else:
                fps_data['fps'] = current_fps
    
    def get_fps():
        """Get current FPS value."""
        return fps_data['fps']
    
    return {
        'update': update,
        'get_fps': get_fps
    }

def find_latest_calibration_file(directory: str = './calibration') -> Optional[str]:
    """
    Find the latest camera calibration file in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        Path to latest calibration file or None if not found
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return None
    
    calibration_files = []
    
    # Find all calibration files
    for file in os.listdir(directory):
        if file.startswith('camera_calibration_') and (file.endswith('.json') or file.endswith('.xml')):
            path = os.path.join(directory, file)
            calibration_files.append((path, os.path.getmtime(path)))
    
    if not calibration_files:
        return None
    
    # Sort by modification time (newest first)
    calibration_files.sort(key=lambda x: x[1], reverse=True)
    
    return calibration_files[0][0]

def threaded_camera_stream(camera_id: int = 0, resolution: Tuple[int, int] = (640, 480), 
                         fps: int = 30) -> Dict[str, Any]:
    """
    Create a threaded camera stream for more efficient frame grabbing.
    
    Args:
        camera_id: Camera device ID
        resolution: Resolution (width, height)
        fps: Target FPS
        
    Returns:
        Dictionary with functions to access the stream
    """
    stream_data = {
        'camera_id': camera_id,
        'resolution': resolution,
        'fps': fps,
        'running': False,
        'thread': None,
        'lock': threading.Lock(),
        'frame': None,
        'last_frame_time': 0,
        'error': None
    }
    
    def update_thread():
        """Thread function for grabbing frames."""
        # Initialize camera
        cap = cv2.VideoCapture(stream_data['camera_id'])
        
        if not cap.isOpened():
            stream_data['error'] = f"Could not open camera {stream_data['camera_id']}"
            stream_data['running'] = False
            return
        
        # Set camera properties
        width, height = stream_data['resolution']
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, stream_data['fps'])
        
        # Grab frames
        while stream_data['running']:
            ret, frame = cap.read()
            
            if not ret:
                stream_data['error'] = "Failed to grab frame"
                continue
            
            with stream_data['lock']:
                stream_data['frame'] = frame
                stream_data['last_frame_time'] = time.time()
        
        # Release resources
        cap.release()
    
    def start():
        """Start the camera stream."""
        if stream_data['running']:
            return
            
        stream_data['running'] = True
        stream_data['error'] = None
        stream_data['thread'] = threading.Thread(target=update_thread, daemon=True)
        stream_data['thread'].start()
    
    def stop():
        """Stop the camera stream."""
        stream_data['running'] = False
        
        if stream_data['thread']:
            stream_data['thread'].join(timeout=1.0)
            stream_data['thread'] = None
    
    def get_frame():
        """Get the latest frame from the stream."""
        with stream_data['lock']:
            if stream_data['frame'] is None:
                return None, "No frame available"
                
            return stream_data['frame'].copy(), None
    
    def is_running():
        """Check if the stream is running."""
        return stream_data['running']
    
    def get_error():
        """Get the latest error message."""
        return stream_data['error']
    
    return {
        'start': start,
        'stop': stop,
        'get_frame': get_frame,
        'is_running': is_running,
        'get_error': get_error
    }

def get_camera_list() -> List[int]:
    """
    Get a list of available camera devices.
    
    Returns:
        List of available camera device IDs
    """
    available_cameras = []
    
    # Try up to 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras

def save_diagnostic_image(image: np.ndarray, prefix: str = "diagnostic", 
                        directory: str = "./diagnostics"):
    """
    Save a diagnostic image with timestamp.
    
    Args:
        image: Image to save
        prefix: Filename prefix
        directory: Directory to save image
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    
    cv2.imwrite(filepath, image)
    print(f"Saved diagnostic image to {filepath}")

def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    Clamp a value between min and max.
    
    Args:
        value: Input value
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))

def normalize_bbox(bbox: Tuple[int, int, int, int], image_width: int, 
                 image_height: int) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box coordinates to [0, 1] range.
    
    Args:
        bbox: Bounding box (x, y, w, h)
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Normalized bounding box (x, y, w, h)
    """
    x, y, w, h = bbox
    return (
        clamp(x / image_width),
        clamp(y / image_height),
        clamp(w / image_width),
        clamp(h / image_height)
    )

def unnormalize_bbox(bbox: Tuple[float, float, float, float], image_width: int, 
                   image_height: int) -> Tuple[int, int, int, int]:
    """
    Convert normalized bounding box coordinates to pixel coordinates.
    
    Args:
        bbox: Normalized bounding box (x, y, w, h)
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Pixel coordinates bounding box (x, y, w, h)
    """
    x, y, w, h = bbox
    return (
        int(x * image_width),
        int(y * image_height),
        int(w * image_width),
        int(h * image_height)
    ) 