#!/usr/bin/env python3

"""
Camera handling and visualization for object detection
"""

import cv2
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from objects import DetectedObject

class Camera:
    """Class for camera handling and visualization"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize the camera
        
        Args:
            camera_id: Camera device number
            width: Camera feed width
            height: Camera feed height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        self.color_map = {}  # To store colors for tracked objects
    
    def open(self) -> bool:
        """
        Open the camera
        
        Returns:
            True if camera was opened successfully, False otherwise
        """
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Reset FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
        
        return self.cap.isOpened()
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera
        
        Returns:
            Frame as numpy array or None if reading failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Update FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[DetectedObject]) -> np.ndarray:
        """
        Draw detection results on the frame
        
        Args:
            frame: Input frame
            detections: List of DetectedObject instances
            
        Returns:
            Frame with detection visualizations
        """
        for obj in detections:
            # Get bounding box coordinates
            xmin, ymin, xmax, ymax = map(int, obj.bbox)
            
            # Get color
            color = obj.get_color()
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
            
            # Draw a circle at the center of the bounding box
            center_x, center_y = map(int, obj.center)
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw text background
            label_text = obj.get_label_text()
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (xmin, ymin - 25), (xmin + text_size[0], ymin), color, -1)
            
            # Draw text
            cv2.putText(frame, label_text, (xmin, ymin - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS info
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def display(self, frame: np.ndarray, window_name: str = "Object Detection") -> None:
        """
        Display a frame in a window
        
        Args:
            frame: Frame to display
            window_name: Name of the window
        """
        cv2.imshow(window_name, frame)
    
    def check_key(self, wait_time: int = 1) -> int:
        """
        Check if a key was pressed
        
        Args:
            wait_time: Time to wait for a key press in milliseconds
            
        Returns:
            Key code or -1 if no key was pressed
        """
        return cv2.waitKey(wait_time) & 0xFF
    
    def close(self) -> None:
        """Close the camera and destroy all windows"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows() 