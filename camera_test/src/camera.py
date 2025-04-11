#!/usr/bin/env python3

"""
Camera handling and visualization for object detection
"""

import cv2
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from src.objects import DetectedObject

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
    
    def draw_detections(self, frame: np.ndarray, detections: List['DetectedObject']) -> np.ndarray:
        """
        Draw detection results on the frame
        
        Args:
            frame: Input frame
            detections: List of DetectedObject instances
            
        Returns:
            Frame with detection results drawn
        """
        # Make a copy of the frame to avoid modifying the original
        result_frame = frame.copy()
        
        # Draw each detection
        for obj in detections:
            # Get bounding box
            x1, y1, x2, y2 = obj.bbox
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get color for this object
            color = obj.get_color()
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Get label text
            label = obj.get_label_text()
            
            # Draw text with semi-transparent background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(
                result_frame,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0] + 10, y1),
                (0, 0, 0, 128),
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            # Draw 3D position information if available
            if hasattr(obj, 'get_position_info') and obj.get_position_info() and obj.get_position_info()['success']:
                position_mm = obj.get_position_info()['position_mm']
                
                # Convert to meters for display
                pos_x, pos_y, pos_z = position_mm[0] / 1000.0, position_mm[1] / 1000.0, position_mm[2] / 1000.0
                
                # Format position text
                pos_text = f"Pos: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})m"
                
                # Draw background rectangle for position text
                text_size, _ = cv2.getTextSize(pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    result_frame,
                    (x1, y2),
                    (x1 + text_size[0] + 10, y2 + text_size[1] + 10),
                    (0, 0, 0, 128),
                    -1
                )
                
                # Draw position text
                cv2.putText(
                    result_frame,
                    pos_text,
                    (x1 + 5, y2 + text_size[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        return result_frame
    
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