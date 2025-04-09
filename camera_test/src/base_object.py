#!/usr/bin/env python3

"""
Base class for detected objects
"""

import time
from typing import Tuple, List, Dict, Optional

class DetectedObject:
    """Base class for detected objects"""
    
    def __init__(self, 
                bbox: Tuple[float, float, float, float], 
                score: float, 
                track_id: Optional[int] = None):
        """
        Initialize a detected object
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            score: Detection confidence score
            track_id: Tracking ID if available
        """
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.score = score
        self.track_id = track_id
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.detection_time = time.time()
    
    def get_width(self) -> float:
        """Get width of the bounding box"""
        return self.bbox[2] - self.bbox[0]
    
    def get_height(self) -> float:
        """Get height of the bounding box"""
        return self.bbox[3] - self.bbox[1]
    
    def get_area(self) -> float:
        """Get area of the bounding box"""
        return self.get_width() * self.get_height()
    
    def get_label_text(self) -> str:
        """Get the label text to display on the bounding box"""
        if self.track_id is not None:
            return f"{self.get_class_name()}: {self.score:.2f} (ID: {self.track_id})"
        else:
            return f"{self.get_class_name()}: {self.score:.2f}"
    
    def get_class_name(self) -> str:
        """Get the class name (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement get_class_name()")
        
    def get_color(self) -> Tuple[int, int, int]:
        """Get color for visualization (BGR format) (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement get_color()") 