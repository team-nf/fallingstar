#!/usr/bin/env python3

"""
Object classes for Coral and Algae detections
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


class Coral(DetectedObject):
    """Class for Coral detections"""
    
    def __init__(self, bbox: Tuple[float, float, float, float], 
                 score: float, 
                 track_id: Optional[int] = None):
        """Initialize a Coral detection"""
        super().__init__(bbox, score, track_id)
    
    def get_class_name(self) -> str:
        """Get the class name"""
        return "Coral"
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get color for visualization (BGR format)"""
        if self.track_id is not None:
            # Generate a color based on track_id but biased toward red
            return (50, 50, 200 + (self.track_id * 20) % 55)
        else:
            return (50, 50, 255)  # Red-ish color for coral


class Algae(DetectedObject):
    """Class for Algae detections"""
    
    def __init__(self, bbox: Tuple[float, float, float, float], 
                 score: float, 
                 track_id: Optional[int] = None):
        """Initialize an Algae detection"""
        super().__init__(bbox, score, track_id)
    
    def get_class_name(self) -> str:
        """Get the class name"""
        return "Algae"
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get color for visualization (BGR format)"""
        if self.track_id is not None:
            # Generate a color based on track_id but biased toward green
            return (50, 200 + (self.track_id * 20) % 55, 50)
        else:
            return (50, 255, 50)  # Green-ish color for algae


def create_object_from_detection(detection_class_id: int, 
                                bbox: Tuple[float, float, float, float], 
                                score: float, 
                                track_id: Optional[int] = None) -> DetectedObject:
    """
    Factory function to create the appropriate DetectedObject subclass
    
    Args:
        detection_class_id: Class ID from the detector (0=algae, 1=coral)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        score: Detection confidence score
        track_id: Tracking ID if available
        
    Returns:
        An instance of the appropriate DetectedObject subclass
    """
    if detection_class_id == 0:
        return Algae(bbox, score, track_id)
    elif detection_class_id == 1:
        return Coral(bbox, score, track_id)
    else:
        # Generic object for unknown class IDs
        return DetectedObject(bbox, score, track_id) 