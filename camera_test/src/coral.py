#!/usr/bin/env python3

"""
Coral detection class
"""

from typing import Tuple, Optional
from src.base_object import DetectedObject

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