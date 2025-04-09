#!/usr/bin/env python3

"""
Algae detection class with 3D position estimation
"""

from typing import Tuple, Optional, Dict
from src.base_object import DetectedObject
from src.pnp import PnPEstimator

class Algae(DetectedObject):
    """Class for Algae detections with 3D position estimation"""
    
    # Algae properties (mm)
    DIAMETER_MIN = 406.0
    DIAMETER_MAX = 419.0
    DIAMETER_AVG = (DIAMETER_MIN + DIAMETER_MAX) / 2
    
    def __init__(self, bbox: Tuple[float, float, float, float], 
                 score: float, 
                 track_id: Optional[int] = None,
                 diameter_mm: float = DIAMETER_AVG,
                 calibration_factor: float = 1.0):
        """
        Initialize an Algae detection
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            score: Detection confidence score
            track_id: Tracking ID if available
            diameter_mm: Diameter of algae object in millimeters
            calibration_factor: Factor to adjust PnP distance calculations
        """
        super().__init__(bbox, score, track_id)
        
        # Initialize PnP estimator for 3D position estimation
        self.diameter_mm = diameter_mm
        self.calibration_factor = calibration_factor
        self.pnp_estimator = PnPEstimator(calibration_factor=calibration_factor)
        
        # Calculate position if bounding box is approximately square
        self.position_info = None
        if self.pnp_estimator.is_approximately_square(bbox):
            self.position_info = self.pnp_estimator.estimate_position(bbox, self.diameter_mm)
    
    def get_class_name(self) -> str:
        """Get the class name"""
        return "Algae"
    
    def get_position_info(self) -> Optional[Dict]:
        """Get the position information if available"""
        return self.position_info
    
    def get_distance_str(self) -> Optional[str]:
        """Get the distance string if available"""
        if self.position_info and self.position_info['success']:
            return self.position_info['distance_str']
        return None
    
    def get_label_text(self) -> str:
        """Get the label text to display on the bounding box"""
        base_text = super().get_label_text()
        
        # Add distance information if available
        distance_str = self.get_distance_str()
        if distance_str:
            return f"{base_text} | Dist: {distance_str}"
        
        return base_text
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get color for visualization (BGR format)"""
        if self.track_id is not None:
            # Generate a color based on track_id but biased toward green
            return (50, 200 + (self.track_id * 20) % 55, 50)
        else:
            return (50, 255, 50)  # Green-ish color for algae 