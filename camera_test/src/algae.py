#!/usr/bin/env python3

"""
Algae detection class with 3D position estimation
"""

import time
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
                 calibration_factor: float = 1.0,
                 calibration_x: float = 1.0,
                 calibration_y: float = 1.0,
                 calibration_z: float = 1.0):
        """
        Initialize an Algae detection
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            score: Detection confidence score
            track_id: Tracking ID if available
            diameter_mm: Diameter of algae object in millimeters
            calibration_factor: Legacy overall calibration factor
            calibration_x: Factor to adjust X-axis measurements
            calibration_y: Factor to adjust Y-axis measurements
            calibration_z: Factor to adjust Z-axis measurements (depth/distance)
        """
        super().__init__(bbox, score, track_id)
        
        # Initialize PnP estimator for 3D position estimation
        self.diameter_mm = diameter_mm
        self.calibration_factor = calibration_factor
        self.calibration_x = calibration_x
        self.calibration_y = calibration_y
        self.calibration_z = calibration_z
        
        self.pnp_estimator = PnPEstimator(
            calibration_factor=calibration_factor,
            calibration_x=calibration_x,
            calibration_y=calibration_y,
            calibration_z=calibration_z
        )
        
        # Position data
        self.position_info = None
        
        # Calculate position if bounding box is approximately square
        if self.pnp_estimator.is_approximately_square(bbox):
            self.position_info = self.pnp_estimator.estimate_position(bbox, self.diameter_mm)
    
    def update_position(self, bbox: Tuple[float, float, float, float]) -> None:
        """
        Update position with a new bounding box (for tracked objects)
        
        Args:
            bbox: New bounding box coordinates
        """
        # Only update if the bounding box is approximately square
        if self.pnp_estimator.is_approximately_square(bbox):
            position_info = self.pnp_estimator.estimate_position(bbox, self.diameter_mm)
            
            # If position estimation was successful, update position info
            if position_info and position_info['success']:
                self.position_info = position_info
    
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
            base_text = f"{base_text} | Dist: {distance_str}"
        
        return base_text
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get color for visualization (BGR format)"""
        if self.track_id is not None:
            # Generate a color based on track_id but biased toward green
            return (50, 200 + (self.track_id * 20) % 55, 50)
        else:
            return (50, 255, 50)  # Green-ish color for algae