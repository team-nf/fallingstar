#!/usr/bin/env python3

"""
Factory module for creating detected objects
"""

from typing import Tuple, Optional
from src.base_object import DetectedObject
from src.algae import Algae
from src.coral import Coral

def create_object_from_detection(detection_class_id: int, 
                                bbox: Tuple[float, float, float, float], 
                                score: float, 
                                track_id: Optional[int] = None,
                                algae_diameter_mm: float = Algae.DIAMETER_AVG,
                                calibration_factor: float = 1.0) -> DetectedObject:
    """
    Factory function to create the appropriate DetectedObject subclass
    
    Args:
        detection_class_id: Class ID from the detector (0=algae, 1=coral)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        score: Detection confidence score
        track_id: Tracking ID if available
        algae_diameter_mm: Diameter of algae object in millimeters
        calibration_factor: Factor to adjust PnP distance calculations
        
    Returns:
        An instance of the appropriate DetectedObject subclass
    """
    if detection_class_id == 0:
        return Algae(bbox, score, track_id, diameter_mm=algae_diameter_mm, calibration_factor=calibration_factor)
    elif detection_class_id == 1:
        return Coral(bbox, score, track_id)
    else:
        # For unrecognized class IDs, return a base object (will raise error on get_class_name)
        # This allows for graceful failure in case of unknown classes
        return DetectedObject(bbox, score, track_id)

# Export the classes for direct import
__all__ = ['create_object_from_detection', 'DetectedObject', 'Algae', 'Coral'] 