#!/usr/bin/env python3

"""
PnP (Perspective-n-Point) algorithm for 3D position estimation
Estimates 3D position of objects using their known dimensions
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

class PnPEstimator:
    """
    Class for estimating 3D position of objects using PnP algorithm
    """
    
    def __init__(self, 
                camera_matrix: Optional[np.ndarray] = None,
                dist_coeffs: Optional[np.ndarray] = None,
                calibration_factor: float = 1.0):
        """
        Initialize the PnP algorithm
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix. If None, will use default estimate
            dist_coeffs: Distortion coefficients. If None, will assume no distortion
            calibration_factor: Factor to adjust distance calculations (e.g., 100/120 = 0.833 
                               if actual distance is 100cm but measured as 120cm)
        """
        # Calibration factor to adjust distance calculations
        self.calibration_factor = calibration_factor
        
        # If camera matrix not provided, use a reasonable default estimate
        # Note: For production use, camera should be properly calibrated
        if camera_matrix is None:
            # Default camera matrix for a 640x480 camera
            # Assumes a FOV of ~60 degrees and centered principal point
            focal_length = 800.0  # Estimated focal length in pixels
            self.camera_matrix = np.array([
                [focal_length, 0, 320.0],
                [0, focal_length, 240.0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
            
        # If distortion coefficients not provided, assume no distortion
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        else:
            self.dist_coeffs = dist_coeffs
            
    def is_approximately_square(self, bbox: Tuple[float, float, float, float], threshold: float = 0.8) -> bool:
        """
        Check if the bounding box is approximately square
        For a spherical object, the bounding box should be approximately square
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            threshold: Ratio threshold for considering a bounding box square-like
            
        Returns:
            True if the bounding box is approximately square
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width <= 0 or height <= 0:
            return False
            
        ratio = min(width, height) / max(width, height)
        return ratio > threshold
        
    def estimate_position(self, bbox: Tuple[float, float, float, float], object_size_mm: float) -> Dict[str, Any]:
        """
        Estimate the 3D position of an object from its bounding box
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            object_size_mm: Size of the object in millimeters (diameter for spherical objects)
            
        Returns:
            Dictionary containing position information:
            - success: Whether position estimation was successful
            - distance_mm: Distance to the object in millimeters
            - position_mm: 3D position of the object (x, y, z) in millimeters
            - distance_str: Formatted string with distance information
        """
        # Calculate the width and height of the bounding box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Check if bounding box is valid
        if width <= 0 or height <= 0:
            return {
                'success': False,
                'distance_mm': None,
                'position_mm': None,
                'distance_str': None
            }
        
        # Get the center of the bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # For a spherical object, use the average of width and height as the apparent diameter
        apparent_size_pixels = (width + height) / 2
        
        # Get the focal length from the camera matrix
        focal_length = self.camera_matrix[0, 0]  # Assuming fx = fy
        
        # Calculate distance using the known size formula: distance = (real_size * focal_length) / apparent_size
        # Apply calibration factor to adjust the calculated distance
        uncalibrated_distance_mm = (object_size_mm * focal_length) / apparent_size_pixels
        distance_mm = uncalibrated_distance_mm * self.calibration_factor
        
        # Calculate the 3D coordinates
        # First convert pixel coordinates to normalized image coordinates
        norm_x = (center_x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        norm_y = (center_y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        
        # Calculate 3D coordinates
        x_mm = norm_x * distance_mm
        y_mm = norm_y * distance_mm
        z_mm = distance_mm
        
        # Format the distance string
        if distance_mm < 1000:
            distance_str = f"{distance_mm:.0f}mm"
        else:
            distance_str = f"{distance_mm/1000.0:.2f}m"
            
        return {
            'success': True,
            'distance_mm': distance_mm,
            'position_mm': (x_mm, y_mm, z_mm),
            'distance_str': distance_str,
            'uncalibrated_distance_mm': uncalibrated_distance_mm
        }
    
    def draw_distance_info(self, image: np.ndarray, bbox: Tuple[float, float, float, float], 
                         object_size_mm: float,
                         color: Tuple[int, int, int] = (50, 255, 50)) -> np.ndarray:
        """
        Draw distance information on the image
        
        Args:
            image: Input image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            object_size_mm: Size of the object in millimeters
            color: Color for text (BGR format)
            
        Returns:
            Image with distance information drawn
        """
        # Estimate position
        position_info = self.estimate_position(bbox, object_size_mm)
        if not position_info['success']:
            return image
        
        # Create a copy of the image to avoid modifying the original
        result_image = image.copy()
        
        # Get the distance string
        distance_str = position_info['distance_str']
        
        # Calculate position for the text (above the bounding box)
        text_x = int(bbox[0])
        text_y = int(bbox[1] - 10)
        
        # Ensure text is within image bounds
        if text_y < 0:
            text_y = int(bbox[3] + 20)  # Place text below the bounding box instead
        
        # Draw the distance information
        cv2.putText(
            result_image,
            f"Dist: {distance_str}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        
        return result_image 