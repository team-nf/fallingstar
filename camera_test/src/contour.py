#!/usr/bin/env python3

"""
Contour detection and visualization for algae and coral objects
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

class ContourDetector:
    """Class for detecting and visualizing contours within object bounding boxes"""
    
    def __init__(self, blur_size: int = 5, 
                 canny_low: int = 50, 
                 canny_high: int = 150,
                 min_contour_area: float = 50.0,
                 use_adaptive_threshold: bool = False):
        """
        Initialize the contour detector
        
        Args:
            blur_size: Size of Gaussian blur kernel
            canny_low: Lower threshold for Canny edge detector
            canny_high: Upper threshold for Canny edge detector
            min_contour_area: Minimum contour area to filter noise
            use_adaptive_threshold: Whether to use adaptive thresholding instead of Canny
        """
        self.blur_size = blur_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
        self.use_adaptive_threshold = use_adaptive_threshold
        
        # Create a named window for contour visualization
        self.window_name = "Algae Contour Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)  # Larger window to show multiple algae
        
        # Create trackbars for parameter adjustment
        self._create_trackbars()
        
        # Store contour information for each object
        self.contour_results = {}
        
        # Store the last frame and objects for parameter updates
        self.last_frame = None
        self.last_objects = None
    
    def _create_trackbars(self) -> None:
        """Create trackbars for parameter adjustment"""
        cv2.createTrackbar("Blur", self.window_name, self.blur_size, 25, self._update_blur_size)
        cv2.createTrackbar("Canny Low", self.window_name, self.canny_low, 255, self._update_canny_low)
        cv2.createTrackbar("Canny High", self.window_name, self.canny_high, 255, self._update_canny_high)
        cv2.createTrackbar("Min Area", self.window_name, int(self.min_contour_area), 1000, self._update_min_area)
        cv2.createTrackbar("Adaptive/Canny", self.window_name, int(self.use_adaptive_threshold), 1, self._update_threshold_method)
    
    def _update_blur_size(self, value: int) -> None:
        """Update Gaussian blur kernel size"""
        self.blur_size = value
        self._reprocess_last_frame()
    
    def _update_canny_low(self, value: int) -> None:
        """Update Canny low threshold"""
        self.canny_low = value
        self._reprocess_last_frame()
    
    def _update_canny_high(self, value: int) -> None:
        """Update Canny high threshold"""
        self.canny_high = value
        self._reprocess_last_frame()
    
    def _update_min_area(self, value: int) -> None:
        """Update minimum contour area"""
        self.min_contour_area = float(value)
        self._reprocess_last_frame()
    
    def _update_threshold_method(self, value: int) -> None:
        """Update whether to use adaptive thresholding"""
        self.use_adaptive_threshold = bool(value)
        self._reprocess_last_frame()
    
    def _reprocess_last_frame(self) -> None:
        """Reprocess the last frame with updated parameters"""
        if self.last_frame is not None and self.last_objects is not None:
            self.detect_contours(self.last_frame, self.last_objects)
    
    def detect_contours(self, frame: np.ndarray, objects: List[Any]) -> Dict[int, Dict]:
        """
        Detect contours for all objects in the frame
        
        Args:
            frame: Input frame
            objects: List of detected objects with bounding boxes
            
        Returns:
            Dictionary mapping object IDs to contour information
        """
        # Store frame and objects for parameter updates
        self.last_frame = frame.copy()
        self.last_objects = objects
        
        # Clear previous contour results
        self.contour_results = {}
        
        # Create a fresh visualization image
        contour_vis = np.zeros((frame.shape[0], frame.shape[1]*2, 3), dtype=np.uint8)  # Double width for multiple objects
        
        # Track the current y position for placing object visualizations
        current_y = 10
        max_height_in_row = 0
        current_x = 10
        max_width = contour_vis.shape[1] - 20  # Leave some margin
        
        # Filter for only algae objects
        algae_objects = [obj for obj in objects if obj.track_id is not None and obj.get_class_name() == "Algae"]
        
        # Process each algae object
        for obj in algae_objects:
            # Extract object bounding box and create a region of interest (ROI)
            x1, y1, x2, y2 = map(int, obj.bbox)
            
            # Ensure valid coordinates
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                continue
                
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            # Convert ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred_roi = cv2.GaussianBlur(gray_roi, (self.blur_size, self.blur_size), 0)
            
            # Create binary image for contour detection
            if self.use_adaptive_threshold:
                # Use adaptive thresholding
                binary = cv2.adaptiveThreshold(
                    blurred_roi, 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 
                    11, 
                    2
                )
            else:
                # Use Canny edge detection
                binary = cv2.Canny(blurred_roi, self.canny_low, self.canny_high)
            
            # Find contours
            contours_output = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Handle different OpenCV versions (3.x returns 3 values, 4.x returns 2 values)
            if len(contours_output) == 3:
                _, contours, _ = contours_output
            else:
                contours, _ = contours_output
            
            # Filter contours by area to remove noise
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
            
            # Create a color for this object based on its track_id
            color = obj.get_color()
            
            # Create visualization images
            roi_with_contours = roi.copy()
            cv2.drawContours(roi_with_contours, filtered_contours, -1, color, 2)
            
            # Create a 3-channel binary image for visualization
            binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            # Calculate contour properties
            contour_info = []
            for cnt in filtered_contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                
                # Calculate circularity (1.0 for perfect circle)
                circularity = 0.0
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Calculate minimum enclosing circle
                (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
                
                # Draw the enclosing circle on the contour visualization
                cv2.circle(
                    roi_with_contours,
                    (int(center_x), int(center_y)),
                    int(radius),
                    (0, 0, 255),
                    1
                )
                
                # Add to contour information
                contour_info.append({
                    'contour': cnt,
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'center': (center_x, center_y),
                    'radius': radius
                })
            
            # Store result in contour_results
            self.contour_results[obj.track_id] = {
                'bbox': obj.bbox,
                'class_name': obj.get_class_name(),
                'contours': filtered_contours,
                'contour_info': contour_info,
                'binary': binary,
                'roi': roi,
                'roi_with_contours': roi_with_contours
            }
            
            # Get the position of the object if available
            position_str = "N/A"
            if hasattr(obj, 'get_position_info') and obj.get_position_info() and obj.get_position_info()['success']:
                pos = obj.get_position_info()['position_mm']
                position_str = f"({pos[0]/1000:.2f}, {pos[1]/1000:.2f}, {pos[2]/1000:.2f})m"
            
            # Create a titled section for this object
            # Calculate the space needed
            section_width = roi.shape[1] * 2 + 10  # Original + binary side by side with gap
            section_height = roi.shape[0] + 30  # Image height + space for title
            
            # Check if we need to start a new row
            if current_x + section_width > max_width:
                current_x = 10
                current_y += max_height_in_row + 10
                max_height_in_row = 0
            
            # Update max height for this row
            max_height_in_row = max(max_height_in_row, section_height)
            
            # Create section title
            title = f"ID:{obj.track_id} | Pos:{position_str}"
            cv2.putText(
                contour_vis,
                title,
                (current_x, current_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Place original ROI and contour ROI side by side
            if current_y + roi.shape[0] <= contour_vis.shape[0] and current_x + section_width <= max_width:
                # Place original ROI
                contour_vis[current_y:current_y+roi.shape[0], 
                           current_x:current_x+roi.shape[1]] = roi
                
                # Place contour ROI to the right
                contour_vis[current_y:current_y+roi.shape[0], 
                           current_x+roi.shape[1]+5:current_x+roi.shape[1]*2+5] = roi_with_contours
                
                # Add contour measurements
                if contour_info:
                    # Get the most circular contour
                    most_circular = max(contour_info, key=lambda c: c['circularity'])
                    
                    # Get algae diameter if available
                    diameter_text = ""
                    if hasattr(obj, 'diameter_mm'):
                        diameter_text = f" | Diameter: {obj.diameter_mm:.1f}mm"
                    
                    cv2.putText(
                        contour_vis,
                        f"Area: {most_circular['area']:.0f}px | Circ: {most_circular['circularity']:.2f}{diameter_text}",
                        (current_x, current_y + roi.shape[0] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1
                    )
            
            # Move to next position
            current_x += section_width + 20
        
        # Add parameter information at the top
        param_text = f"Blur: {self.blur_size} | Canny Low: {self.canny_low} | Canny High: {self.canny_high}"
        method_text = "Method: Adaptive Threshold" if self.use_adaptive_threshold else "Method: Canny Edge"
        
        cv2.putText(
            contour_vis,
            param_text,
            (10, contour_vis.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            contour_vis,
            method_text,
            (10, contour_vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # If no algae objects were found, show a message
        if not algae_objects:
            cv2.putText(
                contour_vis,
                "No algae objects detected",
                (contour_vis.shape[1]//2 - 150, contour_vis.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
        
        # Display contour visualization
        cv2.imshow(self.window_name, contour_vis)
        
        return self.contour_results
    
    def get_contour_info(self, track_id: int) -> Optional[Dict]:
        """
        Get contour information for a specific object
        
        Args:
            track_id: Object tracking ID
            
        Returns:
            Dictionary with contour information for the object, or None if not found
        """
        return self.contour_results.get(track_id)
    
    def close(self) -> None:
        """Close the contour visualization window"""
        cv2.destroyWindow(self.window_name) 