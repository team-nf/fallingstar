#!/usr/bin/env python3
"""
Target selection algorithms for vision tracking.

This module provides various algorithms for selecting which tracked object
to target, based on different criteria like position, size, movement, etc.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple

# Try to import base classes from tracking
try:
    from tracking.tracker_base import TrackedObject, BoundingBox
    HAS_TRACKING = True
except ImportError:
    HAS_TRACKING = False
    # Define simplified versions of base classes for standalone use
    class BoundingBox:
        def __init__(self, xmin, ymin, xmax, ymax):
            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax
            
        @property
        def width(self):
            return self.xmax - self.xmin
            
        @property
        def height(self):
            return self.ymax - self.ymin
            
        @property
        def area(self):
            return self.width * self.height
            
        @property
        def center(self):
            return (self.xmin + self.width // 2, self.ymin + self.height // 2)
    
    class TrackedObject:
        def __init__(self, id, bbox, class_id=None, class_name=None, 
                    score=None, tracking_score=1.0, lost_count=0, age=1, velocity=None):
            self.id = id
            self.bbox = bbox
            self.class_id = class_id
            self.class_name = class_name
            self.score = score
            self.tracking_score = tracking_score
            self.lost_count = lost_count
            self.age = age
            self.velocity = velocity or (0, 0)

# Type alias for selection function
SelectionFunction = Callable[[List[TrackedObject], Dict[str, Any]], Optional[TrackedObject]]

def select_lowest(tracked_objects: List[TrackedObject], 
                 params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select the object with the lowest position on screen (highest y-coordinate).
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters (not used)
        
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Find object with highest y-coordinate (lowest on screen)
    return max(tracked_objects, key=lambda obj: obj.bbox.ymax)

def select_closest_to_lower_center(tracked_objects: List[TrackedObject], 
                                  params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select the object closest to the bottom center of the frame.
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters:
            - frame_width: Width of the frame
            - frame_height: Height of the frame
            - center_x_offset: X offset from center (default: 0)
            - center_y_offset: Y offset from bottom (default: 100)
            
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Get frame dimensions
    params = params or {}
    frame_width = params.get("frame_width", 640)
    frame_height = params.get("frame_height", 480)
    center_x_offset = params.get("center_x_offset", 0)
    center_y_offset = params.get("center_y_offset", 100)
    
    # Define target point (bottom center with optional offsets)
    target_x = frame_width // 2 + center_x_offset
    target_y = frame_height - center_y_offset
    
    # Find object with center closest to target point
    return min(tracked_objects, key=lambda obj: 
              math.sqrt((obj.bbox.center[0] - target_x)**2 + 
                       (obj.bbox.center[1] - target_y)**2))

def select_slowest(tracked_objects: List[TrackedObject], 
                  params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select the object with the slowest movement speed.
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters:
            - min_age: Minimum age of object to consider (default: 3)
            
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Get parameters
    params = params or {}
    min_age = params.get("min_age", 3)
    
    # Filter objects by minimum age
    valid_objects = [obj for obj in tracked_objects if obj.age >= min_age]
    if not valid_objects:
        # If no objects meet age criteria, use all objects
        valid_objects = tracked_objects
    
    # Find object with lowest velocity magnitude
    return min(valid_objects, key=lambda obj: 
              math.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2) if obj.velocity else 0)

def select_largest(tracked_objects: List[TrackedObject], 
                  params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select the object with the largest area.
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters (not used)
        
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Find object with largest area
    return max(tracked_objects, key=lambda obj: obj.bbox.area)

def select_highest_confidence(tracked_objects: List[TrackedObject], 
                             params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select the object with the highest confidence/score.
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters:
            - use_tracking_score: Whether to use tracking_score instead of detection score (default: False)
            
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Get parameters
    params = params or {}
    use_tracking_score = params.get("use_tracking_score", False)
    
    if use_tracking_score:
        # Find object with highest tracking score
        return max(tracked_objects, key=lambda obj: obj.tracking_score)
    else:
        # Find object with highest detection score, defaulting to tracking score if not available
        return max(tracked_objects, key=lambda obj: obj.score if obj.score is not None else obj.tracking_score)

def select_by_class_priority(tracked_objects: List[TrackedObject], 
                            params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select object based on class priority list.
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters:
            - class_priorities: Dict mapping class names to priority values (higher = more priority)
            - default_priority: Default priority for classes not in the list (default: 0)
            - secondary_criterion: Function to break ties (default: select_largest)
            
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Get parameters
    params = params or {}
    class_priorities = params.get("class_priorities", {})
    default_priority = params.get("default_priority", 0)
    
    # Group objects by class priority
    priority_groups = {}
    for obj in tracked_objects:
        priority = class_priorities.get(obj.class_name, default_priority)
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(obj)
    
    # Get objects with highest priority
    if not priority_groups:
        return None
        
    highest_priority = max(priority_groups.keys())
    candidates = priority_groups[highest_priority]
    
    # If only one candidate, return it
    if len(candidates) == 1:
        return candidates[0]
    
    # Otherwise, break ties using secondary criterion
    secondary_criterion = params.get("secondary_criterion", select_largest)
    return secondary_criterion(candidates, params)

def select_newest(tracked_objects: List[TrackedObject], 
                 params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select the most recently detected object.
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters (not used)
        
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Find object with smallest age and lost_count
    return min(tracked_objects, 
              key=lambda obj: (obj.lost_count, obj.age))

def select_center_frame(tracked_objects: List[TrackedObject], 
                       params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select the object closest to the center of the frame.
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters:
            - frame_width: Width of the frame
            - frame_height: Height of the frame
            - center_x_offset: X offset from center (default: 0)
            - center_y_offset: Y offset from center (default: 0)
            
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Get frame dimensions
    params = params or {}
    frame_width = params.get("frame_width", 640)
    frame_height = params.get("frame_height", 480)
    center_x_offset = params.get("center_x_offset", 0)
    center_y_offset = params.get("center_y_offset", 0)
    
    # Define target point (center with optional offsets)
    target_x = frame_width // 2 + center_x_offset
    target_y = frame_height // 2 + center_y_offset
    
    # Find object with center closest to target point
    return min(tracked_objects, key=lambda obj: 
              math.sqrt((obj.bbox.center[0] - target_x)**2 + 
                       (obj.bbox.center[1] - target_y)**2))

def filter_objects(tracked_objects: List[TrackedObject], 
                  params: Dict[str, Any] = None) -> List[TrackedObject]:
    """
    Filter tracked objects based on various criteria.
    
    Args:
        tracked_objects: List of tracked objects
        params: Additional parameters:
            - min_confidence: Minimum confidence score (default: 0.0)
            - max_lost_count: Maximum lost count (default: 3)
            - min_size: Minimum size in pixels (default: 0)
            - max_size: Maximum size in pixels (default: float('inf'))
            - class_filter: List of allowed class names (default: None = all allowed)
            
    Returns:
        Filtered list of tracked objects
    """
    if not tracked_objects:
        return []
        
    # Get parameters
    params = params or {}
    min_confidence = params.get("min_confidence", 0.0)
    max_lost_count = params.get("max_lost_count", 3)
    min_size = params.get("min_size", 0)
    max_size = params.get("max_size", float('inf'))
    class_filter = params.get("class_filter")
    
    # Apply filters
    filtered_objects = tracked_objects
    
    # Filter by confidence
    if min_confidence > 0:
        filtered_objects = [obj for obj in filtered_objects 
                          if (obj.score is not None and obj.score >= min_confidence) or 
                             obj.tracking_score >= min_confidence]
    
    # Filter by lost count
    if max_lost_count is not None:
        filtered_objects = [obj for obj in filtered_objects 
                          if obj.lost_count <= max_lost_count]
    
    # Filter by size
    if min_size > 0 or max_size < float('inf'):
        filtered_objects = [obj for obj in filtered_objects 
                          if min_size <= obj.bbox.area <= max_size]
    
    # Filter by class
    if class_filter:
        if isinstance(class_filter, str):
            # Convert comma-separated string to list
            class_filter = [c.strip() for c in class_filter.split(',') if c.strip()]
            
        filtered_objects = [obj for obj in filtered_objects 
                          if not class_filter or 
                             (obj.class_name and obj.class_name in class_filter)]
    
    return filtered_objects

# Dictionary of available selection functions
SELECTION_FUNCTIONS = {
    "lowest": select_lowest,
    "closest_to_lower_center": select_closest_to_lower_center,
    "slowest": select_slowest,
    "largest": select_largest,
    "highest_confidence": select_highest_confidence,
    "class_priority": select_by_class_priority,
    "newest": select_newest,
    "center_frame": select_center_frame
}

def select_target(tracked_objects: List[TrackedObject], 
                 selection_method: str = "lowest",
                 params: Dict[str, Any] = None) -> Optional[TrackedObject]:
    """
    Select a target from tracked objects using the specified selection method.
    
    Args:
        tracked_objects: List of tracked objects
        selection_method: Name of the selection method to use
        params: Additional parameters for the selection method
        
    Returns:
        Selected tracked object or None if no objects
    """
    if not tracked_objects:
        return None
        
    # Apply filters before selection
    params = params or {}
    filtered_objects = filter_objects(tracked_objects, params)
    if not filtered_objects:
        return None
    
    # Get selection function
    selection_function = SELECTION_FUNCTIONS.get(selection_method, select_lowest)
    
    # Apply selection function
    return selection_function(filtered_objects, params)

def target_info_to_dict(target: TrackedObject, frame_width: int, frame_height: int) -> Dict[str, Any]:
    """
    Convert a target object to a dictionary with useful information.
    
    Args:
        target: The target object
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Dictionary with target information
    """
    if not target:
        return {"available": False}
    
    # Calculate normalized coordinates (-1 to 1, with 0 at center)
    center_x, center_y = target.bbox.center
    norm_x = (center_x - frame_width / 2) / (frame_width / 2)
    norm_y = (center_y - frame_height / 2) / (frame_height / 2)
    
    # Calculate distance from center (0 to 1)
    distance_from_center = math.sqrt(norm_x**2 + norm_y**2)
    
    # Calculate normalized area (0 to 1)
    norm_area = target.bbox.area / (frame_width * frame_height)
    
    # Calculate angle (in degrees) from center of frame
    angle = math.degrees(math.atan2(norm_y, norm_x))
    
    # Create dictionary with target information
    return {
        "available": True,
        "id": target.id,
        "class_id": target.class_id,
        "class_name": target.class_name,
        "score": target.score if target.score is not None else target.tracking_score,
        "bbox": {
            "xmin": target.bbox.xmin,
            "ymin": target.bbox.ymin,
            "xmax": target.bbox.xmax,
            "ymax": target.bbox.ymax,
            "width": target.bbox.width,
            "height": target.bbox.height,
            "area": target.bbox.area,
        },
        "center_x": center_x,
        "center_y": center_y,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "distance_from_center": distance_from_center,
        "norm_area": norm_area,
        "angle": angle,
        "velocity": target.velocity if target.velocity else (0, 0),
        "lost_count": target.lost_count,
        "age": target.age
    } 