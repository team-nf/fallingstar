#!/usr/bin/env python3
"""
Base classes and common utilities for object tracking algorithms.
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class BoundingBox:
    """Bounding box representation."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    
    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.xmax - self.xmin
    
    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.ymax - self.ymin
    
    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point (x, y) of the bounding box."""
        return (self.xmin + self.width // 2, self.ymin + self.height // 2)
    
    def to_opencv_format(self) -> Tuple[int, int, int, int]:
        """Convert to OpenCV format (x, y, width, height)."""
        return (self.xmin, self.ymin, self.width, self.height)
    
    def to_numpy_array(self) -> np.ndarray:
        """Convert to numpy array [xmin, ymin, xmax, ymax]."""
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])
    
    @classmethod
    def from_opencv_format(cls, rect: Tuple[int, int, int, int]) -> 'BoundingBox':
        """Create from OpenCV format (x, y, width, height)."""
        x, y, w, h = rect
        return cls(x, y, x + w, y + h)
    
    @classmethod
    def from_numpy_array(cls, arr: np.ndarray) -> 'BoundingBox':
        """Create from numpy array [xmin, ymin, xmax, ymax]."""
        return cls(int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))
    
    @classmethod
    def from_center_size(cls, center_x: int, center_y: int, width: int, height: int) -> 'BoundingBox':
        """Create from center point and dimensions."""
        half_w = width // 2
        half_h = height // 2
        return cls(center_x - half_w, center_y - half_h, 
                 center_x + half_w, center_y + half_h)

@dataclass
class TrackedObject:
    """Representation of a tracked object."""
    id: int
    bbox: BoundingBox
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    score: Optional[float] = None
    tracking_score: float = 1.0
    lost_count: int = 0
    age: int = 1
    velocity: Optional[Tuple[float, float]] = None

def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Calculate intersection area
    x_left = max(box1.xmin, box2.xmin)
    y_top = max(box1.ymin, box2.ymin)
    x_right = min(box1.xmax, box2.xmax)
    y_bottom = min(box1.ymax, box2.ymax)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = box1.area
    box2_area = box2.area
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

def calculate_bbox_distance(box1: BoundingBox, box2: BoundingBox, metric: str = 'center') -> float:
    """Calculate distance between two bounding boxes."""
    if metric == 'center':
        # Euclidean distance between centers
        center1 = box1.center
        center2 = box2.center
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    elif metric == 'iou':
        # IoU-based distance (1 - IoU)
        return 1.0 - calculate_iou(box1, box2)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

class Tracker(ABC):
    """Abstract base class for all trackers."""
    
    def __init__(self, max_lost: int = 3, **kwargs):
        """
        Initialize the tracker.
        
        Args:
            max_lost: Maximum number of frames an object can be lost before it's removed
            **kwargs: Additional tracker-specific parameters
        """
        self.max_lost = max_lost
        self.next_id = 1
        self.tracked_objects: List[TrackedObject] = []
        self._init_tracker(**kwargs)
    
    @abstractmethod
    def _init_tracker(self, **kwargs):
        """Initialize tracker-specific parameters."""
        pass
    
    @abstractmethod
    def update(self, frame: np.ndarray, detections: List[Any] = None) -> List[TrackedObject]:
        """
        Update the tracker with new frame and optionally new detections.
        
        Args:
            frame: Current video frame
            detections: New detections (format varies by implementation)
            
        Returns:
            List of currently tracked objects
        """
        pass
    
    def get_tracked_objects(self) -> List[TrackedObject]:
        """Get current list of tracked objects."""
        return self.tracked_objects
    
    def clear(self):
        """Reset the tracker."""
        self.tracked_objects = []
        self.next_id = 1
    
    def _assign_new_id(self) -> int:
        """Assign a new unique ID for a tracked object."""
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
def visualize_tracked_objects(frame: np.ndarray, tracked_objects: List[TrackedObject], 
                            show_id: bool = True, show_velocity: bool = False) -> np.ndarray:
    """
    Visualize tracked objects on a frame.
    
    Args:
        frame: Input frame
        tracked_objects: List of tracked objects
        show_id: Whether to show object IDs
        show_velocity: Whether to show velocity vectors
        
    Returns:
        Frame with visualizations
    """
    result = frame.copy()
    
    # Define colors for different track IDs (wrap around after 10 colors)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue (OpenCV uses BGR)
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Orange
        (128, 0, 128),  # Purple
        (255, 255, 255),# White
        (0, 128, 128),  # Teal
    ]
    
    for obj in tracked_objects:
        # Get color based on ID
        color = colors[obj.id % len(colors)]
        
        # Draw bounding box
        bbox = obj.bbox
        cv2.rectangle(result, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color, 2)
        
        # Prepare label
        label_parts = []
        
        if show_id:
            label_parts.append(f"ID: {obj.id}")
        
        if obj.class_name:
            label_parts.append(f"{obj.class_name}")
        
        if obj.tracking_score:
            label_parts.append(f"{obj.tracking_score:.2f}")
        
        if label_parts:
            label = ", ".join(label_parts)
            # Draw label background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, 
                         (bbox.xmin, bbox.ymin - label_size[1] - 10), 
                         (bbox.xmin + label_size[0], bbox.ymin),
                         color, -1)
            # Draw label text
            cv2.putText(result, label, (bbox.xmin, bbox.ymin - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw velocity vector if available and requested
        if show_velocity and obj.velocity:
            center = bbox.center
            vx, vy = obj.velocity
            # Scale velocity for visualization
            end_x = int(center[0] + vx * 10)
            end_y = int(center[1] + vy * 10)
            cv2.arrowedLine(result, center, (end_x, end_y), color, 2)
    
    return result 