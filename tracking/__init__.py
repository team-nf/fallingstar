"""
Object Tracking Package

This package provides various tracking algorithms for following objects
across video frames. It includes:

- IoUTracker: Simple IoU-based tracking
- KalmanTracker: Kalman filter motion prediction
- OpenCVTracker: Using OpenCV's built-in trackers (CSRT, KCF, etc.)
- SORTTracker: Simple Online and Realtime Tracking algorithm
"""

from .tracker_base import (
    BoundingBox,
    TrackedObject,
    Tracker,
    calculate_iou,
    calculate_bbox_distance,
    visualize_tracked_objects
)

from .iou_tracker import IoUTracker
from .kalman_tracker import KalmanTracker
from .opencv_tracker import OpenCVTracker
from .sort_tracker import SORTTracker

__all__ = [
    # Base classes and utilities
    'BoundingBox',
    'TrackedObject',
    'Tracker',
    'calculate_iou',
    'calculate_bbox_distance',
    'visualize_tracked_objects',
    
    # Tracker implementations
    'IoUTracker',
    'KalmanTracker',
    'OpenCVTracker',
    'SORTTracker'
] 