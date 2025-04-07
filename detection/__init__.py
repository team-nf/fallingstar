"""
FRC Vision Detection Package

This package provides object detection and tracking functionality for FRC robots
using Google Coral EdgeTPU for accelerated inference.
"""

from detection.detector import ObjectDetector, verify_coral_edgetpu
from detection.tracker import SimpleTracker, BoundingBox
from detection.networktables_manager import NetworkTablesManager 

# Import calibration and training from their new locations
import sys
import os

# These are now in separate directories
# calibration/calibrate_camera.py
# training/collect_training_data.py 