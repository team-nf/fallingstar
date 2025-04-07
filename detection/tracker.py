#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Tuple, Any

class BoundingBox:
    """
    Bounding box representation for object detection.
    
    Attributes:
        x: x-coordinate of top-left corner
        y: y-coordinate of top-left corner
        w: width of bounding box
        h: height of bounding box
        confidence: detection confidence score
        class_id: class ID of the detected object
    """
    
    def __init__(self, x: int, y: int, w: int, h: int, confidence: float = 0, class_id: int = 0):
        """Initialize bounding box."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.class_id = class_id
    
    def get_coords(self) -> Tuple[int, int, int, int]:
        """Get coordinates as a tuple (x, y, w, h)."""
        return (int(self.x), int(self.y), int(self.w), int(self.h))
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return (self.x + self.w / 2, self.y + self.h / 2)
    
    def get_area(self) -> float:
        """Get area of bounding box."""
        return self.w * self.h
    
    def __repr__(self) -> str:
        """String representation of bounding box."""
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h}, conf={self.confidence:.2f}, cls={self.class_id})"


class SimpleTracker:
    """
    Simple object tracker that uses IoU (Intersection over Union) to track objects across frames.
    
    Attributes:
        max_age: Maximum number of frames to keep track of an object after it disappears
        min_hits: Minimum number of hits needed to start tracking an object
        iou_threshold: IoU threshold for matching detections with existing tracks
    """
    
    def __init__(self, max_age: int, min_hits: int, iou_threshold: float):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum number of frames to keep track of an object after it disappears
            min_hits: Minimum number of hits needed to start tracking an object
            iou_threshold: IoU threshold for matching detections with existing tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = {}
        self.track_id_count = 0
    
    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, BoundingBox]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections with 'bbox' key
            
        Returns:
            Dictionary of track IDs to bounding boxes
        """
        # Handle detections in the format from detector
        bboxes = [d['bbox'] for d in detections]
        
        # Update existing trackers
        if not self.trackers:
            # First frame, initialize trackers
            for det in bboxes:
                self.track_id_count += 1
                self.trackers[self.track_id_count] = {
                    "box": det,
                    "age": 0, 
                    "hits": 1,
                    "active": True
                }
            return self.get_active_tracks()
        
        # Match detections to existing trackers using IoU
        matched_indices = []
        unmatched_detections = []
        unmatched_trackers = list(self.trackers.keys())
        
        for i, det in enumerate(bboxes):
            max_iou = self.iou_threshold
            max_iou_id = -1
            
            for track_id in self.trackers:
                if not self.trackers[track_id]["active"]:
                    continue
                
                iou = self._calculate_iou(det, self.trackers[track_id]["box"])
                if iou > max_iou:
                    max_iou = iou
                    max_iou_id = track_id
            
            if max_iou_id != -1:
                matched_indices.append((max_iou_id, i))
                if max_iou_id in unmatched_trackers:
                    unmatched_trackers.remove(max_iou_id)
            else:
                unmatched_detections.append(i)
        
        # Update matched trackers
        for track_id, det_idx in matched_indices:
            self.trackers[track_id]["box"] = bboxes[det_idx]
            self.trackers[track_id]["hits"] += 1
            self.trackers[track_id]["age"] = 0
        
        # Add new trackers for unmatched detections
        for det_idx in unmatched_detections:
            self.track_id_count += 1
            self.trackers[self.track_id_count] = {
                "box": bboxes[det_idx],
                "age": 0,
                "hits": 1,
                "active": True
            }
        
        # Update age of unmatched trackers
        for track_id in unmatched_trackers:
            self.trackers[track_id]["age"] += 1
            # Deactivate trackers that have been lost for too long
            if self.trackers[track_id]["age"] > self.max_age:
                self.trackers[track_id]["active"] = False
        
        return self.get_active_tracks()
    
    def get_active_tracks(self) -> Dict[int, BoundingBox]:
        """
        Get active tracks.
        
        Returns:
            Dictionary of track IDs to bounding boxes
        """
        active_tracks = {}
        for track_id, tracker in self.trackers.items():
            if tracker["active"] and tracker["hits"] >= self.min_hits:
                active_tracks[track_id] = tracker["box"]
        return active_tracks
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        x1, y1, w1, h1 = box1.get_coords()
        x2, y2, w2, h2 = box2.get_coords()
        
        # Calculate the (x, y)-coordinates of the intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate the areas of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def reset(self):
        """Reset tracker to initial state."""
        self.trackers = {}
        self.track_id_count = 0
        
    def get_track_info(self, track_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific track.
        
        Args:
            track_id: ID of the track
            
        Returns:
            Dictionary with track information or None if track does not exist
        """
        if track_id not in self.trackers:
            return None
            
        track = self.trackers[track_id]
        box = track["box"]
        
        return {
            "id": track_id,
            "bbox": box.get_coords(),
            "center": box.get_center(),
            "area": box.get_area(),
            "confidence": box.confidence,
            "class_id": box.class_id,
            "age": track["age"],
            "hits": track["hits"],
            "active": track["active"]
        } 