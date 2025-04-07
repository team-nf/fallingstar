#!/usr/bin/env python3
"""
OpenCV-based tracker implementation.

This tracker uses OpenCV's built-in tracking algorithms to track objects
between frames, making it useful for tracking when detections are sparse
or computationally expensive.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from .tracker_base import Tracker, TrackedObject, BoundingBox, calculate_iou

class OpenCVTracker(Tracker):
    """
    Tracker implementation using OpenCV's built-in tracking algorithms.
    
    This tracker can use any of OpenCV's tracking algorithms (CSRT, KCF, MOSSE, etc.)
    to track objects between frames without needing new detections every frame.
    """
    
    def _init_tracker(self, tracker_type: str = "CSRT", detection_interval: int = 0, 
                    iou_threshold: float = 0.3, reinit_interval: int = 30):
        """
        Initialize tracker parameters.
        
        Args:
            tracker_type: OpenCV tracker type ('CSRT', 'KCF', 'MOSSE', etc.)
            detection_interval: How often to use detections (0 = always use when available)
            iou_threshold: Minimum IoU for associating a detection with a track
            reinit_interval: How often to reinitialize trackers with detections
        """
        self.tracker_type = tracker_type.upper()
        self.detection_interval = detection_interval
        self.iou_threshold = iou_threshold
        self.reinit_interval = reinit_interval
        
        # OpenCV trackers dictionary
        self.trackers = {}
        
        # Frame counter
        self.frame_count = 0
        
        # Last detection frame
        self.last_detection_frame = 0
    
    def _create_tracker(self, tracker_type: str):
        """
        Create a specific type of OpenCV tracker.
        
        Args:
            tracker_type: Type of tracker to create ('CSRT', 'KCF', etc.)
            
        Returns:
            OpenCV tracker object
        """
        tracker_types = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create,
            'MOSSE': cv2.legacy.TrackerMOSSE_create,
            'MIL': cv2.TrackerMIL_create,
            'BOOSTING': cv2.legacy.TrackerBoosting_create,
            'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create
        }
        
        if tracker_type not in tracker_types:
            print(f"Warning: Tracker type '{tracker_type}' not found. Using CSRT instead.")
            tracker_type = 'CSRT'
            
        return tracker_types[tracker_type]()
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> List[TrackedObject]:
        """
        Update the tracker with new frame and optionally new detections.
        
        Args:
            frame: Current video frame
            detections: List of detection dictionaries with keys:
                - 'bbox': [xmin, ymin, xmax, ymax]
                - 'class_id': class ID (optional)
                - 'class_name': class name (optional)
                - 'score': confidence score (optional)
                
        Returns:
            List of currently tracked objects
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Determine if we should use detections in this frame
        use_detections = False
        if detections:
            if self.detection_interval == 0:
                use_detections = True
            elif (self.frame_count - self.last_detection_frame) >= self.detection_interval:
                use_detections = True
        
        # If using detections, set last detection frame
        if use_detections:
            self.last_detection_frame = self.frame_count
        
        # If there are no existing tracks, create them from detections
        if not self.tracked_objects and detections:
            for det in detections:
                # Extract detection info
                bbox = BoundingBox(det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3])
                class_id = det.get('class_id')
                class_name = det.get('class_name')
                score = det.get('score')
                
                # Create new tracked object
                track_id = self._assign_new_id()
                obj = TrackedObject(
                    id=track_id,
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    score=score
                )
                
                # Create tracker
                tracker = self._create_tracker(self.tracker_type)
                tracker.init(frame, bbox.to_opencv_format())
                self.trackers[track_id] = tracker
                
                self.tracked_objects.append(obj)
            
            return self.tracked_objects
        
        # Update existing trackers with new frame
        update_success = {}
        for obj in self.tracked_objects:
            if obj.id in self.trackers:
                success, bbox = self.trackers[obj.id].update(frame)
                update_success[obj.id] = success
                
                if success:
                    # Update bbox with tracker result
                    obj.bbox = BoundingBox.from_opencv_format(bbox)
                    obj.tracking_score = 0.9  # Arbitrary score for successful tracking
                else:
                    # Mark as lost if tracking failed
                    obj.lost_count += 1
                    obj.tracking_score = 0.5  # Lower score for lost tracks
        
        # If using detections, match them with existing tracks
        if use_detections and detections:
            # Convert detections to format for matching
            detection_boxes = [
                BoundingBox(det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3])
                for det in detections
            ]
            
            # Create IoU matrix
            iou_matrix = np.zeros((len(self.tracked_objects), len(detection_boxes)))
            for i, obj in enumerate(self.tracked_objects):
                for j, det_box in enumerate(detection_boxes):
                    iou_matrix[i, j] = calculate_iou(obj.bbox, det_box)
            
            # Match detections to tracks
            matched_indices = []
            unmatched_tracks = list(range(len(self.tracked_objects)))
            unmatched_detections = list(range(len(detections)))
            
            # Assign detections to tracks based on IoU
            for track_idx in range(len(self.tracked_objects)):
                track_id = self.tracked_objects[track_idx].id
                
                # Skip if tracking was successful and track is not old enough for reinit
                if (track_id in update_success and update_success[track_id] and 
                    self.tracked_objects[track_idx].age % self.reinit_interval != 0):
                    unmatched_tracks.remove(track_idx)
                    continue
                
                # Find best matching detection
                max_iou = self.iou_threshold
                max_det_idx = -1
                
                for det_idx in unmatched_detections:
                    if iou_matrix[track_idx, det_idx] > max_iou:
                        max_iou = iou_matrix[track_idx, det_idx]
                        max_det_idx = det_idx
                
                # If match found, add to matched indices
                if max_det_idx >= 0:
                    matched_indices.append((track_idx, max_det_idx))
                    unmatched_detections.remove(max_det_idx)
                    unmatched_tracks.remove(track_idx)
            
            # Update matched tracks with detection info
            for track_idx, det_idx in matched_indices:
                # Get detection
                det = detections[det_idx]
                det_bbox = detection_boxes[det_idx]
                
                # Update track with detection
                obj = self.tracked_objects[track_idx]
                obj.bbox = det_bbox
                obj.lost_count = 0
                obj.age += 1
                obj.class_id = det.get('class_id', obj.class_id)
                obj.class_name = det.get('class_name', obj.class_name)
                obj.score = det.get('score', obj.score)
                obj.tracking_score = 1.0  # High score for matched detection
                
                # Reinitialize tracker with new detection
                tracker = self._create_tracker(self.tracker_type)
                tracker.init(frame, det_bbox.to_opencv_format())
                self.trackers[obj.id] = tracker
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_detections:
                # Extract detection info
                det = detections[det_idx]
                bbox = detection_boxes[det_idx]
                class_id = det.get('class_id')
                class_name = det.get('class_name')
                score = det.get('score')
                
                # Create new tracked object
                track_id = self._assign_new_id()
                obj = TrackedObject(
                    id=track_id,
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    score=score
                )
                
                # Create tracker
                tracker = self._create_tracker(self.tracker_type)
                tracker.init(frame, bbox.to_opencv_format())
                self.trackers[track_id] = tracker
                
                self.tracked_objects.append(obj)
        
        # Update age for all tracks
        for obj in self.tracked_objects:
            obj.age += 1
        
        # Remove lost tracks
        remove_indices = []
        for i, obj in enumerate(self.tracked_objects):
            if obj.lost_count > self.max_lost:
                remove_indices.append(i)
                if obj.id in self.trackers:
                    del self.trackers[obj.id]
        
        # Remove in reverse order to avoid index issues
        for idx in sorted(remove_indices, reverse=True):
            del self.tracked_objects[idx]
        
        return self.tracked_objects
    
    def clear(self):
        """Reset the tracker."""
        super().clear()
        self.trackers = {}
        self.frame_count = 0
        self.last_detection_frame = 0 