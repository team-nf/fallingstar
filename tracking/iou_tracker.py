#!/usr/bin/env python3
"""
IoU-based tracker implementation.

This is a simple but effective tracker that uses Intersection over Union (IoU)
to associate detections with existing tracks.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .tracker_base import Tracker, TrackedObject, BoundingBox, calculate_iou

class IoUTracker(Tracker):
    """
    Simple IoU-based tracker.
    
    This tracker associates detections with tracks based on the IoU between
    bounding boxes. It does not use appearance features or motion models.
    """
    
    def _init_tracker(self, iou_threshold: float = 0.3):
        """
        Initialize tracker parameters.
        
        Args:
            iou_threshold: Minimum IoU for associating a detection with a track
        """
        self.iou_threshold = iou_threshold
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> List[TrackedObject]:
        """
        Update the tracker with new detections.
        
        Args:
            frame: Current video frame (not used by IoU tracker)
            detections: List of detection dictionaries with keys:
                - 'bbox': [xmin, ymin, xmax, ymax]
                - 'class_id': class ID (optional)
                - 'class_name': class name (optional)
                - 'score': confidence score (optional)
                
        Returns:
            List of currently tracked objects
        """
        # If there are no detections, just update the lost counts
        if not detections:
            for obj in self.tracked_objects:
                obj.lost_count += 1
            
            # Remove objects that have been lost for too long
            self.tracked_objects = [obj for obj in self.tracked_objects 
                                  if obj.lost_count <= self.max_lost]
            
            return self.tracked_objects
        
        # If there are no existing tracks, create new ones for all detections
        if not self.tracked_objects:
            for det in detections:
                # Extract detection info
                bbox = BoundingBox(det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3])
                class_id = det.get('class_id')
                class_name = det.get('class_name')
                score = det.get('score')
                
                # Create new tracked object
                obj = TrackedObject(
                    id=self._assign_new_id(),
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    score=score
                )
                
                self.tracked_objects.append(obj)
            
            return self.tracked_objects
        
        # Convert detections to format for matching
        detection_boxes = [
            BoundingBox(det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3])
            for det in detections
        ]
        
        # Create IoU matrix between all tracked objects and new detections
        iou_matrix = np.zeros((len(self.tracked_objects), len(detection_boxes)))
        for i, obj in enumerate(self.tracked_objects):
            for j, det_box in enumerate(detection_boxes):
                iou_matrix[i, j] = calculate_iou(obj.bbox, det_box)
        
        # Initialize tracking state arrays
        matched_indices = []
        unmatched_tracks = list(range(len(self.tracked_objects)))
        unmatched_detections = list(range(len(detections)))
        
        # Match detections to tracks based on IoU
        for track_idx in range(len(self.tracked_objects)):
            # If track lost, require higher IoU to match
            if self.tracked_objects[track_idx].lost_count > 0:
                threshold = self.iou_threshold * 1.5
            else:
                threshold = self.iou_threshold
                
            # Find the detection with highest IoU for this track
            max_iou = threshold
            max_det_idx = -1
            
            for det_idx in unmatched_detections:
                if iou_matrix[track_idx, det_idx] > max_iou:
                    max_iou = iou_matrix[track_idx, det_idx]
                    max_det_idx = det_idx
            
            # If a good match is found, add to matched pairs
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
            
            # Calculate velocity if possible (basic, could be improved)
            if hasattr(obj, 'prev_center'):
                prev_x, prev_y = obj.prev_center
                curr_x, curr_y = det_bbox.center
                vx = curr_x - prev_x
                vy = curr_y - prev_y
                obj.velocity = (vx, vy)
            
            obj.prev_center = det_bbox.center
        
        # Update unmatched tracks (lost)
        for track_idx in unmatched_tracks:
            obj = self.tracked_objects[track_idx]
            obj.lost_count += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            # Extract detection info
            det = detections[det_idx]
            bbox = detection_boxes[det_idx]
            class_id = det.get('class_id')
            class_name = det.get('class_name')
            score = det.get('score')
            
            # Create new tracked object
            obj = TrackedObject(
                id=self._assign_new_id(),
                bbox=bbox,
                class_id=class_id,
                class_name=class_name,
                score=score
            )
            
            obj.prev_center = bbox.center
            self.tracked_objects.append(obj)
        
        # Remove objects that have been lost for too long
        self.tracked_objects = [obj for obj in self.tracked_objects 
                              if obj.lost_count <= self.max_lost]
        
        return self.tracked_objects 