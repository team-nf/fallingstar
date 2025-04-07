#!/usr/bin/env python3
"""
SORT (Simple Online and Realtime Tracking) implementation.

This is a simplified implementation of the SORT algorithm described in:
https://arxiv.org/abs/1602.00763
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import linear_sum_assignment
from .tracker_base import Tracker, TrackedObject, BoundingBox, calculate_iou

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    The state includes [x, y, s, r, x', y', s'] where:
    - (x, y) is the center of the box
    - s is the scale (area)
    - r is the aspect ratio
    - (x', y', s') are the respective velocities
    """
    count = 0
    
    def __init__(self, bbox: BoundingBox):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: Initial bounding box
        """
        # Define constant velocity model
        self.kf = np.zeros((7, 7))  # State transition matrix
        np.fill_diagonal(self.kf, 1)
        for i in range(4):
            self.kf[i, i+4] = 1
            
        self.kf = np.zeros((7, 4))  # Measurement matrix
        for i in range(4):
            self.kf[i, i] = 1
        
        # Initialize state
        self.x = np.zeros((7, 1))
        self._update_state_from_bbox(bbox)
        
        # Initialize covariance
        self.P = np.eye(7) * 10
        self.Q = np.eye(7) * 0.1
        self.R = np.eye(4) * 1.0
        
        # Basic trackers
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.age = 1
        
        # Store class info
        self.class_id = None
        self.class_name = None
        self.score = None
    
    def _update_state_from_bbox(self, bbox: BoundingBox):
        """Convert bounding box to internal state format."""
        w = bbox.width
        h = bbox.height
        x = bbox.xmin + w/2
        y = bbox.ymin + h/2
        s = w * h  # scale = area
        r = w / float(h)  # aspect ratio
        
        self.x = np.array([[x], [y], [s], [r], [0], [0], [0]])
    
    def update(self, bbox: BoundingBox, class_id=None, class_name=None, score=None):
        """
        Updates the state vector with observed bbox.
        
        Args:
            bbox: New bounding box observation
            class_id: Class ID
            class_name: Class name
            score: Detection confidence score
        """
        self.time_since_update = 0
        self.hits += 1
        
        # Update class info
        if class_id is not None:
            self.class_id = class_id
        if class_name is not None:
            self.class_name = class_name
        if score is not None:
            self.score = score
        
        # Create measurement
        z = np.array([[bbox.center[0]], 
                      [bbox.center[1]], 
                      [bbox.width * bbox.height],  # scale
                      [bbox.width / float(bbox.height)]])  # aspect ratio
        
        # Kalman filter update
        y = z - self.kf[:4, :] @ self.x
        S = self.kf[:4, :] @ self.P @ self.kf[:4, :].T + self.R
        K = self.P @ self.kf[:4, :].T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.kf[:4, :]) @ self.P
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box.
        """
        # Advance state
        self.x = self.kf @ self.x
        
        # Update covariance
        self.P = self.kf @ self.P @ self.kf.T + self.Q
        
        self.age += 1
        self.time_since_update += 1
        
        return self.get_bbox()
    
    def get_bbox(self) -> BoundingBox:
        """Get current bounding box from state."""
        x = self.x[0, 0]
        y = self.x[1, 0]
        s = self.x[2, 0]  # scale (area)
        r = self.x[3, 0]  # aspect ratio (w/h)
        
        # Ensure positive values
        s = max(s, 1)
        r = max(r, 0.1)
        
        # Calculate width and height
        w = np.sqrt(s * r)
        h = np.sqrt(s / r)
        
        # Calculate corner coordinates
        xmin = int(x - w/2)
        ymin = int(y - h/2)
        xmax = int(x + w/2)
        ymax = int(y + h/2)
        
        return BoundingBox(xmin, ymin, xmax, ymax)
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity from state."""
        return (float(self.x[4, 0]), float(self.x[5, 0]))

class SORTTracker(Tracker):
    """
    SORT (Simple Online and Realtime Tracking) implementation.
    
    This tracker uses a Kalman filter with a constant velocity model for
    each tracked object and the Hungarian algorithm for data association.
    """
    
    def _init_tracker(self, iou_threshold: float = 0.3, max_age: int = 7, min_hits: int = 3):
        """
        Initialize tracker parameters.
        
        Args:
            iou_threshold: Minimum IoU for associating a detection with a track
            max_age: Maximum number of frames a track can be lost before it's removed
            min_hits: Minimum number of hits needed before a track is confirmed
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        # Internal SORT trackers
        self.trackers = []
        
        # Map from SORT tracker IDs to our tracker IDs
        self.id_mapping = {}
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> List[TrackedObject]:
        """
        Updates trackers with new detections.
        
        Args:
            frame: Current video frame (not used directly by SORT)
            detections: List of detection dictionaries with keys:
                - 'bbox': [xmin, ymin, xmax, ymax]
                - 'class_id': class ID (optional)
                - 'class_name': class name (optional)
                - 'score': confidence score (optional)
                
        Returns:
            List of currently tracked objects
        """
        # Predict new locations for all trackers
        for t in self.trackers:
            t.predict()
        
        # Clear tracked objects list for rebuilding
        self.tracked_objects = []
        
        # If no detections, just return predictions
        if not detections or len(detections) == 0:
            # Remove trackers that have been lost for too long
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            
            # Convert remaining trackers to tracked objects
            for t in self.trackers:
                # Only include confirmed tracks
                if t.hits >= self.min_hits or t.time_since_update == 0:
                    # Get or create ID
                    if t.id not in self.id_mapping:
                        self.id_mapping[t.id] = self._assign_new_id()
                    
                    # Create tracked object
                    obj = TrackedObject(
                        id=self.id_mapping[t.id],
                        bbox=t.get_bbox(),
                        class_id=t.class_id,
                        class_name=t.class_name,
                        score=t.score,
                        lost_count=t.time_since_update,
                        age=t.age,
                        velocity=t.get_velocity()
                    )
                    
                    self.tracked_objects.append(obj)
            
            return self.tracked_objects
        
        # Convert detections to format for SORT
        detected_boxes = []
        for det in detections:
            bbox = BoundingBox(det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3])
            detected_boxes.append(bbox)
        
        # If there are no existing trackers, create new ones
        if len(self.trackers) == 0:
            for i, det in enumerate(detections):
                bbox = detected_boxes[i]
                trk = KalmanBoxTracker(bbox)
                trk.class_id = det.get('class_id')
                trk.class_name = det.get('class_name')
                trk.score = det.get('score')
                self.trackers.append(trk)
            
            # No associations to make yet
            return self.tracked_objects
        
        # Create IoU matrix between all trackers and detections
        iou_matrix = np.zeros((len(self.trackers), len(detected_boxes)))
        for t, trk in enumerate(self.trackers):
            for d, det_box in enumerate(detected_boxes):
                iou_matrix[t, d] = calculate_iou(trk.get_bbox(), det_box)
        
        # Apply Hungarian algorithm for optimal assignment
        # Note: scipy returns (row_indices, col_indices) for matched pairs
        matched_indices = linear_sum_assignment(-iou_matrix)  # Negate for maximization
        matched_indices = list(zip(matched_indices[0], matched_indices[1]))
        
        # Filter matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                matches.append(m)
            else:
                # This is an unmatched detection/tracker
                pass
        
        # Create lists of matched and unmatched indices
        matched_tracker_indices = [m[0] for m in matches]
        matched_detection_indices = [m[1] for m in matches]
        
        unmatched_trackers = [t for t in range(len(self.trackers)) if t not in matched_tracker_indices]
        unmatched_detections = [d for d in range(len(detected_boxes)) if d not in matched_detection_indices]
        
        # Update matched trackers with assigned detections
        for t, d in matches:
            bbox = detected_boxes[d]
            det = detections[d]
            self.trackers[t].update(bbox, det.get('class_id'), det.get('class_name'), det.get('score'))
        
        # Create new trackers for unmatched detections
        for i in unmatched_detections:
            bbox = detected_boxes[i]
            det = detections[i]
            trk = KalmanBoxTracker(bbox)
            trk.class_id = det.get('class_id')
            trk.class_name = det.get('class_name')
            trk.score = det.get('score')
            self.trackers.append(trk)
        
        # Remove trackers that have been lost for too long
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        # Convert trackers to tracked objects for output
        for t in self.trackers:
            # Only include confirmed tracks
            if t.hits >= self.min_hits or t.time_since_update == 0:
                # Get or create ID
                if t.id not in self.id_mapping:
                    self.id_mapping[t.id] = self._assign_new_id()
                
                # Create tracked object
                obj = TrackedObject(
                    id=self.id_mapping[t.id],
                    bbox=t.get_bbox(),
                    class_id=t.class_id,
                    class_name=t.class_name,
                    score=t.score,
                    lost_count=t.time_since_update,
                    age=t.age,
                    velocity=t.get_velocity()
                )
                
                # Calculate tracking score based on hits and time since update
                if t.time_since_update == 0:
                    obj.tracking_score = 1.0
                else:
                    obj.tracking_score = max(0.0, 1.0 - t.time_since_update / self.max_age)
                
                self.tracked_objects.append(obj)
        
        return self.tracked_objects
    
    def clear(self):
        """Reset the tracker."""
        super().clear()
        self.trackers = []
        self.id_mapping = {} 