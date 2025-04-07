#!/usr/bin/env python3
"""
Kalman filter-based tracker implementation.

This tracker uses Kalman filtering to predict object positions and
smooth their trajectories, making it more robust to detection noise
and missing detections.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .tracker_base import Tracker, TrackedObject, BoundingBox, calculate_iou

class KalmanFilter:
    """
    Simple Kalman filter implementation for tracking objects in 2D space.
    
    This implementation uses a constant velocity model with state vector:
    [x, y, w, h, vx, vy, vw, vh]
    where (x, y) is the center of the box, (w, h) are width and height,
    and (vx, vy, vw, vh) are the respective velocities.
    """
    
    def __init__(self, bbox: BoundingBox):
        """
        Initialize Kalman filter with a bounding box.
        
        Args:
            bbox: Initial bounding box
        """
        # State dimension: [x, y, w, h, vx, vy, vw, vh]
        self.state_dim = 8
        
        # Measurement dimension: [x, y, w, h]
        self.meas_dim = 4
        
        # Initialize state vector
        x, y = bbox.center
        w, h = bbox.width, bbox.height
        self.x = np.zeros((self.state_dim, 1))
        self.x[:4] = np.array([[x], [y], [w], [h]])
        
        # State transition matrix
        self.F = np.eye(self.state_dim)
        for i in range(4):
            self.F[i, i+4] = 1.0  # Add velocity components
        
        # Measurement matrix (maps state to measurement)
        self.H = np.zeros((self.meas_dim, self.state_dim))
        for i in range(self.meas_dim):
            self.H[i, i] = 1.0
        
        # Process covariance
        self.P = np.eye(self.state_dim) * 10
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.1
        self.Q[4:, 4:] *= 10  # Higher uncertainty for velocity
        
        # Measurement noise covariance
        self.R = np.eye(self.meas_dim) * 1.0
        
        # Measurement uncertainty
        self.S = np.zeros((self.meas_dim, self.meas_dim))
        
        # Kalman gain
        self.K = np.zeros((self.state_dim, self.meas_dim))
        
        # Identity matrix
        self.I = np.eye(self.state_dim)
        
        # Prediction time step
        self.dt = 1.0
    
    def predict(self):
        """
        Predict the state forward by one time step.
        """
        # Update transition matrix with current dt
        for i in range(4):
            self.F[i, i+4] = self.dt
        
        # Predict state
        self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.get_bbox()
    
    def update(self, bbox: BoundingBox):
        """
        Update the state with a new measurement.
        
        Args:
            bbox: New bounding box measurement
        """
        # Create measurement vector
        z = np.array([[bbox.center[0]], 
                      [bbox.center[1]], 
                      [bbox.width], 
                      [bbox.height]])
        
        # Calculate measurement residual
        y = z - self.H @ self.x
        
        # Calculate innovation covariance
        self.S = self.H @ self.P @ self.H.T + self.R
        
        # Calculate Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        
        # Update state
        self.x = self.x + self.K @ y
        
        # Update covariance
        self.P = (self.I - self.K @ self.H) @ self.P
        
        return self.get_bbox()
    
    def get_bbox(self) -> BoundingBox:
        """
        Get the current bounding box from the state.
        """
        x, y, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        
        # Ensure positive width and height
        w = max(10, w)
        h = max(10, h)
        
        return BoundingBox.from_center_size(int(x), int(y), int(w), int(h))
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get the current velocity from the state.
        """
        return (float(self.x[4, 0]), float(self.x[5, 0]))

class KalmanTracker(Tracker):
    """
    Tracker implementation using Kalman filtering.
    
    This tracker uses a Kalman filter for each tracked object to predict
    its position in the next frame, making it more robust to detection
    noise and occlusions.
    """
    
    def _init_tracker(self, iou_threshold: float = 0.3, max_prediction_steps: int = 5):
        """
        Initialize tracker parameters.
        
        Args:
            iou_threshold: Minimum IoU for associating a detection with a track
            max_prediction_steps: Maximum number of frames to predict without detection
        """
        self.iou_threshold = iou_threshold
        self.max_prediction_steps = max_prediction_steps
        
        # Dictionary to store Kalman filters for each track
        self.filters = {}
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> List[TrackedObject]:
        """
        Update the tracker with new detections.
        
        Args:
            frame: Current video frame (not used directly)
            detections: List of detection dictionaries with keys:
                - 'bbox': [xmin, ymin, xmax, ymax]
                - 'class_id': class ID (optional)
                - 'class_name': class name (optional)
                - 'score': confidence score (optional)
                
        Returns:
            List of currently tracked objects
        """
        # Update time step for all Kalman filters
        dt = 1.0  # Assume constant time step for now
        for track_id, kf in self.filters.items():
            kf.dt = dt
        
        # Predict new locations for all tracks
        for obj in self.tracked_objects:
            if obj.id in self.filters:
                # Predict new position using Kalman filter
                predicted_bbox = self.filters[obj.id].predict()
                obj.bbox = predicted_bbox
                obj.velocity = self.filters[obj.id].get_velocity()
        
        # If there are no detections, just update the lost counts
        if not detections:
            for obj in self.tracked_objects:
                obj.lost_count += 1
            
            # Remove objects that have been lost for too long
            remove_ids = []
            for i, obj in enumerate(self.tracked_objects):
                if obj.lost_count > self.max_lost:
                    remove_ids.append(i)
                    if obj.id in self.filters:
                        del self.filters[obj.id]
            
            for i in sorted(remove_ids, reverse=True):
                del self.tracked_objects[i]
            
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
                track_id = self._assign_new_id()
                obj = TrackedObject(
                    id=track_id,
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    score=score
                )
                
                # Create Kalman filter for this track
                self.filters[track_id] = KalmanFilter(bbox)
                
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
            obj.lost_count = 0
            obj.age += 1
            obj.class_id = det.get('class_id', obj.class_id)
            obj.class_name = det.get('class_name', obj.class_name)
            obj.score = det.get('score', obj.score)
            
            # Update Kalman filter with new detection
            kf = self.filters[obj.id]
            kf.update(det_bbox)
            
            # Get updated bbox and velocity from Kalman filter
            obj.bbox = kf.get_bbox()
            obj.velocity = kf.get_velocity()
        
        # Update unmatched tracks
        remove_track_indices = []
        for i, track_idx in enumerate(unmatched_tracks):
            obj = self.tracked_objects[track_idx]
            obj.lost_count += 1
            
            # Remove if lost for too long
            if obj.lost_count > self.max_lost:
                remove_track_indices.append(i)
                if obj.id in self.filters:
                    del self.filters[obj.id]
        
        # Remove tracks in reverse order to avoid index issues
        for idx in sorted(remove_track_indices, reverse=True):
            track_idx = unmatched_tracks[idx]
            del self.tracked_objects[track_idx]
            del unmatched_tracks[idx]
        
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
            
            # Create Kalman filter for this track
            self.filters[track_id] = KalmanFilter(bbox)
            obj.velocity = (0, 0)
            
            self.tracked_objects.append(obj)
        
        return self.tracked_objects 