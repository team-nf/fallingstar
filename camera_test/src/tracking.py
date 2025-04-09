#!/usr/bin/env python3

"""
Object tracking implementation using SORT (Simple Online and Realtime Tracking)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import collections
import time
import cv2
from filterpy.kalman import KalmanFilter
from typing import List, Dict, Tuple, Optional, Any

from src.objects import DetectedObject

class KalmanBoxTracker:
    """
    This class implements Kalman Filter for tracking bounding boxes in image space.
    """
    count = 0
    
    def __init__(self, bbox: Tuple[float, float, float, float], score: Optional[float] = None):
        """
        Initialize a tracker using initial bounding box
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            score: Detection confidence score
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.  # Measurement uncertainty
        self.kf.P[4:, 4:] *= 1000.  # Covariance
        self.kf.P *= 10.
        self.kf.Q[4:, 4:] *= 0.01  # Process uncertainty

        # Store tracking info
        x1, y1, x2, y2 = bbox
        self.score = score
        self.bbox = bbox
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.centers_history = collections.deque(maxlen=90)  # Store ~3 seconds of position history (at 30 fps)
        self.last_update_time = time.time()
        
        # Calculate initial center position
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.centers_history.append((center_x, center_y, self.last_update_time))

    def update(self, bbox: Tuple[float, float, float, float], score: Optional[float] = None) -> None:
        """
        Updates the state vector with observed bbox
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            score: Detection confidence score
        """
        # Update tracking info
        x1, y1, x2, y2 = bbox
        self.bbox = bbox
        if score is not None:
            self.score = score
            
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Update Kalman filter
        self.kf.update(self._convert_bbox_to_z(bbox))
        
        # Store center position with timestamp
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.last_update_time = time.time()
        self.centers_history.append((center_x, center_y, self.last_update_time))

    def predict(self) -> Tuple[float, float, float, float]:
        """
        Advances the state vector and returns the predicted bounding box
        
        Returns:
            Predicted bounding box (x1, y1, x2, y2)
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        predicted_bbox = self._convert_x_to_bbox(self.kf.x)
        self.history.append(predicted_bbox)
        self.bbox = predicted_bbox
        
        return predicted_bbox

    def get_state(self) -> Tuple[float, float, float, float]:
        """
        Returns the current bounding box estimate
        
        Returns:
            Current bounding box estimate (x1, y1, x2, y2)
        """
        return self.bbox
    
    def get_centers_history(self, max_age: float = 3.0) -> List[Tuple[float, float, float]]:
        """
        Returns the center position history within the specified time window
        
        Args:
            max_age: Maximum age of history points in seconds
            
        Returns:
            List of (x, y, timestamp) tuples
        """
        current_time = time.time()
        # Filter history points to only include those within max_age seconds
        return [(x, y, t) for x, y, t in self.centers_history if current_time - t <= max_age]

    def _convert_bbox_to_z(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Convert bounding box format from [x1, y1, x2, y2] to [x, y, s, r]
        where x, y is the center of the box, s is the scale/area, and r is the aspect ratio
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            State vector [x, y, s, r]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Convert Kalman state [x, y, s, r] to bounding box [x1, y1, x2, y2]
        
        Args:
            x: State vector [x, y, s, r]
            
        Returns:
            Bounding box coordinates (x1, y1, x2, y2)
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        
        return (
            float(x[0] - w/2.),
            float(x[1] - h/2.),
            float(x[0] + w/2.),
            float(x[1] + h/2.)
        )


class ObjectTracker:
    """
    SORT (Simple Online and Realtime Tracking) implementation for tracking objects
    """
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize the object tracker
        
        Args:
            max_age: Maximum number of frames to keep a track alive without matches
            min_hits: Minimum number of hits to start tracking
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.track_colors = {}  # Color map for visualization
    
    def update(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """
        Update the tracker with new detections
        
        Args:
            detections: List of DetectedObject instances
            
        Returns:
            List of DetectedObject instances with tracking IDs
        """
        self.frame_count += 1
        
        # Convert detections to array format
        if not detections:
            dets = np.empty((0, 5))
        else:
            dets = np.array([
                [obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3], obj.score]
                for obj in detections
            ])
        
        # Make predictions for all existing trackers and remove invalid ones
        valid_trackers = []
        
        for trk in self.trackers:
            # Update tracker state
            try:
                bbox = trk.predict()
                if not np.any(np.isnan(bbox)):
                    valid_trackers.append(trk)
            except:
                # Skip any tracker that causes an error
                continue
        
        # Reset tracker list to only include valid trackers
        self.trackers = valid_trackers
        
        # Create arrays of valid tracker boxes for association
        if self.trackers:
            # Create array of tracker boxes
            trks = np.array([trk.bbox for trk in self.trackers])
        else:
            trks = np.empty((0, 4))
        
        # Match detections to trackers
        if len(dets) == 0:
            # No detections - just return original detections
            return detections
        
        if len(trks) == 0:
            # No trackers - create new trackers for all detections
            for i, det in enumerate(detections):
                self.trackers.append(KalmanBoxTracker(det.bbox, det.score))
            return detections
        
        # Calculate IoU matrix for association
        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = self._calculate_iou(det[:4], trk)
        
        # Use Hungarian algorithm for assignment
        matched_indices = []
        if min(iou_matrix.shape) > 0:
            a, b = linear_sum_assignment(-iou_matrix)
            for idx in range(len(a)):
                if iou_matrix[a[idx], b[idx]] >= self.iou_threshold:
                    matched_indices.append([a[idx], b[idx]])
        
        matched_indices = np.array(matched_indices)
        
        # Find unmatched detections and trackers
        unmatched_detections = []
        if len(matched_indices) == 0:
            unmatched_detections = list(range(len(dets)))
        else:
            matched_det_indices = set(matched_indices[:, 0]) if matched_indices.size > 0 else set()
            unmatched_detections = [d for d in range(len(dets)) if d not in matched_det_indices]
        
        # Update matched trackers with new detections
        for match in matched_indices:
            if match[0] < len(detections) and match[1] < len(self.trackers):
                det = detections[match[0]]
                self.trackers[match[1]].update(det.bbox, det.score)
                det.track_id = self.trackers[match[1]].id
        
        # Create new trackers for unmatched detections
        for i in unmatched_detections:
            if i < len(detections):
                new_tracker = KalmanBoxTracker(detections[i].bbox, detections[i].score)
                self.trackers.append(new_tracker)
                # Assign tracker ID to the detection
                detections[i].track_id = new_tracker.id
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        return detections
    
    def draw_trails(self, frame: np.ndarray, max_age: float = 3.0) -> np.ndarray:
        """
        Draw tracking trails on the frame
        
        Args:
            frame: Input frame
            max_age: Maximum age of history points in seconds
            
        Returns:
            Frame with tracking trails
        """
        for trk in self.trackers:
            track_id = trk.id
            
            # Get color for this track
            if track_id not in self.track_colors:
                # Generate a random color for this track
                self.track_colors[track_id] = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
            
            color = self.track_colors[track_id]
            
            # Get track history
            points = trk.get_centers_history(max_age)
            if len(points) < 2:
                continue
            
            # Convert to list of (x, y) points
            points_list = [(int(x), int(y)) for x, y, t in points]
            
            # Draw trail with varying thickness and alpha based on age
            current_time = time.time()
            for i in range(1, len(points_list)):
                pt1 = points_list[i-1]
                pt2 = points_list[i]
                
                # Get timestamps
                t1 = points[i-1][2]
                t2 = points[i][2]
                
                # Calculate age factor (1.0 means newest, 0.0 means oldest)
                age_factor1 = 1.0 - min(1.0, (current_time - t1) / max_age)
                age_factor2 = 1.0 - min(1.0, (current_time - t2) / max_age)
                
                # Line thickness based on average age
                thickness = max(1, int((age_factor1 + age_factor2) * 3))
                
                # Draw the line segment
                cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame
    
    def _calculate_iou(self, bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
        """
        Calculate IoU (Intersection over Union) between two bounding boxes
        
        Args:
            bb_test: First bounding box
            bb_gt: Second bounding box
            
        Returns:
            IoU value
        """
        # Ensure inputs are in the right format
        bb_test = np.array(bb_test, dtype=np.float32).flatten()[:4]
        bb_gt = np.array(bb_gt, dtype=np.float32).flatten()[:4]
        
        # Calculate intersection
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        
        intersection = w * h
        
        # Calculate areas
        area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        
        # Calculate union
        union = area1 + area2 - intersection
        
        # Calculate IoU
        if union <= 0:
            return 0.0
        
        return intersection / union 