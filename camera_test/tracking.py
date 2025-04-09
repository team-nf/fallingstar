#!/usr/bin/env python3

"""
Object tracking implementation using SORT (Simple Online and Realtime Tracking)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import collections
import time
from filterpy.kalman import KalmanFilter
import cv2

class KalmanBoxTracker(object):
    """
    This class implements Kalman Filter for tracking bounding boxes in image space.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        bbox is in the format [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
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

        # Extract coordinates (handle both [x1,y1,x2,y2] and [x1,y1,x2,y2,score] formats)
        if len(bbox) > 4:
            x1, y1, x2, y2, score = bbox
        else:
            x1, y1, x2, y2 = bbox
            score = None

        # Store the score if available
        self.score = score
            
        # Initialize state with bounding box
        self.kf.x[:4] = convert_bbox_to_z([x1, y1, x2, y2])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_position = None
        self.centers_history = collections.deque(maxlen=90)  # Store ~3 seconds of position history (at 30 fps)
        self.last_update_time = time.time()
        
        # Calculate initial center position
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.centers_history.append((center_x, center_y, self.last_update_time))

    def update(self, bbox):
        """
        Updates the state vector with observed bbox
        """
        # Extract coordinates (handle both [x1,y1,x2,y2] and [x1,y1,x2,y2,score] formats)
        if len(bbox) > 4:
            x1, y1, x2, y2, score = bbox
            self.score = score
        else:
            x1, y1, x2, y2 = bbox
            
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Update Kalman filter
        self.kf.update(convert_bbox_to_z([x1, y1, x2, y2]))
        
        # Store center position with timestamp
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.last_update_time = time.time()
        self.centers_history.append((center_x, center_y, self.last_update_time))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate
        """
        return convert_x_to_bbox(self.kf.x)
    
    def get_centers_history(self, max_age=3.0):
        """
        Returns the center position history within the specified time window
        """
        current_time = time.time()
        # Filter history points to only include those within max_age seconds
        return [(x, y, t) for x, y, t in self.centers_history if current_time - t <= max_age]

def convert_bbox_to_z(bbox):
    """
    Convert bounding box format from [x1, y1, x2, y2] to [x, y, s, r]
    where x, y is the center of the box, s is the scale/area, and r is the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Convert Kalman state [x, y, s, r] to bounding box [x1, y1, x2, y2]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    
    if score is None:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score]).reshape((1, 5))

def calculate_iou(bb_test, bb_gt):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    
    wh = w * h
    
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) 
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
              
    return o

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Associate detections with trackers using IoU
    Returns 3 lists: matches, unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = calculate_iou(det, trk)
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.column_stack((row_ind, col_ind))
    
    # Find unmatched detections
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    # Find unmatched trackers
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    """
    SORT: Simple Online and Realtime Tracking
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker
        max_age is the number of frames to keep track before considering it lost
        min_hits is the minimum number of hits to start tracking
        iou_threshold is the threshold for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets=np.empty((0, 5))):
        """
        Main method to update the tracker with new detections
        dets - a numpy array of detections in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        Returns list of tracked objects in [x1, y1, x2, y2, id] format
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        # Associate detections with trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)
        
        # Return active trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < self.max_age and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.append(d, trk.id + 1))  # +1 as MOT benchmark expects positive IDs
            i -= 1
            
            # Remove dead trackers
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.array(ret)
        
        return np.empty((0, 5))
    
    def get_trackers(self):
        """
        Returns the current active trackers
        """
        return self.trackers

def draw_tracks(frame, tracker, max_age=3.0, color_map=None):
    """
    Draw tracking trails on the frame
    """
    if color_map is None:
        color_map = {}
    
    for trk in tracker.get_trackers():
        track_id = trk.id
        if track_id not in color_map:
            # Generate a random color for this track
            color_map[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        
        points = trk.get_centers_history(max_age)
        if len(points) < 2:
            continue
        
        # Draw the trail
        points_list = [(int(x), int(y)) for x, y, t in points]
        color = color_map[track_id]
        
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
    
    return frame, color_map 