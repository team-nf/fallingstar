# Object Tracking Package

This package provides multiple algorithms for tracking objects across video frames. It includes a common interface for different tracking strategies, allowing for easy comparison and integration with other systems.

## Features

- Common tracking interface for all algorithms
- Multiple tracker implementations
- Visual comparison tool for algorithm evaluation
- Support for integrating with object detection systems

## Tracking Algorithms

### IoU Tracker

A simple but effective tracker that matches objects between frames based on Intersection over Union (IoU) of bounding boxes.

- **Advantages**: Fast, simple, and effective for many scenarios
- **Disadvantages**: Does not handle occlusions or fast-moving objects well
- **Best for**: Simple tracking tasks with minimal occlusions

### Kalman Filter Tracker

Uses Kalman filtering to predict object positions and smooth trajectories, making it more robust to noisy detections.

- **Advantages**: Handles missing detections and noisy inputs well
- **Disadvantages**: More complex, may drift during long occlusions
- **Best for**: Tracking with partial occlusions or noisy detections

### OpenCV Tracker

Leverages OpenCV's built-in tracking algorithms (CSRT, KCF, MOSSE, etc.) for frame-to-frame tracking.

- **Advantages**: Works without continuous detection, good appearance matching
- **Disadvantages**: May drift over time, needs periodic reinitialization
- **Best for**: When running detection every frame is too expensive

### SORT Tracker

Implementation of the Simple Online and Realtime Tracking algorithm, which combines Kalman filtering with Hungarian algorithm for data association.

- **Advantages**: Robust to missed detections, handles ID switching well
- **Disadvantages**: More computationally expensive
- **Best for**: Complex scenes with multiple objects and occlusions

## Usage

### Basic Usage

```python
from tracking import IoUTracker, KalmanTracker, OpenCVTracker, SORTTracker

# Create a tracker
tracker = KalmanTracker(max_lost=10, iou_threshold=0.3)

# Update with detections on each new frame
tracked_objects = tracker.update(frame, detections)

# Use tracked objects
for obj in tracked_objects:
    print(f"Object {obj.id}: {obj.bbox}")
```

### Detection Format

Detections should be provided as a list of dictionaries with the following format:

```python
detections = [
    {
        "bbox": [xmin, ymin, xmax, ymax],
        "class_id": class_id,  # optional
        "class_name": class_name,  # optional
        "score": confidence_score  # optional
    },
    # ...more detections
]
```

### Visualization

```python
from tracking import visualize_tracked_objects

# Visualize tracked objects on frame
result = visualize_tracked_objects(frame, tracked_objects, show_id=True, show_velocity=True)
```

## Testing and Comparison

Use the included `test_trackers.py` script to compare different tracking algorithms:

```bash
# Test with a video file
python test_trackers.py --video path/to/video.mp4 --trackers iou,kalman,opencv,sort

# Test with a camera
python test_trackers.py --camera 0 --trackers iou,kalman --detection-interval 5

# Test with a detection model
python test_trackers.py --video path/to/video.mp4 --model path/to/model.tflite --labels path/to/labels.txt
```

## Algorithm Selection Guide

| Scenario | Recommended Tracker |
|----------|---------------------|
| Simple scene, few objects | IoUTracker |
| Objects frequently occluded | KalmanTracker |
| Need to run detection infrequently | OpenCVTracker |
| Complex scene, many objects | SORTTracker |

## Requirements

- Python 3.6+
- NumPy
- OpenCV
- SciPy (for SORT implementation) 