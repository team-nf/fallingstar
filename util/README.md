# Vision Processing Utilities

This package provides utilities for vision processing applications, particularly for robotics systems like those used in FRC competitions.

## Contents

### NetworkTables Integration

The `networktables.py` module provides an interface for communicating with NetworkTables servers, typically running on a RoboRIO or other robotics controller.

Features:
- Connect to NetworkTables server using team number or direct address
- Publish tracked object data
- Publish selected target information
- Read configuration values from NetworkTables
- Support for both legacy NetworkTables API and NT4

### Target Selection Algorithms

The `selection.py` module provides algorithms for selecting which tracked object to target based on different criteria.

Available selection algorithms:

| Algorithm | Description |
|-----------|-------------|
| `lowest` | Select the object lowest in the frame (highest y-coordinate) |
| `closest_to_lower_center` | Select the object closest to the bottom-center of the frame |
| `slowest` | Select the object with the slowest movement speed |
| `largest` | Select the object with the largest area |
| `highest_confidence` | Select the object with the highest confidence/score |
| `class_priority` | Select object based on class priority list |
| `newest` | Select the most recently detected object |
| `center_frame` | Select the object closest to the center of the frame |

## Usage Examples

### NetworkTables

```python
from util.networktables import NetworkTablesInterface, get_default_configuration

# Connect to NetworkTables
nt = NetworkTablesInterface(team_number=1234)  # Connect to team's RoboRIO
# OR
nt = NetworkTablesInterface(server_address="10.12.34.2")  # Connect directly

# Wait for connection
if nt.wait_for_connection(timeout=5.0):
    print("Connected to NetworkTables server!")
else:
    print("Failed to connect to NetworkTables server")

# Set default configuration
default_config = get_default_configuration()
nt.set_default_configuration(default_config)

# Get configuration from NetworkTables
config = nt.get_configuration()
print(f"Using target selection method: {config.get('targetSelection', 'lowest')}")

# Publish tracked objects
tracked_objects = [
    {
        "id": 1,
        "class_name": "cube",
        "bbox": [100, 200, 150, 250],
        "score": 0.95
    }
]
nt.publish_tracked_objects(tracked_objects)

# Publish target info
target_info = {
    "id": 1,
    "class_name": "cube",
    "center_x": 125,
    "center_y": 225,
    "norm_x": -0.61,  # Normalized x (-1 to 1)
    "norm_y": 0.17,   # Normalized y (-1 to 1)
    "area": 2500
}
nt.publish_target_info(target_info)
```

### Selection Algorithms

```python
from util.selection import select_target, target_info_to_dict
from tracking import TrackedObject, BoundingBox

# Create some tracked objects
tracked_objects = [
    TrackedObject(id=1, bbox=BoundingBox(100, 100, 150, 150), class_name="cone"),
    TrackedObject(id=2, bbox=BoundingBox(200, 300, 250, 350), class_name="cube"),
    TrackedObject(id=3, bbox=BoundingBox(300, 200, 350, 250), class_name="cone")
]

# Select target using different algorithms
frame_width, frame_height = 640, 480

# Select lowest object
target = select_target(tracked_objects, selection_method="lowest")
print(f"Lowest object: ID {target.id} at position {target.bbox.center}")

# Select object closest to lower center
params = {"frame_width": frame_width, "frame_height": frame_height}
target = select_target(tracked_objects, selection_method="closest_to_lower_center", params=params)
print(f"Object closest to lower center: ID {target.id}")

# Select object by class priority
params = {
    "class_priorities": {"cube": 10, "cone": 5}  # Prefer cubes over cones
}
target = select_target(tracked_objects, selection_method="class_priority", params=params)
print(f"Selected by priority: ID {target.id}, class {target.class_name}")

# Convert target to dictionary for publishing
target_dict = target_info_to_dict(target, frame_width, frame_height)
print(f"Target normalized coordinates: ({target_dict['norm_x']:.2f}, {target_dict['norm_y']:.2f})")
```

## Integration with Vision Processing

This utility package is designed to integrate with the larger vision processing system:

1. The detection system identifies objects in the frame
2. The tracking system tracks objects between frames
3. The selection algorithms choose which object to target
4. The NetworkTables interface communicates this information to the robot control system

Example integrated workflow:

```python
import time
from detection import ObjectDetector
from tracking import KalmanTracker
from util import NetworkTablesInterface, select_target, target_info_to_dict

# Initialize components
detector = ObjectDetector("model.tflite")
tracker = KalmanTracker(max_lost=10)
nt = NetworkTablesInterface(team_number=1234)

# Main processing loop
frame_width, frame_height = 640, 480
while True:
    # Get frame from camera...
    
    # Detect objects
    detections = detector.detect(frame)
    
    # Track objects
    tracked_objects = tracker.update(frame, detections)
    
    # Select target
    config = nt.get_configuration()
    selection_method = config.get("targetSelection", "lowest")
    target = select_target(tracked_objects, selection_method=selection_method)
    
    # Convert target to dictionary
    target_info = target_info_to_dict(target, frame_width, frame_height)
    
    # Publish to NetworkTables
    nt.publish_tracked_objects([obj.__dict__ for obj in tracked_objects])
    nt.publish_target_info(target_info)
    
    time.sleep(0.05)  # 20fps
``` 