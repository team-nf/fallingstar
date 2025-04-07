# FallingStar Vision Component

This directory contains the computer vision processing code for FallingStar.

## Requirements

- Python 3.10 or later
- OpenCV
- NumPy
- Flask (for API server)
- Additional dependencies listed in `requirements.txt`

## Installation

All dependencies are installed automatically when using Docker. For manual installation:

```bash
pip install -r requirements.txt
```

## Directory Structure

- `src/` - Python source code
  - `server.py` - Flask API server
  - `get_cameras.py` - Camera detection utility
  - Additional vision processing modules

## API Reference

The vision component exposes these REST API endpoints:

### Camera Management

- `GET /api/cameras` - List all available cameras
- `GET /api/camera/:id/stream` - Get a single frame from camera

### Calibration

- `POST /api/camera/:id/calibrate` - Calibrate a specific camera
  - Body: Calibration parameters (chessboard size, square size, etc.)
  - Returns: Camera matrix and distortion coefficients

### Detection

- `POST /api/detect` - Detect targets in camera feed
  - Body: 
    ```json
    {
      "camera_id": 0,
      "detection_type": "apriltag"
    }
    ```
  - Returns: Array of detected targets with positions and metadata

### Pose Estimation (PnP)

- `POST /api/pnp` - Perform 3D pose estimation
  - Body: 
    ```json
    {
      "camera_id": 0,
      "target_points": [...],
      "object_points": [...]
    }
    ```
  - Returns: Position and rotation of camera relative to target

## Extending the Vision System

To add new detection algorithms:
1. Create a new Python module in the `src/` directory
2. Implement the detection logic
3. Add an API endpoint in `server.py` that uses your module

## Camera Calibration

Camera calibration uses a chessboard pattern to determine:
- Camera matrix (focal length, optical center)
- Distortion coefficients

These parameters are essential for accurate 3D pose estimation.

## FRC Integration

The vision system can publish data to FRC NetworkTables. This feature is currently under development.

## Development

To run the vision component separately:

```bash
cd vision
python -m src.server
```

This will start the Flask server on port 5000. 