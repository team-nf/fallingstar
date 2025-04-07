# FRC Vision Detection Package

This package provides object detection and tracking functionality for FRC robots using Google Coral EdgeTPU for accelerated inference.

## Features

- Real-time object detection using Google Coral EdgeTPU
- Object tracking across frames
- NetworkTables integration for communication with the FRC robot
- Camera utilities for calibration and image processing
- Configurable settings for detection, tracking, and camera parameters

## Requirements

- Raspberry Pi 5 (or compatible hardware)
- Google Coral EdgeTPU (USB accelerator or compatible)
- Global shutter camera (recommended for robotics applications)
- Python 3.7+
- OpenCV 4.x
- TensorFlow Lite libraries
- Google Coral libraries (pycoral, edgetpu)
- PyNetworkTables

## Installation

### Using Docker (Recommended)

The easiest way to get started is using the provided Dockerfile:

```bash
# Build the Docker image
docker build -t frc-vision .

# Run the container with necessary device access
docker run --privileged -v /dev:/dev -v /path/to/config:/app/config -p 1181:1181 -p 1735:1735 frc-vision
```

### Manual Installation

1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-opencv python3-numpy
   ```

2. Install EdgeTPU libraries:
   ```bash
   echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   sudo apt-get update
   sudo apt-get install -y libedgetpu1-std python3-pycoral
   ```

3. Install Python dependencies:
   ```bash
   pip3 install pynetworktables pillow
   ```

## Usage

### Basic Usage

```bash
# Run with default settings
python3 -m detection.main

# Run with a custom configuration file
python3 -m detection.main --config path/to/config.json

# List available cameras
python3 -m detection.main --list-cameras

# Run EdgeTPU diagnostics
python3 -m detection.coral_diagnostic

# Collect training data
python3 -m detection.collect_training_data --output-dir data/training
```

### Configuration

The system is configured via a JSON file. Here's an example configuration:

```json
{
  "camera": {
    "width": 640,
    "height": 480,
    "fps": 30,
    "id": 0,
    "brightness": 50,
    "exposure": 50,
    "white_balance": "auto"
  },
  "model": {
    "path": "model_edgetpu.tflite",
    "labels": "labels.txt",
    "threshold": 0.5
  },
  "tracking": {
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3
  },
  "networking": {
    "team_number": null,
    "server_ip": null,
    "table_name": "Vision"
  },
  "output": {
    "show_window": true,
    "stream_name": "Processed",
    "port": 1181
  },
  "calibration": {
    "enabled": false,
    "file": null
  }
}
```

## Package Structure

- `__init__.py` - Package initialization and exports
- `main.py` - Main entry point for the vision system
- `detector.py` - Object detection using EdgeTPU
- `tracker.py` - Object tracking across frames
- `networktables_manager.py` - NetworkTables communication
- `utils.py` - Utility functions for camera and image processing
- `coral_diagnostic.py` - Diagnostic tool for EdgeTPU
- `collect_training_data.py` - Tool for collecting training data
- `config.json` - Default configuration file

## NetworkTables Integration

The vision system communicates with the robot via NetworkTables. The following tables and keys are used:

- `Vision/status` - System status (0=initializing, 1=running, -1=error)
- `Vision/fps` - Current processing FPS
- `Vision/latency` - Processing latency in milliseconds
- `Vision/targets` - Array of detected targets
  - `class_id` - Class ID of detected object
  - `label` - Label of detected object
  - `confidence` - Detection confidence (0-1)
  - `x` - X coordinate of center (normalized 0-1)
  - `y` - Y coordinate of center (normalized 0-1)
  - `width` - Width of bounding box (normalized 0-1)
  - `height` - Height of bounding box (normalized 0-1)
  - `age` - Age of the track in frames
  - `id` - Unique ID of the track

## Camera Calibration

To improve detection accuracy, you can calibrate your camera:

```bash
# Run camera calibration
python3 -m detection.calibrate_camera --checkerboard 9x6 --square-size 0.025
```

After calibration, update your config.json to enable calibration:

```json
"calibration": {
  "enabled": true,
  "file": "calibration.json"
}
```

## Troubleshooting

### EdgeTPU Not Detected

Run the diagnostic tool to check your EdgeTPU setup:

```bash
python3 -m detection.coral_diagnostic
```

### Camera Issues

- Make sure your camera is supported and properly connected
- Try listing available cameras: `python3 -m detection.main --list-cameras`
- Check camera permissions (run with sudo or add user to video group)

### Networking Issues

- Verify team number in config.json
- Check that robot is accessible on the network
- Test connection with ping or other network diagnostic tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 