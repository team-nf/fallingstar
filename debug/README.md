# Debug Tools

This directory contains diagnostic and testing utilities for the vision processing system.

## Available Tools

### Camera Test (`camera_test.py`)

Tests camera connection and functionality using OpenCV.

```bash
# List available cameras
python camera_test.py --list

# Test specific camera
python camera_test.py --camera-id 0 --width 640 --height 480 --fps 30
```

### WPILib Camera Test (`wpilib_camera_test.py`)

Tests camera connection and streaming using WPILib/cscore (FRC robotics).

```bash
# List available cameras with supported modes
python wpilib_camera_test.py --list

# Start a camera stream (accessible via HTTP)
python wpilib_camera_test.py --stream --camera-id 0 --width 320 --height 240

# Run full camera functionality test
python wpilib_camera_test.py --full-test
```

Note: Requires the `robotpy-cscore` package to be installed.

### Object Detection Test (`object_detection_test.py`)

Tests object detection using image files instead of live camera feed.

```bash
# Basic detection test with EdgeTPU model
python object_detection_test.py --image path/to/image.jpg --model models/model.tflite --labels models/labels.txt

# Save annotated output image
python object_detection_test.py --image path/to/image.jpg --model models/model.tflite --output results.jpg

# Use CPU instead of EdgeTPU for inference
python object_detection_test.py --image path/to/image.jpg --model models/model.tflite --cpu

# Show results in window (requires GUI)
python object_detection_test.py --image path/to/image.jpg --model models/model.tflite --show
```

### NetworkTables Test (`networktables_test.py`)

Tests connection to a NetworkTables server and data publishing.

```bash
# Test connection to robot or local NT server
python networktables_test.py --server 10.XX.XX.2 --name Vision_Debug_Client --table VisionData

# Enable debug logging
python networktables_test.py --server 127.0.0.1 --debug
```

### Google Coral EdgeTPU Test (`coral_test.py`)

Tests if the Coral TPU is available, loads a model, and runs inference.

```bash
# Basic test with default model
python coral_test.py

# Test with custom model
python coral_test.py --model path/to/model.tflite

# Adjust number of inference runs
python coral_test.py --runs 50
```

## Troubleshooting

### Camera Issues
- Ensure camera is connected properly
- Check if camera is in use by another application
- Try different camera ID or resolution settings
- For WPILib camera issues, ensure robotpy-cscore is installed

### NetworkTables Issues
- Verify IP address is correct (typically 10.TE.AM.2)
- Make sure robot or NT server is running
- Check firewall settings

### Coral TPU Issues
- Ensure the Coral device is connected via USB
- Check LED indicators on the device
- Verify correct drivers are installed

### Object Detection Issues
- Check that model is compatible with EdgeTPU (must be quantized and compiled)
- Ensure image format is supported (JPG, PNG)
- Check labels file format (one label per line)
- For poor detection results, try adjusting the confidence threshold 