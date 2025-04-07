# Vision Processing System

This vision processing system provides detection, tracking, target selection, and communication capabilities for robotics applications, particularly for FRC (FIRST Robotics Competition).

## Features

- **Object Detection**: TensorFlow Lite based detection with EdgeTPU (Google Coral) support
- **Object Tracking**: Multiple tracking algorithms (SORT, Kalman, IoU, OpenCV)
- **Target Selection**: Various algorithms for selecting which target to track
- **NetworkTables Integration**: Communication with FRC robotics systems
- **Camera Integration**: Support for both direct OpenCV and FRC CameraServer
- **Camera Calibration**: Support for lens distortion correction using camera calibration

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV
- NumPy
- TensorFlow Lite

For FRC integration:
- NetworkTables
- cscore (CameraServer)

For EdgeTPU acceleration:
- PyCoral libraries

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/vision-processing.git
   cd vision-processing
   ```

2. Install dependencies:
   ```
   pip install opencv-python numpy
   pip install tensorflow tflite-runtime
   pip install pynetworktables
   ```

   For FRC CameraServer:
   ```
   pip install robotpy[cscore]
   ```

   For Coral EdgeTPU (optional):
   ```
   pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-win_amd64.whl
   ```

3. Place your model and labels:
   ```
   mkdir -p models
   # Copy your model to models/model.tflite
   # Copy your labels to models/labels.txt
   ```

## Configuration

Edit the `config/config.json` file to configure the system:

- **Camera settings**: Resolution, FPS, CameraServer options
- **Camera calibration**: Enable/disable distortion correction
- **Detection settings**: Model paths, thresholds
- **Tracking settings**: Algorithm, parameters
- **Selection settings**: Target selection algorithm
- **NetworkTables settings**: Team number, server IP

## Running the System

### Basic Usage

```
python main.py
```

### With Command Line Options

```
python main.py --model path/to/model.tflite --labels path/to/labels.txt
```

### Using Different Configuration Files

The system includes pre-configured settings for PC testing and Raspberry Pi deployment:

```
# For PC testing
python main.py --config config/pc_config.json

# For Raspberry Pi deployment
python main.py --config config/rpi_config.json
```

### Camera Selection

```
python main.py --camera 0  # Use camera device 0
python main.py --video path/to/video.mp4  # Use video file
```

### FRC Integration with CameraServer

```
python main.py --use-cs --team 1234
```

### Raspberry Pi Setup for FRC

For running on a Raspberry Pi as part of an FRC robot system:

1. Install dependencies:
   ```
   pip install robotpy[cscore]
   pip install pynetworktables
   ```

2. Configure for CameraServer:
   ```
   python main.py --use-cs --team YOUR_TEAM_NUMBER
   ```

3. For automatic startup, create a service file:
   ```
   sudo nano /etc/systemd/system/vision.service
   ```
   
   With content:
   ```
   [Unit]
   Description=Vision Processing Service
   After=network.target

   [Service]
   ExecStart=/usr/bin/python3 /home/pi/vision-processing/main.py --use-cs --team YOUR_TEAM_NUMBER --no-display
   WorkingDirectory=/home/pi/vision-processing
   StandardOutput=inherit
   StandardError=inherit
   Restart=always
   User=pi

   [Install]
   WantedBy=multi-user.target
   ```

4. Enable and start the service:
   ```
   sudo systemctl enable vision.service
   sudo systemctl start vision.service
   ```

## Camera Calibration

Camera calibration helps correct lens distortion, which can improve detection accuracy. The system supports reading calibration data from JSON or OpenCV format files.

### Calibration Format

The calibration file (`calibration/camera_calibration.json`) should contain:

```json
{
    "camera_matrix": [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ],
    "dist_coeffs": [k1, k2, p1, p2, k3]
}
```

Where:
- `fx`, `fy` are focal lengths
- `cx`, `cy` are principal point coordinates
- `k1`, `k2`, `k3` are radial distortion coefficients
- `p1`, `p2` are tangential distortion coefficients

### Enabling Calibration

To enable camera calibration, modify the `config/config.json` file:

```json
"camera": {
    ...
    "calibration": {
        "use_calibration": true,
        "calibration_file": "calibration/camera_calibration.json",
        "undistort_frames": true
    }
}
```

### Generating Calibration File

You can generate calibration files using OpenCV's calibration tools or the included `calibrate_camera.py` script from the calibration folder.

## Modules

- **main.py**: Main program entry point
- **detection/**: Object detection implementation
- **tracking/**: Various tracking algorithms
- **util/**: Utility functions for NetworkTables and target selection
- **calibration/**: Camera calibration files and utilities

## Selecting a Target

The system includes multiple algorithms for selecting which target to track:

- **lowest**: Select the object lowest in the frame
- **closest_to_lower_center**: Select the object closest to the bottom-center
- **slowest**: Select the object with the slowest movement
- **largest**: Select the largest object
- **highest_confidence**: Select the object with the highest detection confidence
- **class_priority**: Select based on object class priorities
- **center_frame**: Select the object closest to center of frame

## Troubleshooting

### Cannot Find Camera

If the system cannot find your camera, try specifying it directly:
```
python main.py --camera 1  # Try different numbers for different cameras
```

### CameraServer Issues on Raspberry Pi

Make sure you have the appropriate permissions:
```
sudo usermod -a -G video $USER
```

### EdgeTPU Not Working

If you encounter issues with the EdgeTPU:
```
python main.py --model models/model.tflite --detection.use_coral=false
```

### Calibration Issues

If you experience issues with camera calibration:
- Make sure the calibration file exists and contains valid matrices
- Try setting "undistort_frames" to false to compare results
- Ensure the calibration was performed at the same resolution you're using

## License

[Your License Information]

## Running with Docker

The system includes Docker support for both PC and Raspberry Pi environments.

### Prerequisites

- Docker and Docker Compose installed
- For EdgeTPU support: Connected Coral USB Accelerator

### Running with Docker Compose

For PC testing:

```bash
# Simple way
docker-compose -f docker-compose.pc.yml up

# Or with the configurable docker-compose.yml
DOCKERFILE=Dockerfile.pc CONFIG_FILE=config/pc_config.json docker-compose up
```

For Raspberry Pi deployment:

```bash
# Simple way
docker-compose -f docker-compose.rpi.yml up

# Or with the configurable docker-compose.yml
DOCKERFILE=Dockerfile.rpi CONFIG_FILE=config/rpi_config.json NETWORK_MODE=host \
EXTRA_ARGS="--use-cs --team 1234" DISPLAY_VOLUME="" docker-compose up
```

### Building Custom Docker Images

You can customize the Docker images as needed:

```bash
# For PC
docker build -t vision-processing-pc -f Dockerfile.pc .

# For Raspberry Pi
docker build -t vision-processing-rpi -f Dockerfile.rpi .
```

### Docker Environment Variables

The following environment variables can be used to customize the Docker setup:

- `DOCKERFILE`: Path to the Dockerfile to use (default: `Dockerfile.pc`)
- `CONFIG_FILE`: Path to the configuration file (default: `config/pc_config.json`)
- `TEAM_NUMBER`: FRC team number for NetworkTables
- `NETWORK_MODE`: Network mode for Docker (use `host` for Raspberry Pi)
- `EXTRA_ARGS`: Additional command-line arguments
- `DISPLAY_VOLUME`: X11 display socket mount (empty string to disable) 