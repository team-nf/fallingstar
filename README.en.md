# FRC 9029 Algae Detection System

Hello! This repo contains our vision processing system developed to detect algae in the FRC 2025 game. We've built a compact system running on Docker that can be used both for development on PC and during competitions on Raspberry Pi.

## Features

- 🎯 TensorFlow Lite based object detection for algae
- 🔄 Object tracking (Kalman, IoU, SORT, and OpenCV algorithms)
- 📏 3D position estimation with PnP algorithm (distance and angle calculation)
- 🌐 Communication with FRC robot via NetworkTables
- 🖥️ Modern, sleek web interface (works on both PC and RPI)
- 🐋 Easy setup and deployment with Docker support
- 🔄 Synchronization with RoboRIO

## Getting Started

You can run the system in two modes: Test (PC) mode and Competition (Raspberry Pi) mode.

### Test Mode (PC)

For running on PC during development or before testing on the robot:

```bash
# Build and run the Docker image
docker-compose -f docker-compose.pc.yml up --build
```

Then navigate to: http://localhost:9029

This mode creates fake algae detections, so you can test the UI even without a real camera.

### Raspberry Pi Mode

When working with the robot or during competition:

```bash
# Build and run the Docker image
docker-compose -f docker-compose.rpi.yml up --build
```

You can access it from the Raspberry Pi's IP address: http://[rpi-ip]:9029

This mode uses EdgeTPU (if available) and works with CameraServer integration.

## System Features (Detailed)

### Algae Detection

Our system detects algae using TensorFlow Lite. We've added Google Coral EdgeTPU support for better performance on Raspberry Pi.

You can modify detection settings in `config/pc_config.json` or `config/rpi_config.json`:

```json
"detection": {
    "model_path": "models/model.tflite",
    "labels_path": "labels.txt",
    "threshold": 0.5,
    "use_coral": false,
    "input_size": [300, 300]
}
```

### Object Tracking

The system tracks detected algae across camera frames. You can choose from several algorithms:

- **Kalman Filter**: Position tracking with velocity estimation (default)
- **IoU Tracker**: Simple and fast bounding box matching
- **SORT**: More complex tracking algorithm
- **OpenCV**: OpenCV's built-in tracking algorithms

You can change the tracking algorithm in the config file:

```json
"tracking": {
    "algorithm": "kalman",
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3
}
```

### 3D Position Estimation (PnP)

The PnP (Perspective-n-Point) algorithm calculates the 3D position of objects with known dimensions. Since we know the physical diameter of the algae (39 cm), our system can calculate:

- Distance from the camera (cm)
- Horizontal and vertical angles
- Full 3D position data

You can enable this feature with the `--enable-pnp` parameter or through the UI.

### NetworkTables Integration

We use NetworkTables to send data to the robot code. The positions, distances, and angles of detected algae are automatically sent.

You can modify NetworkTables settings in the config file:

```json
"networktables": {
    "team_number": 9029,
    "server_ip": "",
    "table_name": "VisionTracking"
}
```

## Web Interface

Our system includes a user-friendly web interface. With this interface, you can:

- Watch the camera feed live
- See algae detections and tracking information
- Change all settings through the graphical interface
- View processing steps (grayscale, edge detection, etc.)
- Switch between dark and light themes

The interface runs on port 9029 (our team number) and is available on both PC and Raspberry Pi.

### Interface Features

- **Live Video Stream**: Watch the camera feed in real-time
- **Detection Information**: List and properties of detected objects
- **Settings Menu**: Edit all system settings
- **Process Visualization**: See each image processing step separately
- **Theme Selection**: Dark (black+green) or light (gray+navy) theme

## Training Your Own Model

If you want to train your own algae detection model:

1. Collect data using `training/collect_training_data.py`
2. Label the data (you can use a tool like Roboflow)
3. Train a model with TensorFlow Object Detection API
4. Convert the model to TFLite format
5. If using EdgeTPU, compile with the EdgeTPU compiler
6. Place the model in the `models/` folder and update the config file

## Troubleshooting

If you encounter any issues:

- Check Docker logs
- Review settings in the UI
- Verify NetworkTables connection
- Check camera access

---

Made with �� by FRC Team 9029 