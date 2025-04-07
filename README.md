# FallingStar

An open-source computer vision solution for FRC robotics teams, inspired by Limelight.

## Project Structure

FallingStar is divided into two main components:

1. **UI** - A React-based user interface for configuring and monitoring the vision system
2. **Vision** - A Python-based computer vision backend that processes camera feeds and detects targets

These components communicate via a REST API, allowing them to run on separate devices if needed.

## Features

- Multi-camera support 
- Camera calibration tools
- Configurable target detection (AprilTags, color-based, custom contours)
- Target selection and filtering
- 3D pose estimation (PNP solver)
- FRC NetworkTables integration
- Save/load configurations
- Themeable UI

## Requirements

- Docker and Docker Compose
- Web browser (Chrome/Firefox recommended)
- USB cameras (or other compatible camera devices)

## Setup

1. Clone this repository
2. Run `docker-compose up --build`
3. Access the UI at `http://localhost:9029`

## Development

### UI Component

The UI is built with:
- React 
- Material-UI for components
- Socket.IO for real-time updates
- Express backend for API routing

UI code is located in the `ui/` directory.

### Vision Component

The vision processing is built with:
- Python 3.10+
- OpenCV for computer vision algorithms
- Flask for REST API
- NumPy for numerical computation

Vision code is located in the `vision/` directory.

## API Reference

The vision component exposes these API endpoints:

- `GET /api/cameras` - List all available cameras
- `GET /api/camera/:id/stream` - Get a single frame from the camera
- `POST /api/camera/:id/calibrate` - Calibrate a specific camera
- `POST /api/detect` - Detect targets in camera feed
- `POST /api/pnp` - Perform 3D pose estimation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 