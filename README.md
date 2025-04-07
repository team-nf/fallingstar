# FRC 2025 - Algae and Coral Detector

This Docker container uses the [FRC 2025 - Algae and Coral](https://universe.roboflow.com/frc-team-503-frog-force/frc-2025-algae-and-coral) Roboflow model to detect algae and coral using your camera.

## Prerequisites

- Docker and Docker Compose installed
- A webcam or camera device connected to your computer
- Roboflow API key (get it from [Roboflow](https://app.roboflow.com/settings/api))

## Setup

1. Clone this repository
2. Set your Roboflow API key as an environment variable:

```bash
export ROBOFLOW_API_KEY="your-api-key-here"
```

3. Build and run the container:

```bash
docker-compose up --build
```

## Configuration

You can modify the camera index or confidence threshold in the `docker-compose.yml` file:

```yaml
command: --camera 1 --confidence 0.6
```

Where:
- `--camera` is the index of your camera device (default: 0)
- `--confidence` is the detection threshold (default: 0.5)

## Using a Different Camera

If your camera is not at `/dev/video0`, modify the `docker-compose.yml` file to specify the correct device:

```yaml
devices:
  - /dev/video1:/dev/video0  # If your camera is at /dev/video1
```

## X11 Display Configuration

To display the camera feed, you need to allow Docker to connect to your X server:

```bash
xhost +local:docker
```

## Troubleshooting

- If you can't see the camera feed, make sure you've allowed Docker to connect to your X server.
- If the camera isn't detected, verify the camera device path and permissions.
- For Docker Desktop users, additional configuration may be needed for device access. 