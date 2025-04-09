#!/bin/bash

MODEL_PATH="models/ll/GOOGLECORAL_coral_and_algae_monochrome.tflite"
LABELS_PATH="models/ll/labels.txt"

# Detection confidence threshold
THRESHOLD=0.6

# Camera settings (adjust as needed)
CAMERA_ID=0  # Usually 0 for built-in webcam, 1 for external
WIDTH=640
HEIGHT=480


# Build the Docker image if it doesn't exist
if ! docker image inspect fallingstar:latest >/dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t fallingstar:latest .
fi

# Run the application with camera access and X11 forwarding
echo "Starting Coral and Algae detection with tracking..."
echo "Press 'q' to quit the application"

docker compose run --rm coral python3 main.py \
    --model "$MODEL_PATH" \
    --labels "$LABELS_PATH" \
    --threshold "$THRESHOLD" \
    --camera "$CAMERA_ID" \
    --width "$WIDTH" \
    --height "$HEIGHT" 