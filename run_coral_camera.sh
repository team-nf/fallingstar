#!/bin/bash

# This script runs the Coral TPU camera detection

# Path to the model and labels
MODEL_PATH="models/ll/GOOGLECORAL_coral_and_algae_monochrome.tflite"
LABELS_PATH="models/ll/labels.txt"

# Detection confidence threshold
THRESHOLD=0.3

# Camera settings (adjust as needed)
CAMERA_ID=0  # Usually 0 for built-in webcam, 1 for external
WIDTH=640
HEIGHT=480

echo "Starting Coral TPU camera detection..."
echo "Press 'q' to quit the application"

# Run the detection script
python3 coral_camera_detection.py \
    --model "$MODEL_PATH" \
    --labels "$LABELS_PATH" \
    --threshold "$THRESHOLD" \
    --camera "$CAMERA_ID" \
    --width "$WIDTH" \
    --height "$HEIGHT" 