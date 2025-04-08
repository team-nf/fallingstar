#!/bin/bash

# Check if we should use the Coral-specific Dockerfile
if [ "$1" == "--coral" ]; then
    COMPOSE_FILE="docker-compose.coral.yml"
    shift
else
    COMPOSE_FILE="docker-compose.yml"
fi

# Allow X server connections for GUI
xhost +local:docker

# Build the Docker image
echo "Building Docker image..."
docker compose -f $COMPOSE_FILE build

# Check if additional arguments were provided
if [ $# -eq 0 ]; then
    # No arguments, run with default settings
    echo "Running training with default settings..."
    docker compose -f $COMPOSE_FILE run coral-trainer train_coral.py \
        --dataset_dir /app/dataset \
        --annotations_file /app/dataset/train/_annotations.coco.json \
        --val_annotations_file /app/dataset/valid/_annotations.coco.json \
        --test_annotations_file /app/dataset/test/_annotations.coco.json
else
    # Arguments provided, run with custom command
    echo "Running with custom arguments: $@"
    docker compose -f $COMPOSE_FILE run coral-trainer train_coral.py "$@"
fi 