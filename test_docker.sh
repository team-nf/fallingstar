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

# Build the Docker image if needed
if [ "$1" == "--build" ]; then
    echo "Building Docker image..."
    docker compose -f $COMPOSE_FILE build
    shift
fi

# Check if we want to use camera
if [ "$1" == "--camera" ]; then
    echo "Running test with camera..."
    docker compose -f $COMPOSE_FILE run coral-trainer test_coral.py --use_camera
    exit 0
fi

# Check if an image path was specified
if [ -n "$1" ]; then
    # Image path provided
    echo "Running test on image: $1"
    docker compose -f $COMPOSE_FILE run coral-trainer test_coral.py --image_path "$1"
else
    # Show usage
    echo "Usage: $0 [--coral] [--build] [--camera | image_path]"
    echo "  --coral    Use the Coral-specific Dockerfile"
    echo "  --build    Rebuild the Docker image before running"
    echo "  --camera   Run test with camera input"
    echo "  image_path Path to an image file to test"
fi 