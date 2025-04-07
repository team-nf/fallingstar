#!/bin/bash

# Allow X server connections
xhost +local:docker

# Check if API key is provided
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "Error: ROBOFLOW_API_KEY environment variable is not set"
    echo "Please set it with: export ROBOFLOW_API_KEY=\"your-api-key-here\""
    exit 1
fi

# Run the container with Docker Compose
docker-compose up --build 