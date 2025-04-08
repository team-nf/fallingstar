#!/bin/bash

# Build the container if it doesn't exist
docker compose build

# Run the container with access to the Coral device
docker compose run --rm coral bash 