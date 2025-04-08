#!/bin/bash

# This script runs the Coral TPU detection on a few test images

# Make the test script executable
chmod +x test_coral_detection.py

# Define the model and labels paths
MODEL_PATH="models/ll/GOOGLECORAL_coral_and_algae_monochrome.tflite"
LABELS_PATH="models/ll/labels.txt"

# Test on a few sample images from the test directory
# These images have annotations indicating they contain both coral and algae
echo "Testing detection on sample images..."

# Create a results directory if it doesn't exist
mkdir -p results

# Define detection threshold
THRESHOLD=0.2

# Run detection on several test images
for img in \
    "dataset/test/ANNOTATED_Manual10_675_jpg.rf.f338a74edaf77eb326cf78ae70fdf961.jpg" \
    "dataset/test/ANNOTATED_Manual17_210_jpg.rf.e42c8304570ac9612c4dbe6e377de977.jpg" \
    "dataset/test/ANNOTATED_Manual30_24_jpg.rf.faf864a83c62e799baca836c6c7b571e.jpg" \
    "dataset/test/ANNOTATED_Manual99_562_jpg.rf.fc988b37734c7c45ecaefc0e07014cd8.jpg"
do
    echo -e "\nProcessing $img..."
    python3 test_coral_detection.py --model "$MODEL_PATH" --labels "$LABELS_PATH" --image "$img" --threshold $THRESHOLD
    
    # Copy the result to the results directory
    OUTPUT_FILE="${img%.*}_detection.jpg"
    if [ -f "$OUTPUT_FILE" ]; then
        BASENAME=$(basename "$OUTPUT_FILE")
        cp "$OUTPUT_FILE" "results/$BASENAME"
        echo "Copied result to results/$BASENAME"
    fi
done

echo -e "\nDetection complete! Output images have been saved to the 'results' directory."
echo "Use 'ls -la results/' to see the detection results." 