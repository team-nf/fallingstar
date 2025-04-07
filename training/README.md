# Object Detection Training for FRC with Google Coral

This directory contains tools for training custom object detection models for FRC game objects and deploying them to Google Coral EdgeTPU.

## Overview

The training pipeline consists of several steps:
1. **Data collection**: Capture images of game objects and annotate them with bounding boxes
2. **Model training**: Train a TensorFlow model to detect objects
3. **Model conversion**: Convert the model to TensorFlow Lite format for edge deployment
4. **EdgeTPU compilation**: Compile the model for Google Coral EdgeTPU accelerator

## Requirements

- Python 3.6+ with TensorFlow 2.x
- OpenCV
- Numpy
- Edge TPU Compiler (for final deployment)

## Data Collection

Before training, you need to collect and annotate images of FRC game objects. We provide a tool to make this process easier.

### 1. Capturing Images with Live Annotation

To capture images from your camera and annotate them in real-time:

```bash
python collect_training_data.py collect \
    --class-name cube \
    --output ./training_data \
    --num-images 100 \
    --camera-id 0 \
    --resolution 640x480
```

For each image:
1. Draw a bounding box around the object with your mouse
2. Press 'c' to capture the image with the annotation
3. Press 's' to skip the current frame
4. Press 'q' to quit

### 2. Importing and Annotating Existing Images

If you already have images:

```bash
python collect_training_data.py import \
    --class-name cone \
    --output ./training_data \
    --images /path/to/your/images
```

For each image:
1. Draw a bounding box around the object
2. Press 'n' to move to the next image
3. Press 's' to skip the current image
4. Press 'q' to quit

### 3. Creating a Balanced Dataset

For best results:
- Collect at least 50-100 images per class
- Capture objects in different positions, orientations, and lighting conditions
- Include images with multiple objects
- Include some images with partial occlusion
- Ensure background variety in your training images

## Training the Model

Once you have collected and annotated your images, you can train the model:

```bash
python train_model.py \
    --annotations ./training_data/annotations \
    --output ./model_output \
    --img-size 300,300 \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 0.001 \
    --compile-edgetpu
```

### Training Options

| Option | Description |
|--------|-------------|
| `--annotations` | Directory containing annotation files (required) |
| `--output` | Directory to save trained model and outputs |
| `--img-size` | Input image size as width,height |
| `--batch-size` | Batch size for training |
| `--epochs` | Number of training epochs |
| `--learning-rate` | Learning rate for optimizer |
| `--compile-edgetpu` | Compile the model for EdgeTPU after training |

### Training Architecture

The training script:
1. Uses MobileNetV2 as the base model, which is compatible with EdgeTPU
2. Fine-tunes the model for object detection with two outputs:
   - Bounding box coordinates (x, y, width, height)
   - Class probabilities
3. Quantizes the model to int8 for EdgeTPU compatibility
4. Automatically evaluates on a validation set

## Model Output Files

After training, you'll find these files in your output directory:
- `best_model.h5`: The best Keras model checkpoint
- `model.tflite`: TensorFlow Lite model (quantized)
- `model_edgetpu.tflite`: Edge TPU compiled model (if --compile-edgetpu was used)
- `labels.txt`: Class labels file
- `class_map.json`: Mapping between class names and indices
- `visualizations/`: Directory containing sample predictions

## Deployment on Google Coral

To use the trained model on Google Coral:

1. Copy these files to your Raspberry Pi with Google Coral:
   - `model_edgetpu.tflite`
   - `labels.txt`

2. Update your vision_processing.py config:
```python
CONFIG = {
    # ...
    "model": {
        "path": "model_edgetpu.tflite",
        "labels": "labels.txt",
        "threshold": 0.5,
    },
    # ...
}
```

## Troubleshooting

### Training Issues
- **Out of memory**: Reduce batch size or image size
- **Poor accuracy**: Collect more training data or adjust learning rate
- **Overfitting**: Add data augmentation or reduce model complexity

### EdgeTPU Compilation Issues
- **Unsupported operations**: Ensure your model only uses operations supported by EdgeTPU
- **Compilation errors**: Check that you have the latest edgetpu_compiler installed
- **Edge TPU Runtime errors**: Make sure model is properly quantized

## Additional Resources

- [TensorFlow Lite Object Detection](https://www.tensorflow.org/lite/examples/object_detection/overview)
- [Edge TPU Compiler Documentation](https://coral.ai/docs/edgetpu/compiler/)
- [FRC Vision Processing Guide](https://docs.wpilib.org/en/stable/docs/software/vision-processing/introduction/index.html) 