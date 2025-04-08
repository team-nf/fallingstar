# Google Coral TPU Training and Inference for COCO Dataset

This project provides scripts for training and deploying a TensorFlow Lite model with int8 quantization to run on Google Coral Edge TPU. It includes features for training on GPU and pause/resume functionality.

## Setup

1. Place your COCO format dataset in the `dataset` folder. The dataset should contain:
   - Images in JPG/PNG format
   - `annotations.json` file with COCO annotations

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For Google Coral support, install the PyCoral library:
   ```bash
   pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral
   ```

## Training a Model

The training script (`train_coral.py`) does the following:
- Loads a COCO-format dataset
- Creates a multi-label classification model based on MobileNetV2
- Trains the model on your dataset
- Converts the model to TensorFlow Lite with int8 quantization
- Saves both float and quantized models

### Basic Training

Run the training script:
```bash
python train_coral.py --dataset_dir dataset --annotations_file dataset/annotations.json
```

### Pause and Resume Training

The script now supports pausing and resuming training:

1. Start training as normal
2. If you need to stop, press Ctrl+C to interrupt (training state is saved every epoch)
3. To resume, run:
   ```bash
   python train_coral.py --dataset_dir dataset --annotations_file dataset/annotations.json --resume --initial_epoch 10
   ```
   (where 10 is the epoch to resume from)

### GPU Memory Management

For training on a GPU with limited VRAM (like RTX 3070 8GB), use:
```bash
python train_coral.py --dataset_dir dataset --annotations_file dataset/annotations.json --gpu_mem 0.8
```

This restricts TensorFlow to use only 80% of available GPU memory.

### Additional Options

```
--image_size      Input image size for model (default: 320)
--batch_size      Batch size for training (default: 16)
--epochs          Number of epochs to train (default: 50)
--output_dir      Directory to save models (default: models)
--lr              Initial learning rate (default: 0.001)
```

## Testing the Model

The test script (`test_coral.py`) can:
- Run the model on a static image
- Run the model on a live camera feed
- Use Google Coral Edge TPU acceleration if available

Test on a static image:
```bash
python test_coral.py --image_path dataset/test_image.jpg
```

Test with live camera:
```bash
python test_coral.py --use_camera
```

Additional options:
```
--model_path      Path to TFLite model (default: models/model_int8.tflite)
--labels_path     Path to labels file (default: models/labels.txt)
--camera          Camera index for live testing (default: 0)
--image_size      Input image size (default: 320)
--threshold       Detection threshold (default: 0.5)
```

## Using Docker

For convenience, this project includes Docker support with GPU acceleration:

Build and run the training container:
```bash
docker-compose -f docker-compose.coral.yml build
docker-compose -f docker-compose.coral.yml up
```

To resume training using Docker:
```bash
docker-compose -f docker-compose.coral.yml run coral-training python train_coral.py --dataset_dir /app/dataset --annotations_file /app/dataset/annotations.json --resume --initial_epoch 10
```

## Google Coral Deployment

To use the trained model on a Google Coral device:

1. Copy the quantized model `models/model_int8.tflite` to your Coral device
2. Copy the labels file `models/labels.txt` to your Coral device
3. Install PyCoral on your Coral device
4. Run the test script with the Edge TPU model:
   ```bash
   python test_coral.py --model_path models/model_int8.tflite --use_camera
   ```

## Performance Notes

- The int8 quantized model can run on both CPU and Google Coral Edge TPU
- Edge TPU acceleration typically provides 10x or greater speedup
- The model is optimized for real-time inference on edge devices
- GPU acceleration is used during training (making use of your RTX 3070)
- Training creates checkpoints after each epoch for safety 