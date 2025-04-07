#!/usr/bin/env python3

import os
import glob
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import shutil
from sklearn.model_selection import train_test_split

def load_dataset(annotations_path, img_size=(300, 300)):
    """
    Load dataset from annotation files.
    
    Args:
        annotations_path: Path to directory containing annotation files
        img_size: Image size for model input (height, width)
    
    Returns:
        images, labels, class_map: Dataset for training
    """
    # Find all annotation files
    annotation_files = glob.glob(os.path.join(annotations_path, '*.json'))
    
    if not annotation_files:
        raise ValueError(f"No annotation files found in {annotations_path}")
    
    images = []
    labels = []
    class_names = set()
    
    # Load annotations from all files
    for ann_file in annotation_files:
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        for annotation in data.get('annotations', []):
            img_path = annotation.get('image_path')
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping")
                continue
                
            # Load and resize image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping")
                continue
                
            img = cv2.resize(img, (img_size[1], img_size[0]))
            img = img / 255.0  # Normalize to [0, 1]
            
            # Get bounding box
            bbox = annotation.get('bbox')
            if not bbox or len(bbox) != 4:
                print(f"Warning: Invalid bbox for {img_path}, skipping")
                continue
                
            # Normalize bbox coordinates
            x, y, w, h = bbox
            h_img, w_img = img.shape[:2]
            
            # Convert to yolo format (x_center, y_center, width, height)
            x_center = (x + w / 2) / w_img
            y_center = (y + h / 2) / h_img
            width = w / w_img
            height = h / h_img
            
            # Get class
            class_name = annotation.get('category_id')
            if not class_name:
                print(f"Warning: No class name for {img_path}, skipping")
                continue
                
            class_names.add(class_name)
            
            # Add to dataset
            images.append(img)
            labels.append([x_center, y_center, width, height, class_name])
    
    # Create class map
    class_list = sorted(list(class_names))
    class_map = {class_name: i for i, class_name in enumerate(class_list)}
    
    # Convert class names to indices
    for i in range(len(labels)):
        class_name = labels[i][4]
        labels[i][4] = class_map[class_name]
    
    return np.array(images), np.array(labels), class_map

def create_model(img_size=(300, 300), num_classes=1, learning_rate=0.001):
    """
    Create a simple object detection model based on MobileNetV2.
    
    Args:
        img_size: Input image size (height, width)
        num_classes: Number of object classes
        learning_rate: Learning rate for optimizer
    
    Returns:
        model: TensorFlow Keras model
    """
    # Base model
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output: [x, y, width, height, class_confidence]
    # For bounding box coordinates (x, y, width, height)
    output_bbox = layers.Dense(4, activation='sigmoid', name='bbox')(x)
    
    # For class confidence (probability for each class)
    output_class = layers.Dense(num_classes, activation='softmax', name='class')(x)
    
    # Combine outputs
    model = models.Model(inputs=base_model.input, outputs=[output_bbox, output_class])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss={
            'bbox': 'mse',
            'class': 'sparse_categorical_crossentropy'
        },
        metrics={
            'bbox': 'mae',
            'class': 'accuracy'
        }
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, output_dir='./model_output'):
    """
    Train the object detection model.
    
    Args:
        model: TensorFlow Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size for training
        epochs: Number of epochs
        output_dir: Directory to save model and logs
    
    Returns:
        history: Training history
        best_model_path: Path to the best model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare callbacks
    checkpoint_path = os.path.join(output_dir, 'best_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        mode='min',
        verbose=1
    )
    
    log_dir = os.path.join(output_dir, 'logs')
    tensorboard = TensorBoard(log_dir=log_dir)
    
    callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard]
    
    # Prepare target data
    y_train_bbox = y_train[:, :4]  # [x, y, width, height]
    y_train_class = y_train[:, 4]  # class index
    
    y_val_bbox = y_val[:, :4]
    y_val_class = y_val[:, 4]
    
    # Train model
    history = model.fit(
        X_train,
        {'bbox': y_train_bbox, 'class': y_train_class},
        validation_data=(X_val, {'bbox': y_val_bbox, 'class': y_val_class}),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, checkpoint_path

def visualize_results(model, X_test, y_test, class_map, num_samples=5, output_dir='./model_output'):
    """
    Visualize model prediction results.
    
    Args:
        model: Trained TensorFlow Keras model
        X_test: Test images
        y_test: Test labels
        class_map: Mapping from class index to class name
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Invert class map
    class_map_inv = {v: k for k, v in class_map.items()}
    
    # Select random samples
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_bbox = y_test[idx, :4]
        true_class_idx = int(y_test[idx, 4])
        true_class = class_map_inv[true_class_idx]
        
        # Make prediction
        pred_bbox, pred_class_prob = model.predict(np.expand_dims(img, axis=0))
        pred_bbox = pred_bbox[0]
        pred_class_idx = np.argmax(pred_class_prob[0])
        pred_class = class_map_inv[pred_class_idx]
        pred_confidence = pred_class_prob[0][pred_class_idx]
        
        # Prepare visualization
        img_vis = (img * 255).astype(np.uint8)
        h, w = img_vis.shape[:2]
        
        # Draw true bounding box (green)
        x_center, y_center, width, height = true_bbox
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, f"True: {true_class}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw predicted bounding box (blue)
        x_center, y_center, width, height = pred_bbox
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_vis, f"Pred: {pred_class} ({pred_confidence:.2f})", (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Save visualization
        vis_path = os.path.join(vis_dir, f"sample_{i+1}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

def convert_to_tflite(model_path, output_dir, quantize=True):
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model_path: Path to Keras model
        output_dir: Directory to save TFLite model
        quantize: Whether to quantize the model for EdgeTPU
    
    Returns:
        tflite_path: Path to converted TFLite model
    """
    # Load Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Configure the converter for int8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 300, 300, 3) * 255
                yield [data.astype(np.float32)]
                
        converter.representative_dataset = representative_dataset
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    tflite_path = os.path.join(output_dir, 'model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    return tflite_path

def save_class_labels(class_map, output_dir):
    """
    Save class labels to a text file for inference.
    
    Args:
        class_map: Mapping from class name to class index
        output_dir: Directory to save labels file
    
    Returns:
        labels_path: Path to labels file
    """
    # Invert class map
    class_map_inv = {v: k for k, v in class_map.items()}
    
    # Create labels file
    labels_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_path, 'w') as f:
        for i in range(len(class_map_inv)):
            f.write(f"{class_map_inv[i]}\n")
    
    return labels_path

def compile_for_edgetpu(tflite_path, output_dir):
    """
    Compile TFLite model for EdgeTPU (requires edgetpu_compiler).
    
    Args:
        tflite_path: Path to TFLite model
        output_dir: Directory to save compiled model
    
    Returns:
        edgetpu_path: Path to compiled EdgeTPU model
    """
    edgetpu_path = os.path.join(output_dir, 'model_edgetpu.tflite')
    
    try:
        import subprocess
        result = subprocess.run(
            ['edgetpu_compiler', tflite_path, '-o', output_dir],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"EdgeTPU compilation error: {result.stderr}")
            return None
            
        print(f"EdgeTPU compilation result: {result.stdout}")
        
        # The compiler outputs the model with a specific name pattern
        # We need to rename it to our desired name
        if os.path.exists(edgetpu_path):
            os.remove(edgetpu_path)
            
        compiled_files = glob.glob(os.path.join(output_dir, '*_edgetpu.tflite'))
        if compiled_files:
            # Rename the file
            os.rename(compiled_files[0], edgetpu_path)
        else:
            print("Warning: Compiled EdgeTPU model not found")
            return None
            
    except FileNotFoundError:
        print("edgetpu_compiler not found. Please install Edge TPU compiler.")
        print("See https://coral.ai/docs/edgetpu/compiler/ for installation instructions.")
        return None
        
    return edgetpu_path

def main():
    parser = argparse.ArgumentParser(description='Train Object Detection Model for EdgeTPU')
    parser.add_argument('--annotations', required=True,
                      help='Path to directory containing annotation files')
    parser.add_argument('--output', default='./model_output',
                      help='Output directory for model and logs')
    parser.add_argument('--img-size', default='300,300',
                      help='Input image size as width,height')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--compile-edgetpu', action='store_true',
                      help='Compile model for EdgeTPU')
    
    args = parser.parse_args()
    
    # Parse image size
    width, height = map(int, args.img_size.split(','))
    img_size = (height, width)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading dataset...")
    images, labels, class_map = load_dataset(args.annotations, img_size=img_size)
    
    # Save class map
    class_map_path = os.path.join(args.output, 'class_map.json')
    with open(class_map_path, 'w') as f:
        json.dump(class_map, f, indent=4)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of classes: {len(class_map)}")
    
    # Create model
    print("Creating model...")
    model = create_model(img_size=img_size, num_classes=len(class_map), learning_rate=args.learning_rate)
    model.summary()
    
    # Train model
    print("Training model...")
    history, best_model_path = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output
    )
    
    # Visualize results
    print("Visualizing results...")
    model = tf.keras.models.load_model(best_model_path)
    visualize_results(model, X_test, y_test, class_map, output_dir=args.output)
    
    # Convert to TFLite
    print("Converting to TFLite...")
    tflite_path = convert_to_tflite(best_model_path, args.output, quantize=True)
    
    # Save class labels
    print("Saving class labels...")
    labels_path = save_class_labels(class_map, args.output)
    
    # Compile for EdgeTPU if requested
    if args.compile_edgetpu:
        print("Compiling for EdgeTPU...")
        edgetpu_path = compile_for_edgetpu(tflite_path, args.output)
        if edgetpu_path:
            print(f"EdgeTPU model saved to {edgetpu_path}")
            
            # Copy labels.txt to match filename
            shutil.copy(labels_path, os.path.join(args.output, 'labels.txt'))
    
    print(f"Training complete! Outputs saved to {args.output}")

if __name__ == "__main__":
    main() 