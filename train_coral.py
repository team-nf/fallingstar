import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
import argparse
import json
import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a TFLite int8 model for Google Coral from COCO dataset')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to the dataset directory')
    parser.add_argument('--annotations_file', type=str, default='dataset/train/_annotations.coco.json', help='Path to the COCO annotations file')
    parser.add_argument('--val_annotations_file', type=str, default='dataset/valid/_annotations.coco.json', help='Path to the validation COCO annotations file')
    parser.add_argument('--test_annotations_file', type=str, default='dataset/test/_annotations.coco.json', help='Path to the test COCO annotations file')
    parser.add_argument('--image_size', type=int, default=320, help='Input image size for model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Epoch to resume training from')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--gpu_mem', type=float, default=0.8, help='GPU memory fraction to use')
    return parser.parse_args()


def setup_gpu(gpu_memory_fraction=0.8):
    """Set up GPU configuration for training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Restrict TensorFlow to only use a fraction of the GPU memory
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(gpu_memory_fraction * 8192))]
            )
            print(f"GPU configured to use {gpu_memory_fraction * 100}% of memory")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found. Using CPU for training.")


def load_dataset(annotations_file, output_dir):
    """Load the COCO dataset and save category labels."""
    # Load COCO dataset
    coco = COCO(annotations_file)
    
    # Get all categories
    categories = coco.loadCats(coco.getCatIds())
    print(f"Number of categories: {len(categories)}")
    print("Categories:", [cat['name'] for cat in categories])
    
    # Save category names to a file for later use
    with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")
    
    # Save category mapping for resume training
    category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    with open(os.path.join(output_dir, 'category_mapping.json'), 'w') as f:
        json.dump(category_mapping, f)
    
    return coco, categories


class CocoDataGenerator(tf.keras.utils.Sequence):
    """COCO dataset generator for TensorFlow training."""
    def __init__(self, coco, img_dir, img_ids, batch_size, img_size, num_classes, is_training=True):
        self.coco = coco
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.is_training = is_training
        
        # Load category mapping to ensure consistent ordering
        category_mapping_file = os.path.join(os.path.dirname(os.path.dirname(img_dir)), 'models', 'category_mapping.json')
        if os.path.exists(category_mapping_file):
            with open(category_mapping_file, 'r') as f:
                self.category_mapping = json.load(f)
        else:
            # Create mapping from category IDs to indices
            self.category_mapping = {cat_id: idx for idx, cat_id in enumerate(self.coco.getCatIds())}
        
    def __len__(self):
        return len(self.img_ids) // self.batch_size
    
    def __getitem__(self, idx):
        batch_img_ids = self.img_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        batch_labels = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
        
        for i, img_id in enumerate(batch_img_ids):
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}. Using zeros.")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            
            # Get annotations for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Create one-hot encoded label
            for ann in anns:
                cat_id = ann['category_id']
                # Map category_id to 0-based index using our consistent mapping
                cat_idx = self.category_mapping.get(str(cat_id), self.category_mapping.get(cat_id))
                if cat_idx is not None:
                    batch_labels[i, cat_idx] = 1
            
            batch_images[i] = img
            
        return batch_images, batch_labels


def split_dataset(coco, val_split=0.2):
    """Split the dataset into training and validation sets."""
    # Get all image IDs
    img_ids = list(coco.imgs.keys())
    print(f"Total images: {len(img_ids)}")
    
    # Split into training and validation
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(img_ids)
    split_idx = int(len(img_ids) * (1 - val_split))
    train_img_ids = img_ids[:split_idx]
    val_img_ids = img_ids[split_idx:]
    print(f"Training images: {len(train_img_ids)}")
    print(f"Validation images: {len(val_img_ids)}")
    
    return train_img_ids, val_img_ids


def create_data_generators(coco, dataset_dir, train_img_ids, val_img_ids, batch_size, img_size, num_classes):
    """Create training and validation data generators."""
    train_generator = CocoDataGenerator(
        coco=coco,
        img_dir=dataset_dir,
        img_ids=train_img_ids,
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        is_training=True
    )
    
    val_generator = CocoDataGenerator(
        coco=coco,
        img_dir=dataset_dir,
        img_ids=val_img_ids,
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        is_training=False
    )
    
    return train_generator, val_generator


def representative_dataset_gen(coco, dataset_dir, val_img_ids, img_size, max_samples=100):
    """Generate a representative dataset for quantization."""
    def representative_dataset():
        for i in range(min(max_samples, len(val_img_ids))):
            img_id = val_img_ids[i]
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(dataset_dir, img_info['file_name'])
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            yield [img]
    
    return representative_dataset


def create_model(img_size, num_classes, learning_rate=0.001):
    """Create and compile the model."""
    # Use MobileNetV2 as base model (compatible with Edge TPU)
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Make base model trainable
    base_model.trainable = True
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def setup_callbacks(output_dir):
    """Set up training callbacks including those for pause/resume."""
    # Create a unique experiment folder with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(output_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Checkpoint for best model
    best_checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Checkpoint for latest model (for resuming)
    latest_checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'latest_model.h5'),
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=False,
        verbose=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    )
    
    # Reduce learning rate when plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # CSV logger
    csv_logger = CSVLogger(
        os.path.join(experiment_dir, 'training_log.csv'),
        append=True
    )
    
    callbacks = [
        best_checkpoint,
        latest_checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ]
    
    return callbacks


def train_model(model, train_generator, val_generator, callbacks, epochs, initial_epoch=0):
    """Train the model with the given data generators and callbacks."""
    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=val_generator,
        callbacks=callbacks
    )
    return history


def convert_to_tflite(model, representative_dataset, output_dir):
    """Convert the trained model to TFLite formats."""
    print("Converting to TFLite...")
    
    # Convert to int8 quantized TFLite model (for Edge TPU)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization options
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert and save
    try:
        tflite_model = converter.convert()
        with open(os.path.join(output_dir, 'model_int8.tflite'), 'wb') as f:
            f.write(tflite_model)
        print(f"Saved quantized model to {os.path.join(output_dir, 'model_int8.tflite')}")
    except Exception as e:
        print(f"Error during int8 conversion: {e}")
    
    # Also save a float model for comparison (and as backup)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    try:
        tflite_model = converter.convert()
        with open(os.path.join(output_dir, 'model_float.tflite'), 'wb') as f:
            f.write(tflite_model)
        print(f"Saved float model to {os.path.join(output_dir, 'model_float.tflite')}")
    except Exception as e:
        print(f"Error during float conversion: {e}")


def plot_training_history(history, output_dir):
    """Plot and save the training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    print(f"Saved training history plot to {os.path.join(output_dir, 'training_history.png')}")


def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup GPU
    setup_gpu(args.gpu_mem)
    
    print(f"Loading dataset from {args.dataset_dir}")
    print(f"Annotations file: {args.annotations_file}")
    
    # Load train dataset
    train_coco, categories = load_dataset(args.annotations_file, args.output_dir)
    num_classes = len(categories)
    
    # Get all image paths from training data
    train_img_ids = list(train_coco.imgs.keys())
    print(f"Training images: {len(train_img_ids)}")
    
    # Load validation dataset
    val_coco = COCO(args.val_annotations_file)
    val_img_ids = list(val_coco.imgs.keys())
    print(f"Validation images: {len(val_img_ids)}")
    
    # Create data generators
    train_dir = os.path.dirname(args.annotations_file)
    val_dir = os.path.dirname(args.val_annotations_file)
    
    train_generator = CocoDataGenerator(
        coco=train_coco,
        img_dir=train_dir,
        img_ids=train_img_ids,
        batch_size=args.batch_size,
        img_size=args.image_size,
        num_classes=num_classes,
        is_training=True
    )
    
    val_generator = CocoDataGenerator(
        coco=val_coco,
        img_dir=val_dir,
        img_ids=val_img_ids,
        batch_size=args.batch_size,
        img_size=args.image_size,
        num_classes=num_classes,
        is_training=False
    )
    
    # Create or load model
    if args.resume and os.path.exists(os.path.join(args.output_dir, 'latest_model.h5')):
        print(f"Resuming training from epoch {args.initial_epoch}")
        model = load_model(os.path.join(args.output_dir, 'latest_model.h5'))
    else:
        print("Creating new model")
        model = create_model(args.image_size, num_classes, learning_rate=args.lr)
    
    # Setup callbacks
    callbacks = setup_callbacks(args.output_dir)
    
    # Train the model
    history = train_model(model, train_generator, val_generator, callbacks, args.epochs, args.initial_epoch)
    
    # Plot training history
    if history:
        plot_training_history(history, args.output_dir)
    
    # Load the best model
    best_model_path = os.path.join(args.output_dir, 'best_model.h5')
    if os.path.exists(best_model_path):
        print("Loading best model for conversion")
        model = load_model(best_model_path)
    
    # Generate representative dataset for quantization
    rep_dataset = representative_dataset_gen(val_coco, val_dir, val_img_ids, args.image_size)
    
    # Convert to TFLite
    convert_to_tflite(model, rep_dataset, args.output_dir)
    
    print("Training and conversion completed!")


if __name__ == "__main__":
    main() 