#!/usr/bin/env python3

"""
Model handling for Coral TPU object detection
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

# Import Coral TPU libraries
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect

from objects import create_object_from_detection, DetectedObject

class ObjectDetector:
    """Class for handling object detection using Google Coral TPU"""
    
    def __init__(self, model_path: str, labels_path: str):
        """
        Initialize the object detector
        
        Args:
            model_path: Path to the TFLite model
            labels_path: Path to the labels file
        """
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load label map
        self.labels = {}
        if labels_path and os.path.exists(labels_path):
            self._load_labels(labels_path)
        
        # Initialize the TPU with the detection model
        print(f"Loading model on Edge TPU: {model_path}")
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        input_details = self.interpreter.get_input_details()
        self.input_shape = input_details[0]['shape']
        _, self.input_height, self.input_width, _ = self.input_shape
        self.input_size = (self.input_width, self.input_height)  # (width, height)
        
        print(f"Model input shape: {self.input_shape}")
    
    def _load_labels(self, labels_file: str) -> None:
        """
        Load labels from file
        
        Args:
            labels_file: Path to the labels file
        """
        with open(labels_file, 'r') as f:
            self.labels = {i: line.strip() for i, line in enumerate(f.readlines())}
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """
        Preprocess the input image to meet model requirements
        
        Args:
            image: Input image in OpenCV format (BGR)
            
        Returns:
            Tuple of (preprocessed PIL image, original RGB image)
        """
        # Convert OpenCV BGR to RGB
        rgb_image = np.array(image[:, :, ::-1])  # BGR to RGB
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Resize the image to the required input dimensions
        resized_image = pil_image.resize(self.input_size, Image.LANCZOS)
        
        return resized_image, rgb_image
    
    def detect(self, image: np.ndarray, threshold: float = 0.3) -> List[DetectedObject]:
        """
        Perform object detection on an image
        
        Args:
            image: Input image in OpenCV format (BGR)
            threshold: Detection confidence threshold
            
        Returns:
            List of DetectedObject instances
        """
        # Preprocess the image
        preprocessed_image, _ = self.preprocess_image(image)
        
        # Run inference
        common.set_input(self.interpreter, np.expand_dims(np.array(preprocessed_image), axis=0))
        self.interpreter.invoke()
        
        # Get detection results
        raw_detections = detect.get_objects(self.interpreter)
        
        # Filter results by threshold
        filtered_detections = [det for det in raw_detections if det.score >= threshold]
        
        # Convert detections to DetectedObject instances
        results = []
        frame_height, frame_width = image.shape[:2]
        
        for det in filtered_detections:
            # Get bounding box coordinates
            bbox = det.bbox
            xmin, ymin, xmax, ymax = bbox
            
            # Scale coordinates to match the original frame size
            x_scale = frame_width / self.input_width
            y_scale = frame_height / self.input_height
            
            scaled_bbox = (
                max(0, int(xmin * x_scale)),
                max(0, int(ymin * y_scale)),
                min(frame_width, int(xmax * x_scale)),
                min(frame_height, int(ymax * y_scale))
            )
            
            # Ensure valid coordinates
            if scaled_bbox[0] >= scaled_bbox[2] or scaled_bbox[1] >= scaled_bbox[3]:
                continue
            
            # Create the appropriate object type
            detected_obj = create_object_from_detection(
                det.id, scaled_bbox, det.score
            )
            
            results.append(detected_obj)
        
        return results
    
    def get_input_size(self) -> Tuple[int, int]:
        """Get the input size required by the model"""
        return self.input_size 