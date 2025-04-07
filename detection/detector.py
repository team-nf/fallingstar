#!/usr/bin/env python3

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Any
import os

class ObjectDetector:
    """
    Object detector using Google Coral EdgeTPU for accelerated inference.
    """
    
    def __init__(self, model_path: str, labels_path: str, threshold: float = 0.5):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to the Edge TPU model (.tflite)
            labels_path: Path to the labels file
            threshold: Detection confidence threshold
        """
        self.threshold = threshold
        self.model_path = model_path
        self.labels_path = labels_path
        
        # Import EdgeTPU libraries here to handle errors gracefully
        try:
            from edgetpu.detection.engine import DetectionEngine
            from pycoral.utils import edgetpu
            self._has_coral = True
            self._using_coral = True
        except ImportError:
            self._has_coral = False
            self._using_coral = False
            print("WARNING: Google Coral libraries not found. Running in CPU-only mode.")
        
        # Load labels
        self._load_labels()
            
        # Initialize Edge TPU if available
        self._init_engine()
    
    def _load_labels(self):
        """Load labels from file."""
        try:
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.labels)} class labels.")
        except Exception as e:
            print(f"Error loading labels: {e}")
            self.labels = ["unknown"]
    
    def _init_engine(self):
        """Initialize detection engine with EdgeTPU if available."""
        # Verify EdgeTPU is available first
        has_tpu = verify_coral_edgetpu()
        
        try:
            if has_tpu:
                # Initialize with EdgeTPU
                from edgetpu.detection.engine import DetectionEngine
                self.engine = DetectionEngine(self.model_path)
                self._using_coral = True
                print("Successfully initialized EdgeTPU detection engine.")
            else:
                # Fallback to TFLite
                import tflite_runtime.interpreter as tflite
                self._using_coral = False
                
                # Load TFLite model and allocate tensors
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                
                # Get model details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.input_shape = self.input_details[0]['shape']
                
                print("Initialized TFLite interpreter (no EdgeTPU acceleration).")
        except Exception as e:
            print(f"Error initializing detection engine: {e}")
            self._using_coral = False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of detection results with format:
            [
                {
                    'bbox': [x, y, width, height],
                    'score': confidence score,
                    'class_id': class id,
                    'class_name': class name
                },
                ...
            ]
        """
        # Convert to RGB (EdgeTPU expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        start_time = time.time()
        
        if self._using_coral:
            # Use EdgeTPU for inference
            detections = self._detect_with_coral(rgb_frame)
        else:
            # Use TFLite for inference
            detections = self._detect_with_tflite(rgb_frame)
        
        inference_time = (time.time() - start_time) * 1000.0  # ms
        print(f"Inference time: {inference_time:.2f}ms on {'EdgeTPU' if self._using_coral else 'CPU'}")
        
        return detections
    
    def _detect_with_coral(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection using EdgeTPU."""
        from detection.tracker import BoundingBox
        
        # Perform inference on EdgeTPU
        detections = self.engine.detect_with_image(
            rgb_image, 
            threshold=self.threshold, 
            keep_aspect_ratio=True
        )
        
        # Process results
        results = []
        for det in detections:
            box = det.bounding_box
            x1, y1 = int(box[0][0]), int(box[0][1])
            x2, y2 = int(box[1][0]), int(box[1][1])
            
            w = x2 - x1
            h = y2 - y1
            
            class_id = det.label_id
            class_name = self.labels[class_id] if class_id < len(self.labels) else "unknown"
            confidence = det.score
            
            results.append({
                'bbox': BoundingBox(x1, y1, w, h, confidence, class_id),
                'score': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
        
        return results
    
    def _detect_with_tflite(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection using TFLite (CPU fallback)."""
        from detection.tracker import BoundingBox
        
        # Resize and normalize image according to model requirements
        input_shape = self.input_shape
        resized_image = cv2.resize(rgb_image, (input_shape[1], input_shape[2]))
        input_data = np.expand_dims(resized_image, axis=0)
        
        # Normalize image if needed (depends on model)
        input_data = input_data.astype(np.float32) / 255.0
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output tensors
        # This is model-specific and might need adjustment based on your model
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Process results
        results = []
        height, width = rgb_image.shape[:2]
        
        for i in range(len(scores)):
            if scores[i] >= self.threshold:
                # TFLite object detection models typically output normalized coords [0,1]
                # Convert to pixel coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * width)
                xmax = int(xmax * width)
                ymin = int(ymin * height)
                ymax = int(ymax * height)
                
                w = xmax - xmin
                h = ymax - ymin
                
                class_id = int(classes[i])
                class_name = self.labels[class_id] if class_id < len(self.labels) else "unknown"
                confidence = float(scores[i])
                
                results.append({
                    'bbox': BoundingBox(xmin, ymin, w, h, confidence, class_id),
                    'score': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return results

def verify_coral_edgetpu() -> bool:
    """
    Check if EdgeTPU is available and working.
    
    Returns:
        bool: True if EdgeTPU is available, False otherwise
    """
    try:
        from pycoral.utils import edgetpu
        devices = edgetpu.list_edge_tpus()
        if not devices:
            print("WARNING: No EdgeTPU devices found!")
            return False
        
        print(f"Found {len(devices)} EdgeTPU device(s):")
        for i, device in enumerate(devices):
            print(f"  Device {i+1}: {device}")
        return True
    except ImportError:
        print("WARNING: EdgeTPU libraries not found")
        return False
    except Exception as e:
        print(f"ERROR checking EdgeTPU: {e}")
        return False

def run_coral_diagnostic(model_path: Optional[str] = None, 
                         labels_path: Optional[str] = None, 
                         test_image_path: Optional[str] = None) -> bool:
    """
    Run a complete diagnostic test on the Google Coral EdgeTPU.
    
    Args:
        model_path: Path to a test model (model_edgetpu.tflite)
        labels_path: Path to labels file
        test_image_path: Path to a test image
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\nGoogle Coral EdgeTPU Diagnostic\n" + "="*30)
    
    # Step 1: Check for EdgeTPU device
    tpu_available = verify_coral_edgetpu()
    if not tpu_available:
        print("❌ EdgeTPU diagnostic failed: No devices detected.")
        return False
    
    # Step 2: Try loading libraries
    try:
        from pycoral.utils import edgetpu
        from pycoral.adapters import common
        from pycoral.adapters import detect
        print("✅ EdgeTPU libraries loaded successfully.")
    except ImportError as e:
        print(f"❌ Failed to import EdgeTPU libraries: {e}")
        return False
    
    # Step 3: Try loading a model if provided
    if model_path and os.path.exists(model_path):
        try:
            from edgetpu.detection.engine import DetectionEngine
            engine = DetectionEngine(model_path)
            print(f"✅ Successfully loaded model: {model_path}")
            
            # Step 4: Try inference if a test image is provided
            if test_image_path and os.path.exists(test_image_path):
                try:
                    import PIL.Image
                    image = PIL.Image.open(test_image_path).convert('RGB')
                    
                    # Run inference timing
                    start_time = time.time()
                    results = engine.detect_with_image(image, threshold=0.5)
                    inference_time = (time.time() - start_time) * 1000.0
                    
                    print(f"✅ Test inference completed in {inference_time:.2f}ms")
                    print(f"   Detected {len(results)} objects.")
                    
                    # If labels are available, print top results
                    if labels_path and os.path.exists(labels_path):
                        with open(labels_path, 'r') as f:
                            labels = [line.strip() for line in f.readlines()]
                        
                        for i, result in enumerate(results):
                            class_id = result.label_id
                            class_name = labels[class_id] if class_id < len(labels) else "unknown"
                            confidence = result.score
                            print(f"   {i+1}. {class_name} ({confidence:.2f})")
                except Exception as e:
                    print(f"❌ Test inference failed: {e}")
                    return False
        except Exception as e:
            print(f"❌ Failed to initialize EdgeTPU engine: {e}")
            return False
    
    print("\n✅ EdgeTPU diagnostic completed successfully!")
    return True

if __name__ == "__main__":
    # Run diagnostic when this file is executed directly
    run_coral_diagnostic(
        model_path="model_edgetpu.tflite" if os.path.exists("model_edgetpu.tflite") else None,
        labels_path="labels.txt" if os.path.exists("labels.txt") else None,
        test_image_path="test_image.jpg" if os.path.exists("test_image.jpg") else None
    ) 