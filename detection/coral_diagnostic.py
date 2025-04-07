#!/usr/bin/env python3

import os
import time
import argparse
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Any, Optional

def verify_coral_edgetpu() -> bool:
    """
    Check if EdgeTPU is available and working.
    
    Returns:
        bool: True if EdgeTPU is available, False otherwise
    """
    print("\n=== EdgeTPU AVAILABILITY CHECK ===")
    
    try:
        from pycoral.utils import edgetpu
        devices = edgetpu.list_edge_tpus()
        if not devices:
            print("❌ No EdgeTPU devices found!")
            return False
        
        print(f"✅ Found {len(devices)} EdgeTPU device(s):")
        for i, device in enumerate(devices):
            print(f"   Device {i+1}: {device}")
        return True
    except ImportError:
        print("❌ EdgeTPU libraries not found!")
        print("   Make sure you have installed the required libraries:")
        print("   - pycoral")
        print("   - tflite_runtime")
        return False
    except Exception as e:
        print(f"❌ ERROR checking EdgeTPU: {e}")
        return False

def check_libraries() -> bool:
    """
    Check if all required libraries are installed.
    
    Returns:
        bool: True if all libraries are available, False otherwise
    """
    print("\n=== LIBRARY AVAILABILITY CHECK ===")
    required_libraries = [
        ("pycoral.utils.edgetpu", "EdgeTPU utilities"),
        ("pycoral.adapters.common", "EdgeTPU adapters"),
        ("pycoral.adapters.detect", "EdgeTPU detection adapters"),
        ("tflite_runtime.interpreter", "TFLite runtime"),
        ("PIL.Image", "Pillow image library"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV")
    ]
    
    all_available = True
    
    for module_path, description in required_libraries:
        try:
            # Try to import the module
            module_parts = module_path.split('.')
            
            if len(module_parts) == 1:
                module = __import__(module_parts[0])
            else:
                module = __import__(module_parts[0], fromlist=[module_parts[-1]])
                
                for part in module_parts[1:]:
                    module = getattr(module, part)
            
            print(f"✅ {description} available ({module_path})")
        except ImportError:
            print(f"❌ {description} NOT available ({module_path})")
            all_available = False
        except Exception as e:
            print(f"❌ Error checking {description}: {e}")
            all_available = False
    
    return all_available

def test_model_loading(model_path: str) -> bool:
    """
    Test loading a TFLite model on EdgeTPU.
    
    Args:
        model_path: Path to EdgeTPU model file
        
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    print(f"\n=== MODEL LOADING TEST: {model_path} ===")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Check if file is a valid TFLite model
        with open(model_path, 'rb') as f:
            header = f.read(8)
            is_tflite = header[:4] == b'TFL3'
            
        if not is_tflite:
            print(f"❌ Not a valid TFLite model: {model_path}")
            return False
        
        # Check if model is compiled for EdgeTPU
        is_edgetpu = '_edgetpu' in model_path
        if not is_edgetpu:
            print(f"⚠️ Model name does not contain '_edgetpu'. May not be compiled for EdgeTPU.")
        
        # Try to load the model
        from edgetpu.detection.engine import DetectionEngine
        start_time = time.time()
        engine = DetectionEngine(model_path)
        load_time = (time.time() - start_time) * 1000.0
        
        print(f"✅ Model loaded successfully in {load_time:.2f}ms")
        
        # Print model info
        try:
            print("   Model information:")
            print(f"   - Size: {os.path.getsize(model_path)/1024:.1f} KB")
            if hasattr(engine, 'get_input_tensor_shape'):
                input_shape = engine.get_input_tensor_shape()
                print(f"   - Input shape: {input_shape}")
        except Exception as e:
            print(f"   - Unable to get detailed model info: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_inference(model_path: str, labels_path: Optional[str] = None, 
                 image_path: Optional[str] = None) -> bool:
    """
    Run test inference on the EdgeTPU.
    
    Args:
        model_path: Path to EdgeTPU model file
        labels_path: Path to labels file (optional)
        image_path: Path to test image (optional)
        
    Returns:
        bool: True if inference succeeded, False otherwise
    """
    print("\n=== INFERENCE TEST ===")
    
    # Load labels if available
    labels = None
    if labels_path and os.path.exists(labels_path):
        try:
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(labels)} labels from {labels_path}")
        except Exception as e:
            print(f"⚠️ Failed to load labels: {e}")
    
    # Load or create test image
    if image_path and os.path.exists(image_path):
        try:
            input_image = Image.open(image_path).convert('RGB')
            print(f"✅ Loaded test image from {image_path}")
        except Exception as e:
            print(f"❌ Failed to load test image: {e}")
            print("   Creating a test pattern instead...")
            input_image = Image.new('RGB', (300, 300), color=(128, 128, 128))
    else:
        print("⚠️ No test image provided, creating test pattern")
        input_image = Image.new('RGB', (300, 300), color=(128, 128, 128))
        
        # Draw some shapes on the test pattern
        draw = Image.new('RGB', (300, 300), color=(0, 0, 0))
        cv_image = np.array(draw)
        cv2.rectangle(cv_image, (50, 50), (200, 200), (0, 255, 0), 2)
        cv2.circle(cv_image, (150, 150), 50, (255, 0, 0), -1)
        input_image = Image.fromarray(cv_image)
    
    try:
        # Run inference
        from edgetpu.detection.engine import DetectionEngine
        engine = DetectionEngine(model_path)
        
        # Measure inference time
        start_time = time.time()
        results = engine.detect_with_image(
            input_image, 
            threshold=0.25,
            keep_aspect_ratio=True, 
            relative_coord=False,
            top_k=10
        )
        inference_time = (time.time() - start_time) * 1000.0
        
        print(f"✅ Inference completed in {inference_time:.2f}ms")
        print(f"   Detected {len(results)} objects.")
        
        # Show results
        if len(results) > 0:
            print("\n   Detection results:")
            for i, obj in enumerate(results):
                label_id = obj.label_id
                label = labels[label_id] if labels and label_id < len(labels) else f"Class {label_id}"
                print(f"   {i+1}. {label} (Score: {obj.score:.4f})")
                
                box = obj.bounding_box
                x1, y1 = int(box[0][0]), int(box[0][1])
                x2, y2 = int(box[1][0]), int(box[1][1])
                print(f"      Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Save annotated image for verification
        if len(results) > 0:
            try:
                # Create a copy of input image and draw detection results
                cv_image = np.array(input_image)
                for obj in results:
                    box = obj.bounding_box
                    x1, y1 = int(box[0][0]), int(box[0][1])
                    x2, y2 = int(box[1][0]), int(box[1][1])
                    
                    # Draw bounding box
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label_id = obj.label_id
                    label = labels[label_id] if labels and label_id < len(labels) else f"Class {label_id}"
                    label_text = f"{label} ({obj.score:.2f})"
                    
                    cv2.putText(cv_image, label_text, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save the annotated image
                output_path = "diagnostic_output.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
                print(f"\n✅ Saved annotated image to {output_path}")
            except Exception as e:
                print(f"⚠️ Failed to save annotated image: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def run_performance_test(model_path: str, num_iterations: int = 100) -> bool:
    """
    Run performance test on the EdgeTPU.
    
    Args:
        model_path: Path to EdgeTPU model file
        num_iterations: Number of inference iterations to run
        
    Returns:
        bool: True if test succeeded, False otherwise
    """
    print(f"\n=== PERFORMANCE TEST ({num_iterations} iterations) ===")
    
    try:
        from edgetpu.detection.engine import DetectionEngine
        engine = DetectionEngine(model_path)
        
        # Create a test image
        input_image = Image.new('RGB', (300, 300), color=(128, 128, 128))
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            engine.detect_with_image(input_image, threshold=0.5)
        
        # Run performance test
        print(f"Running {num_iterations} iterations...")
        inference_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            engine.detect_with_image(input_image, threshold=0.5)
            inference_time = (time.time() - start_time) * 1000.0
            inference_times.append(inference_time)
            
            # Show progress
            if (i+1) % 10 == 0:
                print(f"Completed {i+1}/{num_iterations} iterations")
        
        # Calculate statistics
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        
        print("\nPerformance results:")
        print(f"   Average inference time: {avg_time:.2f}ms ({1000/avg_time:.1f} FPS)")
        print(f"   Min inference time: {min_time:.2f}ms ({1000/min_time:.1f} FPS)")
        print(f"   Max inference time: {max_time:.2f}ms ({1000/max_time:.1f} FPS)")
        
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main entry point for the diagnostic script."""
    parser = argparse.ArgumentParser(description='Google Coral EdgeTPU Diagnostic Tool')
    parser.add_argument('--model', default='model_edgetpu.tflite', help='Path to EdgeTPU model file')
    parser.add_argument('--labels', default='labels.txt', help='Path to labels file')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--perf-test', action='store_true', help='Run performance test')
    parser.add_argument('--perf-iterations', type=int, default=100, help='Number of iterations for performance test')
    
    args = parser.parse_args()
    
    print("=================================================")
    print(" Google Coral EdgeTPU Diagnostic Tool")
    print("=================================================")
    
    # Check if EdgeTPU is available
    has_edgetpu = verify_coral_edgetpu()
    
    # Check if required libraries are available
    has_libraries = check_libraries()
    
    # Only continue if EdgeTPU and libraries are available
    if has_edgetpu and has_libraries:
        # Test model loading
        model_ok = test_model_loading(args.model)
        
        # Test inference
        if model_ok:
            inference_ok = test_inference(args.model, args.labels, args.image)
            
            # Run performance test if requested
            if args.perf_test and inference_ok:
                run_performance_test(args.model, args.perf_iterations)
    
    print("\n=================================================")
    print(" Diagnostic Summary")
    print("=================================================")
    print(f"EdgeTPU available: {'✅ YES' if has_edgetpu else '❌ NO'}")
    print(f"Required libraries: {'✅ All available' if has_libraries else '❌ Some missing'}")
    
    if has_edgetpu and has_libraries:
        print(f"Model loading: {'✅ Success' if model_ok else '❌ Failed'}")
        if model_ok:
            print(f"Inference test: {'✅ Success' if inference_ok else '❌ Failed'}")
            if args.perf_test and inference_ok:
                print(f"Performance test: {'✅ Completed'}")
    
    print("\nDiagnostic complete.")

if __name__ == "__main__":
    main() 