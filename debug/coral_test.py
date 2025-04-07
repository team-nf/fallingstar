#!/usr/bin/env python3
"""
Google Coral EdgeTPU diagnostic test utility.
Tests if the Coral TPU is available, loads a model, and runs inference.
"""

import os
import sys
import time
import argparse
import numpy as np
import platform
import subprocess
import importlib.util

def check_system_info():
    """Gather basic system information."""
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine()
    }
    
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    return system_info

def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        ("PIL", "Pillow"),
        ("tflite_runtime", "tflite-runtime"),
        ("pycoral", "pycoral")
    ]
    
    missing_packages = []
    installed_packages = []
    
    print("\nChecking required packages:")
    for package_name, install_name in required_packages:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            print(f"  {package_name}: NOT FOUND")
            missing_packages.append(install_name)
        else:
            # Try to get version if possible
            try:
                if package_name == "PIL":
                    import PIL
                    version = PIL.__version__
                elif package_name == "tflite_runtime":
                    import tflite_runtime
                    version = tflite_runtime.__version__
                elif package_name == "pycoral":
                    import pycoral
                    version = pycoral.__version__
                else:
                    version = "unknown"
                
                print(f"  {package_name}: Found (version {version})")
                installed_packages.append((package_name, version))
            except (ImportError, AttributeError):
                print(f"  {package_name}: Found (version unknown)")
                installed_packages.append((package_name, "unknown"))
    
    if missing_packages:
        print("\nSome required packages are missing. Install them with:")
        for package in missing_packages:
            print(f"  pip install {package}")
        return False
    
    return True

def check_coral_device():
    """Check if a Coral TPU device is connected."""
    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        
        print("\nChecking for Coral TPU devices:")
        devices = list_edge_tpus()
        
        if not devices:
            print("  No Coral TPU devices found")
            return None
        
        for i, device in enumerate(devices):
            print(f"  Device {i+1}: {device}")
        
        return devices[0]
    
    except ImportError:
        print("  Unable to check for Coral TPU devices: pycoral not installed")
        return None
    except Exception as e:
        print(f"  Error checking for Coral TPU devices: {e}")
        return None

def download_test_model(model_dir="models"):
    """Download a test model if it doesn't exist."""
    import urllib.request
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_url = "https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite"
    model_path = os.path.join(model_dir, "mobilenet_v2_1.0_224_quant_edgetpu.tflite")
    
    if not os.path.exists(model_path):
        print(f"\nDownloading test model to {model_path}...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("  Download complete")
        except Exception as e:
            print(f"  Error downloading model: {e}")
            return None
    else:
        print(f"\nUsing existing model at {model_path}")
    
    # Also download labels
    labels_url = "https://github.com/google-coral/test_data/raw/master/imagenet_labels.txt"
    labels_path = os.path.join(model_dir, "imagenet_labels.txt")
    
    if not os.path.exists(labels_path):
        print(f"Downloading labels to {labels_path}...")
        try:
            urllib.request.urlretrieve(labels_url, labels_path)
            print("  Download complete")
        except Exception as e:
            print(f"  Error downloading labels: {e}")
    
    return model_path

def load_test_model(model_path):
    """Load a test model and report success or failure."""
    try:
        from pycoral.utils import edgetpu
        from pycoral.adapters import common
        
        print("\nLoading test model...")
        start_time = time.time()
        
        interpreter = edgetpu.make_interpreter(model_path)
        interpreter.allocate_tensors()
        
        load_time = time.time() - start_time
        print(f"  Model loaded successfully in {load_time:.2f} seconds")
        
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("  Model details:")
        print(f"    Input shape: {input_details[0]['shape']}")
        print(f"    Input type: {input_details[0]['dtype']}")
        print(f"    Output shape: {output_details[0]['shape']}")
        print(f"    Output type: {output_details[0]['dtype']}")
        
        return interpreter, input_details, output_details
    
    except ImportError:
        print("  Unable to load model: pycoral not installed")
        return None, None, None
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None, None, None

def run_inference_test(interpreter, input_details, output_details, num_runs=100):
    """Run inference test with random input data."""
    if not interpreter:
        return
    
    try:
        print("\nRunning inference test...")
        
        # Create random input data
        input_shape = input_details[0]['shape']
        input_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        
        # Warm-up run
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Test multiple runs
        print(f"  Running {num_runs} inferences...")
        start_time = time.time()
        
        for _ in range(num_runs):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        fps = num_runs / total_time
        
        print(f"  Results:")
        print(f"    Total time: {total_time:.2f} seconds")
        print(f"    Average inference time: {avg_time*1000:.2f} ms")
        print(f"    Throughput: {fps:.2f} FPS")
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        top_result = np.argmax(output_data)
        
        # Try to get label
        try:
            labels_path = os.path.join("models", "imagenet_labels.txt")
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    labels = [line.strip() for line in f.readlines()]
                print(f"    Top class: {top_result} ({labels[top_result]})")
            else:
                print(f"    Top class: {top_result}")
        except Exception:
            print(f"    Top class: {top_result}")
        
        return {
            "inference_time_ms": avg_time * 1000,
            "fps": fps,
            "runs": num_runs
        }
    
    except Exception as e:
        print(f"  Error running inference: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test Google Coral EdgeTPU")
    parser.add_argument("--model", help="Path to a custom EdgeTPU model")
    parser.add_argument("--runs", type=int, default=100, help="Number of inference runs")
    
    args = parser.parse_args()
    
    print("Google Coral EdgeTPU Test Utility")
    print("================================\n")
    
    # Check system info
    check_system_info()
    
    # Check required packages
    if not check_required_packages():
        print("\nPlease install the missing packages and try again.")
        return
    
    # Check for Coral device
    device = check_coral_device()
    if not device:
        print("\nNo Coral EdgeTPU found. Is it connected properly?")
        print("Things to check:")
        print("  1. The device is connected via USB")
        print("  2. The device has power (should have LED indicator)")
        print("  3. Proper drivers are installed (especially on Windows)")
        return
    
    # Get model path
    model_path = args.model
    if not model_path:
        model_path = download_test_model()
        if not model_path:
            print("\nFailed to get a test model. Please provide a model path.")
            return
    
    # Load model
    interpreter, input_details, output_details = load_test_model(model_path)
    if not interpreter:
        print("\nFailed to load the model. The EdgeTPU may not be functioning correctly.")
        return
    
    # Run inference test
    result = run_inference_test(interpreter, input_details, output_details, num_runs=args.runs)
    if not result:
        print("\nFailed to run inference test.")
        return
    
    # Final summary
    print("\nTest Summary:")
    print("  Coral EdgeTPU: Detected and working")
    print(f"  Inference performance: {result['fps']:.2f} FPS ({result['inference_time_ms']:.2f} ms per inference)")
    
    if result['fps'] < 30:
        print("\nNote: Performance seems lower than expected. This could be due to:")
        print("  - USB connection issues (try a different port or cable)")
        print("  - System load (other processes using CPU/memory)")
        print("  - Thermal throttling (check if the device is hot)")
    
    print("\nTest completed successfully.")

if __name__ == "__main__":
    main() 