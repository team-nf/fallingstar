#!/usr/bin/env python3

import os
import cv2
import time
import json
import argparse
import datetime
import numpy as np
import sys
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path to access detection package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from detection if needed
from detection.utils import fps_counter

def get_camera_list() -> List[int]:
    """
    Get a list of available camera devices.
    
    Returns:
        List[int]: List of available camera indices
    """
    available_cameras = []
    for i in range(10):  # Check camera indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def setup_camera(camera_id: int, width: int = 640, height: int = 480, 
                fps: int = 30, brightness: Optional[int] = None, 
                exposure: Optional[int] = None, 
                white_balance: Optional[str] = None) -> cv2.VideoCapture:
    """
    Set up a camera with the specified parameters.
    
    Args:
        camera_id: Camera index
        width: Camera frame width
        height: Camera frame height
        fps: Frames per second
        brightness: Camera brightness (0-100)
        exposure: Camera exposure (0-100)
        white_balance: White balance mode
        
    Returns:
        cv2.VideoCapture: Configured camera object
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness / 100.0)
    
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure / 100.0)
    
    if white_balance is not None and white_balance.lower() != "auto":
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
        try:
            temp = int(white_balance)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temp)
        except ValueError:
            # Not a number, might be a named preset
            pass
    
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera initialized:")
    print(f"  Resolution: {actual_width}x{actual_height} (requested: {width}x{height})")
    print(f"  FPS: {actual_fps} (requested: {fps})")
    
    # Warm up the camera
    for _ in range(5):
        cap.read()
    
    return cap

def create_output_structure(output_dir: str, class_names: List[str]) -> Dict[str, str]:
    """
    Create the directory structure for saving training data.
    
    Args:
        output_dir: Base output directory
        class_names: List of class names
        
    Returns:
        Dict[str, str]: Dictionary mapping class names to their directories
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create directory for each class
    class_dirs = {}
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        class_dirs[class_name] = class_dir
    
    # Create metadata directory
    metadata_dir = os.path.join(output_dir, "metadata")
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    
    return class_dirs

def generate_filename(prefix: str, class_name: str) -> str:
    """
    Generate a filename using timestamp and class name.
    
    Args:
        prefix: Prefix for the filename
        class_name: Class name
        
    Returns:
        str: Generated filename
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{class_name}_{timestamp}.jpg"

def save_frame(frame: np.ndarray, directory: str, class_name: str, 
              prefix: str = "image") -> str:
    """
    Save a frame to disk.
    
    Args:
        frame: The frame to save
        directory: Output directory
        class_name: Class name
        prefix: Filename prefix
        
    Returns:
        str: Path to the saved file
    """
    filename = generate_filename(prefix, class_name)
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, frame)
    return filepath

def save_metadata(metadata_dir: str, filepath: str, class_name: str, 
                 frame_time: float, extra_metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save metadata for a captured frame.
    
    Args:
        metadata_dir: Directory to save metadata
        filepath: Path to the image file
        class_name: Class name
        frame_time: Timestamp of frame capture
        extra_metadata: Additional metadata to save
        
    Returns:
        str: Path to the metadata file
    """
    # Extract filename
    filename = os.path.basename(filepath)
    metadata_filename = os.path.splitext(filename)[0] + ".json"
    metadata_filepath = os.path.join(metadata_dir, metadata_filename)
    
    # Create metadata
    metadata = {
        "image_file": filename,
        "class": class_name,
        "timestamp": frame_time,
        "datetime": datetime.datetime.fromtimestamp(frame_time).isoformat()
    }
    
    # Add any extra metadata
    if extra_metadata:
        metadata.update(extra_metadata)
    
    # Save metadata
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_filepath

def capture_training_data(camera_id: int, output_dir: str, class_names: List[str], 
                         width: int = 640, height: int = 480, fps: int = 30,
                         max_frames: Optional[int] = None,
                         capture_interval: float = 0.5) -> None:
    """
    Capture training data using a camera.
    
    Args:
        camera_id: Camera index
        output_dir: Output directory for images
        class_names: List of class names
        width: Camera frame width
        height: Camera frame height
        fps: Frames per second
        max_frames: Maximum number of frames to capture per class (None for unlimited)
        capture_interval: Minimum interval between captures (seconds)
    """
    # Set up camera
    cap = setup_camera(camera_id, width, height, fps)
    
    # Create output directories
    class_dirs = create_output_structure(output_dir, class_names)
    metadata_dir = os.path.join(output_dir, "metadata")
    
    # Initialize variables
    current_class_idx = 0
    current_class = class_names[current_class_idx]
    frame_count = {class_name: 0 for class_name in class_names}
    last_capture_time = 0
    auto_capture = False
    
    print("\nTraining data collection started")
    print("--------------------------------")
    print(f"Output directory: {output_dir}")
    print(f"Classes: {', '.join(class_names)}")
    print("\nControls:")
    print("  SPACE - Capture current frame")
    print("  A - Toggle auto-capture mode")
    print("  C - Cycle through classes")
    print("  +/- - Increase/decrease auto-capture interval")
    print("  Q or ESC - Quit")
    print("\nCurrent class:", current_class)
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                break
            
            current_time = time.time()
            
            # Create display frame with info
            display_frame = frame.copy()
            
            # Add overlay text
            cv2.putText(display_frame, f"Class: {current_class}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            count_text = f"Count: {frame_count[current_class]}"
            if max_frames:
                count_text += f" / {max_frames}"
            cv2.putText(display_frame, count_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            auto_text = f"Auto: {'ON' if auto_capture else 'OFF'}"
            if auto_capture:
                auto_text += f" (interval: {capture_interval:.1f}s)"
            cv2.putText(display_frame, auto_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Handle auto-capture
            should_capture = False
            if auto_capture and (current_time - last_capture_time) >= capture_interval:
                should_capture = True
            
            # Show frame
            cv2.imshow("Training Data Collection", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or Q to quit
                break
            elif key == ord(' '):  # Space to capture
                should_capture = True
            elif key == ord('a'):  # 'A' to toggle auto-capture
                auto_capture = not auto_capture
                print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
            elif key == ord('c'):  # 'C' to cycle through classes
                current_class_idx = (current_class_idx + 1) % len(class_names)
                current_class = class_names[current_class_idx]
                print(f"Current class: {current_class}")
            elif key == ord('+') or key == ord('='):  # + to increase interval
                capture_interval += 0.1
                print(f"Capture interval: {capture_interval:.1f}s")
            elif key == ord('-') or key == ord('_'):  # - to decrease interval
                capture_interval = max(0.1, capture_interval - 0.1)
                print(f"Capture interval: {capture_interval:.1f}s")
            
            # Capture frame if needed
            if should_capture:
                # Reset capture timer
                last_capture_time = current_time
                
                # Check if we've reached max frames for this class
                if max_frames and frame_count[current_class] >= max_frames:
                    print(f"Reached maximum frames ({max_frames}) for class {current_class}")
                    if auto_capture:
                        # Automatically move to next class
                        current_class_idx = (current_class_idx + 1) % len(class_names)
                        current_class = class_names[current_class_idx]
                        print(f"Switched to class: {current_class}")
                        continue
                
                # Save frame
                filepath = save_frame(frame, class_dirs[current_class], current_class)
                
                # Save metadata
                save_metadata(metadata_dir, filepath, current_class, current_time)
                
                # Update frame count
                frame_count[current_class] += 1
                print(f"Captured frame for {current_class} ({frame_count[current_class]})")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nTraining data collection summary:")
        print("--------------------------------")
        total_frames = sum(frame_count.values())
        print(f"Total frames captured: {total_frames}")
        for class_name in class_names:
            print(f"  {class_name}: {frame_count[class_name]} frames")
        print(f"\nData saved to: {output_dir}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Collect training data for object detection")
    parser.add_argument("--output-dir", default="training_data", help="Output directory for training data")
    parser.add_argument("--classes", default="object", help="Comma-separated list of class names")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    parser.add_argument("--fps", type=int, default=30, help="Camera frames per second")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to capture per class")
    parser.add_argument("--interval", type=float, default=0.5, help="Auto-capture interval in seconds")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")
    
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        cameras = get_camera_list()
        if cameras:
            print("Available cameras:")
            for i, camera_id in enumerate(cameras):
                cap = cv2.VideoCapture(camera_id)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                print(f"  {camera_id}: {width}x{height}")
        else:
            print("No cameras found")
        return
    
    # Parse class names
    class_names = [name.strip() for name in args.classes.split(",")]
    
    # Capture training data
    capture_training_data(
        camera_id=args.camera_id,
        output_dir=args.output_dir,
        class_names=class_names,
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_frames=args.max_frames,
        capture_interval=args.interval
    )

if __name__ == "__main__":
    main() 