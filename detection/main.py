#!/usr/bin/env python3

import os
import sys
import time
import cv2
import numpy as np
import argparse
import json
from typing import Dict, Any, Optional

# Import detection modules
from detection.detector import ObjectDetector, verify_coral_edgetpu, run_coral_diagnostic
from detection.tracker import SimpleTracker
from detection.networktables_manager import NetworkTablesManager
from detection.utils import (
    load_calibration, 
    undistort_image, 
    draw_detection_overlay, 
    fps_counter,
    find_latest_calibration_file,
    threaded_camera_stream,
    get_camera_list,
    save_diagnostic_image
)

# Default configuration
DEFAULT_CONFIG = {
    "camera": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "id": 0,
        "brightness": 50,
        "exposure": 50,
        "white_balance": "auto"
    },
    "model": {
        "path": "model_edgetpu.tflite",
        "labels": "labels.txt",
        "threshold": 0.5,
    },
    "tracking": {
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3
    },
    "networking": {
        "team_number": None,
        "server_ip": None,
        "table_name": "Vision"
    },
    "output": {
        "show_window": True,
        "stream_name": "Processed",
        "port": 1181
    },
    "calibration": {
        "enabled": False,
        "file": None
    }
}

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a file, falling back to defaults for missing values.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}, using defaults")
        return config
    
    try:
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            
        # Update configuration with values from file
        for section, values in file_config.items():
            if section in config:
                if isinstance(values, dict):
                    # Update section
                    config[section].update(values)
                else:
                    # Replace section
                    config[section] = values
        
        print(f"Loaded configuration from {config_file}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    return config

def save_config(config: Dict[str, Any], config_file: str):
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to save configuration
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved configuration to {config_file}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

def initialize_system(config: Dict[str, Any]):
    """
    Initialize the vision system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with system components
    """
    system = {}
    
    # Verify EdgeTPU is available
    has_edgetpu = verify_coral_edgetpu()
    system['has_edgetpu'] = has_edgetpu
    
    # Initialize NetworkTables
    team_number = config['networking']['team_number']
    server_ip = config['networking']['server_ip']
    table_name = config['networking']['table_name']
    
    if team_number or server_ip:
        system['nt_manager'] = NetworkTablesManager(
            team_number=team_number,
            server_ip=server_ip,
            table_name=table_name
        )
    else:
        print("WARNING: No team number or server IP configured, NetworkTables disabled")
        system['nt_manager'] = None
    
    # Initialize detector
    model_path = config['model']['path']
    labels_path = config['model']['labels']
    threshold = config['model']['threshold']
    
    if os.path.exists(model_path) and os.path.exists(labels_path):
        try:
            system['detector'] = ObjectDetector(
                model_path=model_path,
                labels_path=labels_path,
                threshold=threshold
            )
            print(f"Initialized detector with model {model_path}")
        except Exception as e:
            print(f"Error initializing detector: {e}")
            system['detector'] = None
    else:
        print(f"WARNING: Model or labels file not found: {model_path}, {labels_path}")
        system['detector'] = None
    
    # Initialize tracker
    system['tracker'] = SimpleTracker(
        max_age=config['tracking']['max_age'],
        min_hits=config['tracking']['min_hits'],
        iou_threshold=config['tracking']['iou_threshold']
    )
    
    # Initialize camera
    camera_config = config['camera']
    system['camera'] = threaded_camera_stream(
        camera_id=camera_config['id'],
        resolution=(camera_config['width'], camera_config['height']),
        fps=camera_config['fps']
    )
    system['camera']['start']()
    
    # Load camera calibration if enabled
    if config['calibration']['enabled']:
        calibration_file = config['calibration']['file']
        if not calibration_file:
            # Try to find latest calibration file
            calibration_file = find_latest_calibration_file()
        
        if calibration_file and os.path.exists(calibration_file):
            camera_matrix, dist_coeffs = load_calibration(calibration_file)
            if camera_matrix is not None and dist_coeffs is not None:
                system['calibration'] = {
                    'camera_matrix': camera_matrix,
                    'dist_coeffs': dist_coeffs,
                    'file': calibration_file
                }
                config['calibration']['file'] = calibration_file
            else:
                system['calibration'] = None
        else:
            print(f"WARNING: Calibration file not found: {calibration_file}")
            system['calibration'] = None
    else:
        system['calibration'] = None
    
    # Initialize FPS counter
    system['fps_counter'] = fps_counter(smoothing=0.9)
    
    # Initialize CameraServer output stream if needed
    if config['output']['stream_name']:
        try:
            from cscore import CameraServer
            width = camera_config['width']
            height = camera_config['height']
            
            system['cs_output'] = CameraServer.putVideo(
                config['output']['stream_name'], 
                width, 
                height
            )
            print(f"Initialized CameraServer output stream: {config['output']['stream_name']}")
        except ImportError:
            print("WARNING: cscore not available, CameraServer disabled")
            system['cs_output'] = None
    else:
        system['cs_output'] = None
    
    return system

def process_frame(frame: np.ndarray, system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single frame.
    
    Args:
        frame: Input frame
        system: System components
        
    Returns:
        Dictionary with processing results
    """
    results = {
        'original_frame': frame,
        'processed_frame': frame.copy(),
        'detections': [],
        'tracked_objects': {},
        'fps': 0.0
    }
    
    # Apply camera calibration if available
    if system['calibration']:
        frame = undistort_image(
            frame, 
            system['calibration']['camera_matrix'],
            system['calibration']['dist_coeffs']
        )
        results['processed_frame'] = frame.copy()
    
    # Detect objects
    if system['detector']:
        try:
            start_time = time.time()
            detections = system['detector'].detect(frame)
            inference_time = (time.time() - start_time) * 1000.0
            
            results['detections'] = detections
            results['inference_time'] = inference_time
        except Exception as e:
            print(f"Error in object detection: {e}")
            results['inference_time'] = 0
    
    # Track objects
    if system['tracker'] and results['detections']:
        try:
            tracked_objects = system['tracker'].update(results['detections'])
            results['tracked_objects'] = {
                track_id: {
                    'bbox': bbox,
                    'class_id': bbox.class_id,
                    'class_name': system['detector'].labels[bbox.class_id] if bbox.class_id < len(system['detector'].labels) else "unknown",
                    'confidence': bbox.confidence
                }
                for track_id, bbox in tracked_objects.items()
            }
        except Exception as e:
            print(f"Error in object tracking: {e}")
    
    # Update FPS counter
    system['fps_counter']['update']()
    results['fps'] = system['fps_counter']['get_fps']()
    
    # Draw results on frame
    results['display_frame'] = draw_detection_overlay(
        results['processed_frame'],
        results['tracked_objects'],
        show_fps=True,
        fps_value=results['fps']
    )
    
    return results

def run_vision_system(config: Dict[str, Any]):
    """
    Run the vision processing system.
    
    Args:
        config: Configuration dictionary
    """
    # Initialize system
    system = initialize_system(config)
    
    if system['detector'] is None:
        print("ERROR: Failed to initialize detector, exiting")
        return
    
    # Main processing loop
    try:
        print("Starting vision processing loop")
        
        while True:
            # Get frame from camera
            frame, error = system['camera']['get_frame']()
            
            if frame is None:
                print(f"Error getting frame: {error}")
                time.sleep(0.1)
                continue
            
            # Process frame
            results = process_frame(frame, system)
            
            # Publish results to NetworkTables
            if system['nt_manager']:
                system['nt_manager'].publish_detection_results(
                    results['tracked_objects'],
                    fps=results['fps'],
                    latency_ms=results.get('inference_time', 0)
                )
            
            # Output to CameraServer
            if system['cs_output']:
                try:
                    system['cs_output'].putFrame(results['display_frame'])
                except Exception as e:
                    print(f"Error sending frame to CameraServer: {e}")
            
            # Display results
            if config['output']['show_window']:
                cv2.imshow('FRC Vision', results['display_frame'])
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC key
                    print("ESC pressed, exiting")
                    break
                elif key == ord('s'):
                    # Save diagnostic image
                    save_diagnostic_image(results['display_frame'])
                elif key == ord('c'):
                    # Toggle calibration
                    config['calibration']['enabled'] = not config['calibration']['enabled']
                    print(f"Calibration {'enabled' if config['calibration']['enabled'] else 'disabled'}")
                    # Reload calibration
                    if config['calibration']['enabled']:
                        calibration_file = config['calibration']['file']
                        if calibration_file and os.path.exists(calibration_file):
                            camera_matrix, dist_coeffs = load_calibration(calibration_file)
                            if camera_matrix is not None and dist_coeffs is not None:
                                system['calibration'] = {
                                    'camera_matrix': camera_matrix,
                                    'dist_coeffs': dist_coeffs,
                                    'file': calibration_file
                                }
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        if system['camera']:
            system['camera']['stop']()
        
        if system['nt_manager']:
            system['nt_manager'].shutdown()
        
        if config['output']['show_window']:
            cv2.destroyAllWindows()
        
        print("Vision system shutdown complete")

def main():
    """Main entry point for the detection system."""
    parser = argparse.ArgumentParser(description='FRC Vision Processing System')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--team', type=int, help='FRC team number')
    parser.add_argument('--server', help='NetworkTables server IP')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--edgetpu-test', action='store_true', help='Run EdgeTPU diagnostic test')
    parser.add_argument('--no-window', action='store_true', help='Don\'t display window')
    parser.add_argument('--list-cameras', action='store_true', help='List available cameras and exit')
    
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        cameras = get_camera_list()
        print(f"Available cameras: {cameras}")
        return
    
    # Run EdgeTPU diagnostic if requested
    if args.edgetpu_test:
        run_coral_diagnostic()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.team:
        config['networking']['team_number'] = args.team
    
    if args.server:
        config['networking']['server_ip'] = args.server
    
    if args.camera is not None:
        config['camera']['id'] = args.camera
    
    if args.no_window:
        config['output']['show_window'] = False
    
    # Save updated configuration
    save_config(config, args.config)
    
    # Run vision system
    run_vision_system(config)

if __name__ == "__main__":
    main() 