#!/usr/bin/env python3
"""
Integration module for connecting the vision processing system with the UI.
This module provides a bridge between the main vision processing loop and the UI server.
"""

import os
import cv2
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisionIntegration")

# Global variables
ui_server = None
is_processing = False
processing_thread = None


def start_ui_server(host='0.0.0.0', port=5000, debug=False):
    """
    Start the UI server in a separate thread.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        debug: Whether to enable debug mode
    """
    global ui_server
    try:
        from ui.server import run_server
        
        # Start server in a thread
        server_thread = threading.Thread(
            target=run_server,
            args=(host, port, debug),
            daemon=True
        )
        server_thread.start()
        logger.info(f"UI server started at http://{host}:{port}")
        ui_server = server_thread
        return True
    except Exception as e:
        logger.error(f"Error starting UI server: {e}")
        return False


def update_ui_frame(frame, detections=None, tracked_objects=None, selected_target=None):
    """
    Update the UI with the current frame and processing results.
    
    Args:
        frame: Current video frame
        detections: List of detections
        tracked_objects: List of tracked objects
        selected_target: Selected target object
    """
    try:
        from ui.server import update_frame
        
        # Clone frame to avoid modification by other threads
        if frame is not None:
            frame_copy = frame.copy()
        else:
            frame_copy = None
            
        # Update UI
        update_frame(frame_copy, detections, tracked_objects, selected_target)
        return True
    except Exception as e:
        logger.error(f"Error updating UI frame: {e}")
        return False


def get_ui_settings():
    """Get current UI settings."""
    try:
        from ui.server import ui_settings
        return ui_settings.copy()
    except Exception as e:
        logger.error(f"Error getting UI settings: {e}")
        return {}


def create_diagnostic_image(frame, processing_steps):
    """
    Create a diagnostic image showing different processing steps.
    
    Args:
        frame: Original input frame
        processing_steps: Dictionary of named processing steps (frames)
        
    Returns:
        Composite image showing all processing steps
    """
    if frame is None:
        return None
        
    # Get original frame dimensions
    h, w = frame.shape[:2]
    
    # Determine grid layout based on number of steps
    num_steps = len(processing_steps) + 1  # +1 for original frame
    cols = min(3, num_steps)
    rows = (num_steps + cols - 1) // cols
    
    # Create composite image
    cell_h, cell_w = h // rows, w // cols
    composite = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    
    # Add original frame
    resized_frame = cv2.resize(frame, (cell_w, cell_h))
    composite[0:cell_h, 0:cell_w] = resized_frame
    
    # Add step label to original frame
    cv2.putText(composite, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add processing steps
    i = 1
    for name, step_frame in processing_steps.items():
        if step_frame is None:
            continue
            
        # Calculate position
        row = i // cols
        col = i % cols
        
        # Convert to color if grayscale
        if len(step_frame.shape) == 2:
            step_frame = cv2.cvtColor(step_frame, cv2.COLOR_GRAY2BGR)
            
        # Resize to fit cell
        resized_step = cv2.resize(step_frame, (cell_w, cell_h))
        
        # Place in composite
        y_start = row * cell_h
        x_start = col * cell_w
        composite[y_start:y_start+cell_h, x_start:x_start+cell_w] = resized_step
        
        # Add step label
        cv2.putText(composite, name, (x_start + 10, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        i += 1
    
    return composite


def run_processing_loop(config_path, frame_processor, stop_event):
    """
    Run the vision processing loop and update the UI.
    
    Args:
        config_path: Path to configuration file
        frame_processor: Function that processes frames and returns results
        stop_event: Threading event to signal stopping
    """
    global is_processing
    
    try:
        import json
        import cv2
        import numpy as np
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Set up camera
        camera_config = config.get('camera', {})
        cap = cv2.VideoCapture(camera_config.get('device', 0))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('height', 480))
        cap.set(cv2.CAP_PROP_FPS, camera_config.get('fps', 30))
        
        # Create a blank initial frame
        blank_frame = np.zeros((
            camera_config.get('height', 480),
            camera_config.get('width', 640), 
            3
        ), np.uint8)
        
        blank_frame = cv2.putText(
            blank_frame, 
            "Starting vision processing...", 
            (50, blank_frame.shape[0] // 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        # Update UI with initial frame
        update_ui_frame(blank_frame)
        
        is_processing = True
        logger.info("Vision processing loop started")
        
        # Main processing loop
        while not stop_event.is_set():
            # Read frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Process frame
            try:
                result = frame_processor(frame, config)
                
                # Extract results
                processed_frame = result.get('processed_frame', frame)
                detections = result.get('detections', [])
                tracked_objects = result.get('tracked_objects', [])
                selected_target = result.get('selected_target')
                
                # Create diagnostic view if processing steps are available
                processing_steps = result.get('processing_steps', {})
                if processing_steps and get_ui_settings().get('show_processing_steps', True):
                    diagnostic_frame = create_diagnostic_image(frame, processing_steps)
                    if diagnostic_frame is not None:
                        processed_frame = diagnostic_frame
                
                # Update UI
                update_ui_frame(
                    processed_frame, 
                    detections, 
                    tracked_objects, 
                    selected_target
                )
            except Exception as e:
                logger.error(f"Error in processing frame: {e}")
                # Update UI with original frame on error
                update_ui_frame(frame)
            
            # Limit processing rate
            time.sleep(0.01)  # Allow other threads to run
        
        # Clean up
        cap.release()
        logger.info("Vision processing loop stopped")
    
    except Exception as e:
        logger.error(f"Error in processing loop: {e}")
    
    finally:
        is_processing = False


def start_processing(config_path, frame_processor):
    """
    Start the vision processing loop in a separate thread.
    
    Args:
        config_path: Path to configuration file
        frame_processor: Function that processes frames and returns results
        
    Returns:
        True if started successfully, False otherwise
    """
    global processing_thread, is_processing
    
    if is_processing:
        logger.warning("Processing already running")
        return False
    
    try:
        # Create stop event
        stop_event = threading.Event()
        
        # Start processing thread
        thread = threading.Thread(
            target=run_processing_loop,
            args=(config_path, frame_processor, stop_event),
            daemon=True
        )
        thread.start()
        
        # Store thread and stop event
        processing_thread = {
            'thread': thread,
            'stop_event': stop_event
        }
        
        return True
    except Exception as e:
        logger.error(f"Error starting processing: {e}")
        return False


def stop_processing():
    """Stop the vision processing loop."""
    global processing_thread, is_processing
    
    if not is_processing or processing_thread is None:
        logger.warning("No processing running")
        return False
    
    try:
        # Signal thread to stop
        processing_thread['stop_event'].set()
        
        # Wait for thread to finish
        processing_thread['thread'].join(timeout=5.0)
        
        # Reset
        processing_thread = None
        is_processing = False
        
        logger.info("Processing stopped")
        return True
    except Exception as e:
        logger.error(f"Error stopping processing: {e}")
        return False


# Import numpy module at the bottom to avoid import errors
try:
    import numpy as np
except ImportError:
    logger.error("NumPy not available, some features may not work") 