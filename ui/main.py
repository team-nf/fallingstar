#!/usr/bin/env python3
"""
Main entry point for the Vision Processing UI.
This script initializes the UI server and integrates it with the vision processing system.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisionUI")

def ensure_ui_directories():
    """Ensure all necessary directories exist."""
    # Define the directories to ensure
    directories = [
        "ui/static",
        "ui/static/css",
        "ui/static/js",
        "ui/static/images",
        "ui/templates"
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create placeholder logo if it doesn't exist
    logo_path = "ui/static/images/logo.png"
    if not os.path.exists(logo_path):
        try:
            # Try to generate a simple placeholder logo
            import numpy as np
            import cv2
            
            # Create a black image
            img = np.zeros((200, 200, 3), np.uint8)
            
            # Add some text
            cv2.putText(img, "Vision", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 50), 2)
            cv2.putText(img, "UI", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 50), 2)
            
            # Add a circle
            cv2.circle(img, (100, 100), 80, (0, 255, 50), 2)
            
            # Get team number from environment for logo, if available
            team_number = os.environ.get('TEAM_NUMBER', '9029')
            cv2.putText(img, f"Team {team_number}", (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 50), 2)
            
            # Save the image
            cv2.imwrite(logo_path, img)
            logger.info(f"Created placeholder logo at {logo_path}")
        except Exception as e:
            logger.warning(f"Could not create placeholder logo: {e}")
            # Create an empty file as fallback
            with open(logo_path, 'wb') as f:
                f.write(b'')


def create_dummy_frame_processor():
    """
    Create a dummy frame processor for testing the UI.
    
    Returns:
        A function that processes frames for testing
    """
    import cv2
    import numpy as np
    import time
    
    def processor(frame, config):
        """
        Process a frame for UI testing.
        
        Args:
            frame: Input video frame
            config: Configuration dictionary
            
        Returns:
            Dictionary with processing results
        """
        # Create a copy of the frame
        processed = frame.copy()
        
        # Add a timestamp
        cv2.putText(
            processed,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add some random detections
        detections = []
        for i in range(3):
            # Random position and size
            x = np.random.randint(50, frame.shape[1] - 100)
            y = np.random.randint(50, frame.shape[0] - 100)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            
            # Draw rectangle
            cv2.rectangle(processed, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add to detections
            detections.append({
                'id': i,
                'class_id': 0,
                'class_name': 'object',
                'score': np.random.uniform(0.5, 1.0),
                'bbox': {
                    'xmin': x,
                    'ymin': y,
                    'xmax': x + w,
                    'ymax': y + h
                }
            })
        
        # Create tracking objects (same as detections for simplicity)
        tracked_objects = detections.copy()
        
        # Select a random detection as the target
        selected_target = tracked_objects[0] if tracked_objects else None
        
        # Create processing steps for demonstration
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        
        # Convert edges to color for display
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        processing_steps = {
            "Grayscale": gray,
            "Edges": edges_color,
            "Blur": blur
        }
        
        return {
            'processed_frame': processed,
            'detections': detections,
            'tracked_objects': tracked_objects,
            'selected_target': selected_target,
            'processing_steps': processing_steps
        }
    
    return processor


def main():
    """Main entry point for the vision UI."""
    parser = argparse.ArgumentParser(description="Vision Processing UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=None, help="Port to run the server on")
    parser.add_argument("--config", default="config/pc_config.json", help="Path to configuration file")
    parser.add_argument("--test", action="store_true", help="Run in test mode with a dummy processor")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set environment variable for config path
    os.environ['CONFIG_PATH'] = args.config
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure UI directories exist
    ensure_ui_directories()
    
    # Determine port from TEAM_NUMBER environment variable if not specified
    port = args.port
    if port is None:
        team_number = os.environ.get('TEAM_NUMBER', '9029')
        try:
            port = int(team_number)
            logger.info(f"Using team number {team_number} as port")
        except ValueError:
            port = 9029
            logger.info(f"Invalid team number format, using default port {port}")
    
    try:
        # Import integration module
        from ui.integration import start_ui_server, start_processing
        
        # Start UI server
        if start_ui_server(host=args.host, port=port, debug=args.debug):
            logger.info(f"UI server started at http://{args.host}:{port}")
            
            # Run in test mode if requested
            if args.test:
                logger.info("Running in test mode with dummy processor")
                # Create dummy processor
                processor = create_dummy_frame_processor()
                
                # Start vision processing
                if start_processing(args.config, processor):
                    logger.info("Test processing started")
                else:
                    logger.error("Failed to start test processing")
            else:
                logger.info("UI server ready for integration with vision system")
                logger.info("To start processing, call start_processing() from the main script")
            
            # Keep the script running
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
        else:
            logger.error("Failed to start UI server")
    
    except Exception as e:
        logger.error(f"Error starting UI: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 