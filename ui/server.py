#!/usr/bin/env python3
"""
Web server for the vision processing UI.
Provides camera feed streaming and configuration API.
"""

import os
import json
import cv2
import time
import logging
import threading
import base64
from typing import Dict, Any, Optional, List
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisionUI")

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'vision-processing-ui'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
config_path = os.environ.get('CONFIG_PATH', 'config/pc_config.json')
frame_buffer = None
latest_detections = []
latest_tracked_objects = []
selected_target = None
ui_settings = {
    'theme': 'dark',
    'show_processing_steps': True,
    'show_detection_boxes': True,
    'show_tracking_info': True,
    'show_selection_info': True,
    'show_debug_info': False,
    'stream_resolution': 'original'  # 'original', 'hd', 'sd'
}

# Lock for thread synchronization
frame_lock = threading.Lock()
config_lock = threading.Lock()


def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with config_lock:
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to JSON file."""
    try:
        with config_lock:
            # Create backup of current config
            if os.path.exists(config_path):
                backup_path = f"{config_path}.bak"
                with open(config_path, 'r') as f_in:
                    with open(backup_path, 'w') as f_out:
                        f_out.write(f_in.read())
            
            # Write new config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False


def get_config_sections() -> List[str]:
    """Get the main sections of the configuration."""
    config = load_config()
    return list(config.keys())


def encode_frame_to_jpeg(frame):
    """Encode OpenCV frame to JPEG for streaming."""
    if frame is None:
        return None
    
    # Scale frame based on UI settings
    resolution = ui_settings.get('stream_resolution', 'original')
    if resolution == 'hd':
        frame = cv2.resize(frame, (1280, 720))
    elif resolution == 'sd':
        frame = cv2.resize(frame, (640, 480))
        
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


def encode_frame_to_base64(frame):
    """Encode OpenCV frame to base64 for Socket.IO streaming."""
    if frame is None:
        return None
    
    # Scale frame based on UI settings
    resolution = ui_settings.get('stream_resolution', 'original')
    if resolution == 'hd':
        frame = cv2.resize(frame, (1280, 720))
    elif resolution == 'sd':
        frame = cv2.resize(frame, (640, 480))
        
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')


def update_frame(frame, detections=None, tracked_objects=None, target=None):
    """Update the current frame and processing results."""
    global frame_buffer, latest_detections, latest_tracked_objects, selected_target
    
    with frame_lock:
        frame_buffer = frame
        if detections is not None:
            latest_detections = detections
        if tracked_objects is not None:
            latest_tracked_objects = tracked_objects
        if target is not None:
            selected_target = target
            
        # Emit frame to connected clients using Socket.IO
        try:
            encoded_frame = encode_frame_to_base64(frame)
            if encoded_frame:
                socketio.emit('frame_update', {
                    'frame': encoded_frame,
                    'timestamp': time.time()
                })
        except Exception as e:
            logger.error(f"Error emitting frame: {e}")


# Flask routes
@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html', 
                          ui_settings=ui_settings,
                          config_sections=get_config_sections())


@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint for HTTP clients."""
    def generate():
        while True:
            with frame_lock:
                current_frame = frame_buffer
                
            if current_frame is not None:
                encoded_frame = encode_frame_to_jpeg(current_frame)
                if encoded_frame:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')
            else:
                # If no frame is available, send a blank frame
                blank_frame = np.zeros((480, 640, 3), np.uint8)
                blank_frame = cv2.putText(blank_frame, "No video feed available", (50, 240), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                encoded_blank = encode_frame_to_jpeg(blank_frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + encoded_blank + b'\r\n')
                
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/config', methods=['GET'])
def get_configuration():
    """API endpoint to get current configuration."""
    config = load_config()
    return jsonify(config)


@app.route('/config/<section>', methods=['GET'])
def get_config_section(section):
    """API endpoint to get specific configuration section."""
    config = load_config()
    if section in config:
        return jsonify(config[section])
    return jsonify({"error": f"Section {section} not found"}), 404


@app.route('/config/<section>', methods=['POST'])
def update_config_section(section):
    """API endpoint to update a specific configuration section."""
    try:
        config = load_config()
        if section not in config:
            return jsonify({"error": f"Section {section} not found"}), 404
            
        # Update the section
        updated_section = request.json
        config[section] = updated_section
        
        # Save the updated config
        if save_config(config):
            return jsonify({"success": True, "message": f"Updated {section} configuration"})
        else:
            return jsonify({"error": "Failed to save configuration"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/config/<section>/<key>', methods=['PUT'])
def update_config_value(section, key):
    """API endpoint to update a specific configuration value."""
    try:
        config = load_config()
        if section not in config:
            return jsonify({"error": f"Section {section} not found"}), 404
            
        # Update the specific key
        value = request.json.get('value')
        config[section][key] = value
        
        # Save the updated config
        if save_config(config):
            return jsonify({"success": True, "message": f"Updated {section}.{key} to {value}"})
        else:
            return jsonify({"error": "Failed to save configuration"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ui_settings', methods=['GET'])
def get_ui_settings():
    """API endpoint to get UI settings."""
    return jsonify(ui_settings)


@app.route('/ui_settings', methods=['POST'])
def update_ui_settings():
    """API endpoint to update UI settings."""
    global ui_settings
    try:
        new_settings = request.json
        ui_settings.update(new_settings)
        return jsonify({"success": True, "settings": ui_settings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/state', methods=['GET'])
def get_state():
    """API endpoint to get current processing state."""
    with frame_lock:
        state = {
            "has_video": frame_buffer is not None,
            "detections_count": len(latest_detections),
            "tracked_objects_count": len(latest_tracked_objects),
            "has_selected_target": selected_target is not None
        }
    return jsonify(state)


# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection to Socket.IO."""
    logger.info(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection from Socket.IO."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('get_frame')
def handle_get_frame():
    """Handle frame request from client."""
    with frame_lock:
        if frame_buffer is not None:
            encoded_frame = encode_frame_to_base64(frame_buffer)
            if encoded_frame:
                return {'frame': encoded_frame, 'timestamp': time.time()}
    return {'frame': None, 'timestamp': time.time()}


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    # Import numpy here to avoid circular import
    import numpy as np
    
    # Create a blank initial frame
    blank_frame = np.zeros((480, 640, 3), np.uint8)
    blank_frame = cv2.putText(blank_frame, "Vision UI Server Running", (50, 240), 
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    update_frame(blank_frame)
    
    # Run the server
    run_server(debug=True) 