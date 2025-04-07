import os
import sys
import json
import logging
import argparse
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_socketio import SocketIO, emit
import threading
import time
import cv2
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(__file__), 'build/static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'build'))
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
config = {
    'team_number': os.environ.get('TEAM_NUMBER', '9029'),
    'port': int(os.environ.get('PORT', 9029)),
    'config_path': os.environ.get('CONFIG_PATH', 'config/pc_config.json'),
    'enable_ui': os.environ.get('ENABLE_UI', 'true').lower() == 'true'
}

# Global variables for camera and processing
camera = None
frame_buffer = None
frame_lock = threading.Lock()
processing_active = False

class Camera:
    def __init__(self):
        self.cap = None
        self.connected = False
        
    def connect(self, device_id=0):
        try:
            self.cap = cv2.VideoCapture(device_id)
            self.connected = self.cap.isOpened()
            if self.connected:
                logger.info(f"Connected to camera {device_id}")
            else:
                logger.error(f"Failed to connect to camera {device_id}")
        except Exception as e:
            logger.error(f"Error connecting to camera: {e}")
            self.connected = False
    
    def disconnect(self):
        if self.cap:
            self.cap.release()
        self.connected = False
        
    def get_frame(self):
        if not self.connected or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame

def process_frames():
    global camera, frame_buffer, processing_active
    
    fps_count = 0
    fps = 0
    fps_time = time.time()
    
    while processing_active:
        if camera and camera.connected:
            frame = camera.get_frame()
            
            if frame is not None:
                # Process frame (add detection, etc.)
                
                # Calculate FPS
                fps_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_count
                    fps_count = 0
                    fps_time = time.time()
                    socketio.emit('stream_data', {'fps': fps})
                
                # Add FPS text to frame
                cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update frame buffer
                with frame_lock:
                    _, encoded_frame = cv2.imencode('.jpg', frame)
                    frame_buffer = encoded_frame.tobytes()
        
        time.sleep(0.01)  # Small sleep to avoid excessive CPU usage

def generate_frames():
    global frame_buffer
    
    while True:
        if frame_buffer is not None:
            with frame_lock:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
        else:
            # If no frame, return a blank frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, encoded_frame = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

# API Routes
@app.route('/api/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/config', methods=['GET'])
def get_config():
    # Read config file
    try:
        with open(config['config_path'], 'r') as f:
            config_data = json.load(f)
        return jsonify(config_data)
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    try:
        data = request.json
        with open(config['config_path'], 'w') as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({"error": str(e)}), 500

# React App Routing (for serving the React app)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # For development, show a simple template with information about the server
        # when React development server is running separately
        if os.environ.get('NODE_ENV') == 'development':
            return render_template('index.html')
        # In production, serve the React build
        else:
            logger.info("Serving React app from build directory")
            return render_template('index.html')

# Socket.IO Events
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    socketio.emit('connection_status', {'connected': True})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

def initialize_camera():
    global camera, processing_active
    
    # Initialize camera
    camera = Camera()
    camera.connect(0)  # Connect to default camera
    
    # Start processing thread
    processing_active = True
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Vision Processing Server')
    parser.add_argument('--config', type=str, help='Path to config file', default='config/pc_config.json')
    parser.add_argument('--port', type=int, help='Server port', default=9029)
    parser.add_argument('--no-ui', action='store_true', help='Disable UI')
    parser.add_argument('--react-port', type=int, help='React dev server port', default=3000)
    return parser.parse_args()

def main():
    global config
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Update config with command line arguments
    if args.config:
        config['config_path'] = args.config
    if args.port:
        config['port'] = args.port
    if args.no_ui:
        config['enable_ui'] = False
    
    # Initialize camera and processing
    initialize_camera()
    
    # Start the server
    host = '0.0.0.0' if config['enable_ui'] else 'localhost'
    port = config['port']
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Config path: {config['config_path']}")
    
    if os.environ.get('NODE_ENV') == 'development':
        logger.info(f"Development mode: React app available at http://localhost:{args.react_port}")
    
    try:
        socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        # Cleanup
        global processing_active
        processing_active = False
        if camera:
            camera.disconnect()

if __name__ == '__main__':
    main() 