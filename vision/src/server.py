#!/usr/bin/env python3
import os
import json
import time
import base64
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np

# Import our vision modules
from get_cameras import get_available_cameras, create_mock_camera

app = Flask(__name__)
CORS(app)

# Global variables
cameras = {}  # Store open camera connections
config = {}   # Store configuration
mock_frame = None  # Store the mock camera frame

# Config path
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'settings.json')

# Create config directory if it doesn't exist
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

# Load config if it exists
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
else:
    # Default config
    config = {
        "camera": {
            "selectedCamera": "",
            "brightness": 50,
            "contrast": 50,
            "exposure": 50
        },
        "calibration": {
            "chessboardSize": {"width": 9, "height": 6},
            "squareSize": 25
        },
        "detection": {},
        "tracking": {},
        "selection": {},
        "pnp": {}
    }
    # Save default config
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# Initialize mock camera
mock_frame = create_mock_camera()

@app.route('/api/cameras', methods=['GET'])
def list_cameras():
    """List all available camera devices"""
    try:
        available_cameras = get_available_cameras()
        return jsonify(available_cameras)
    except Exception as e:
        print(f"Error listing cameras: {str(e)}")
        # Always return at least the mock camera in case of error
        return jsonify([{
            "id": 999,
            "name": "Mock Camera (error listing cameras)",
            "resolution": "640x480"
        }])

@app.route('/api/camera/<int:camera_id>/stream', methods=['GET'])
def camera_stream(camera_id):
    """Get a single frame from the specified camera"""
    try:
        print(f"Request for camera {camera_id} stream")
        
        # Special handling for mock camera
        if camera_id == 999:
            # Use mock camera frame
            global mock_frame
            frame = mock_frame
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                return jsonify({"error": "Failed to encode mock frame"}), 500
            
            # Convert to base64 for JSON response
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Return the frame data
            return jsonify({
                "camera_id": camera_id,
                "timestamp": time.time(),
                "frame": frame_data
            })
        
        if camera_id not in cameras:
            # Open the camera
            print(f"Opening camera {camera_id}")
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"Could not open camera {camera_id}")
                # Try a slightly different approach
                cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
                if not cap.isOpened():
                    print(f"Still could not open camera {camera_id}")
                    # Fallback to mock camera
                    print("Falling back to mock camera")
                    return camera_stream(999)
            
            # Apply settings if available
            if "camera" in config and config["camera"]["selectedCamera"] != "":
                try:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, config["camera"]["brightness"] / 100)
                    cap.set(cv2.CAP_PROP_CONTRAST, config["camera"]["contrast"] / 100)
                    cap.set(cv2.CAP_PROP_EXPOSURE, config["camera"]["exposure"] / 100)
                except Exception as e:
                    print(f"Error setting camera properties: {e}")
                    
            cameras[camera_id] = cap
            print(f"Successfully opened camera {camera_id}")
        
        # Get frame
        print(f"Reading frame from camera {camera_id}")
        ret, frame = cameras[camera_id].read()
        if not ret:
            print(f"Failed to read frame from camera {camera_id}")
            # Try to reopen the camera
            cameras[camera_id].release()
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print("Failed to reopen camera, falling back to mock camera")
                return camera_stream(999)
            
            cameras[camera_id] = cap
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame even after reopening, falling back to mock camera")
                return camera_stream(999)
            
            print(f"Successfully reopened camera {camera_id}")
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({"error": "Failed to encode frame"}), 500
        
        # Convert to base64 for JSON response
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Return the frame data
        return jsonify({
            "camera_id": camera_id,
            "timestamp": time.time(),
            "frame": frame_data
        })
    except Exception as e:
        print(f"Error in camera_stream: {str(e)}")
        print("Falling back to mock camera due to exception")
        # Fallback to mock camera in case of any exception
        try:
            return camera_stream(999)
        except Exception as inner_e:
            print(f"Error in mock camera fallback: {str(inner_e)}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/camera/<int:camera_id>/stream/mjpeg', methods=['GET'])
def mjpeg_stream(camera_id):
    """Stream camera as MJPEG"""
    def generate_frames():
        global mock_frame
        
        # Special handling for mock camera
        if camera_id == 999:
            while True:
                try:
                    # Randomly vary the colors slightly to simulate a live feed
                    frame = mock_frame.copy()
                    # Add a timestamp
                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    cv2.putText(frame, current_time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue
                        
                    frame_bytes = buffer.tobytes()
                    
                    # Yield frame in MJPEG format
                    yield b'--frame\r\n'
                    yield b'Content-Type: image/jpeg\r\n\r\n'
                    yield frame_bytes
                    yield b'\r\n'
                    
                    # Limit frame rate
                    time.sleep(0.05)  # ~20 fps
                except Exception as e:
                    print(f"Error generating mock frames: {e}")
                    time.sleep(0.5)  # Sleep longer on error
                    continue
        
        # Normal camera handling
        try:
            if camera_id not in cameras:
                # Open the camera
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print(f"MJPEG: Could not open camera {camera_id}, falling back to mock")
                    # Fallback to mock camera
                    for frame in generate_frames():  # Call the function again with mock camera ID
                        yield frame
                    return
                
                # Apply settings if available
                if "camera" in config and config["camera"]["selectedCamera"] != "":
                    try:
                        cap.set(cv2.CAP_PROP_BRIGHTNESS, config["camera"]["brightness"] / 100)
                        cap.set(cv2.CAP_PROP_CONTRAST, config["camera"]["contrast"] / 100)
                        cap.set(cv2.CAP_PROP_EXPOSURE, config["camera"]["exposure"] / 100)
                    except Exception as e:
                        print(f"Error setting camera properties: {e}")
                        
                cameras[camera_id] = cap
            
            cap = cameras[camera_id]
            
            while True:
                success, frame = cap.read()
                if not success:
                    print(f"MJPEG: Failed to read from camera {camera_id}, trying again")
                    # Try a few more times
                    for retry in range(3):
                        success, frame = cap.read()
                        if success:
                            break
                        time.sleep(0.1)
                    
                    if not success:
                        print(f"MJPEG: Camera {camera_id} failed permanently, falling back to mock")
                        # Switch to mock camera
                        cap.release()
                        del cameras[camera_id]
                        return generate_frames()
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                # Yield frame in MJPEG format
                yield b'--frame\r\n'
                yield b'Content-Type: image/jpeg\r\n\r\n'
                yield frame_bytes
                yield b'\r\n'
                
                # Limit frame rate
                time.sleep(0.05)  # ~20 fps
        except Exception as e:
            print(f"MJPEG Error: {e}, falling back to mock camera")
            # Fallback to mock camera in case of exception
            return generate_frames()
    
    try:
        return Response(generate_frames(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"MJPEG Response error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera/<int:camera_id>/calibrate', methods=['POST'])
def calibrate_camera(camera_id):
    """Perform camera calibration"""
    # Here you would implement the calibration logic
    # This is a placeholder that returns dummy data
    try:
        data = request.get_json()
        
        # In a real implementation, you would:
        # 1. Use the provided calibration parameters
        # 2. Capture frames from the camera
        # 3. Find chessboard corners or other calibration patterns
        # 4. Compute camera matrix, distortion coefficients
        
        # For now, return dummy calibration data
        calibration_result = {
            "camera_id": camera_id,
            "status": "success",
            "camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
            "distortion_coefficients": [0.1, 0.01, 0, 0, 0],
            "timestamp": time.time()
        }
        
        return jsonify(calibration_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def detect_targets():
    """Detect targets in an image"""
    # This is a placeholder for target detection
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        detection_type = data.get('detection_type', 'apriltag')
        
        # Get a frame from the camera
        if camera_id not in cameras:
            # Open the camera
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                return jsonify({"error": f"Could not open camera {camera_id}"}), 400
            cameras[camera_id] = cap
        
        # Get frame
        ret, frame = cameras[camera_id].read()
        if not ret:
            return jsonify({"error": "Failed to grab frame"}), 500
        
        # In a real implementation, you would:
        # 1. Process the frame based on detection_type
        # 2. Detect objects/tags
        # 3. Return information about detected targets
        
        # For now, return dummy detection data
        dummy_results = {
            "camera_id": camera_id,
            "timestamp": time.time(),
            "detection_type": detection_type,
            "targets": [
                {
                    "id": 1,
                    "type": "apriltag",
                    "position": [320, 240],  # x, y in pixels
                    "size": 50,  # size in pixels
                    "rotation": [0, 0, 0],  # rotation in radians
                    "confidence": 0.95
                }
            ]
        }
        
        return jsonify(dummy_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pnp', methods=['POST'])
def estimate_pose():
    """Estimate pose using PnP"""
    # This is a placeholder for pose estimation
    try:
        data = request.get_json()
        
        # In a real implementation, you would:
        # 1. Use camera calibration data
        # 2. Use 3D model of the target
        # 3. Use detected 2D points
        # 4. Solve PnP to get camera pose relative to target
        
        # For now, return dummy pose data
        dummy_pose = {
            "timestamp": time.time(),
            "position": [1.5, 0.3, 3.2],  # x, y, z in meters
            "rotation": [0.1, 0.05, 0.2],  # roll, pitch, yaw in radians
            "confidence": 0.85
        }
        
        return jsonify(dummy_pose)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    return jsonify(config)

@app.route('/api/settings', methods=['POST'])
def save_settings():
    """Save settings"""
    try:
        new_settings = request.get_json()
        # Update config
        global config
        config.update(new_settings)
        
        # Save to file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera/settings', methods=['POST'])
def update_camera_settings():
    """Update camera settings"""
    try:
        data = request.get_json()
        camera_id = data.get('camera')
        settings = data.get('settings', {})
        
        # Update config
        global config
        if "camera" not in config:
            config["camera"] = {}
        
        config["camera"]["selectedCamera"] = camera_id
        config["camera"].update(settings)
        
        # Save to file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Apply settings to camera if it's open
        if camera_id.isdigit() and int(camera_id) in cameras:
            try:
                cap = cameras[int(camera_id)]
                cap.set(cv2.CAP_PROP_BRIGHTNESS, settings.get("brightness", 50) / 100)
                cap.set(cv2.CAP_PROP_CONTRAST, settings.get("contrast", 50) / 100)
                cap.set(cv2.CAP_PROP_EXPOSURE, settings.get("exposure", 50) / 100)
            except Exception as e:
                print(f"Error applying camera settings: {e}")
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "timestamp": time.time()})

def cleanup():
    """Release all camera resources"""
    for camera_id, cap in cameras.items():
        cap.release()

if __name__ == '__main__':
    # Run the Flask app
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        cleanup() 