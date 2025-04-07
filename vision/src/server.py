#!/usr/bin/env python3
import os
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

# Import our vision modules
from get_cameras import get_available_cameras

app = Flask(__name__)
CORS(app)

# Global variables
cameras = {}  # Store open camera connections

@app.route('/api/cameras', methods=['GET'])
def list_cameras():
    """List all available camera devices"""
    try:
        available_cameras = get_available_cameras()
        return jsonify(available_cameras)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera/<int:camera_id>/stream', methods=['GET'])
def camera_stream(camera_id):
    """Get a single frame from the specified camera"""
    try:
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
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({"error": "Failed to encode frame"}), 500
        
        # Convert to base64 for JSON response
        import base64
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Return the frame data
        return jsonify({
            "camera_id": camera_id,
            "timestamp": time.time(),
            "frame": frame_data
        })
    except Exception as e:
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