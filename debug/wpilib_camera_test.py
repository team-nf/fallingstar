#!/usr/bin/env python3
"""
Camera test utility using WPILib/cscore camera functions.
This script demonstrates how to connect to cameras using the 
same methods as WPILibPi for FRC robotics.
"""

import sys
import time
import argparse
import json
import threading
import importlib.util

# Check for required packages
try:
    import cscore
    from cscore import CameraServer, VideoSource, UsbCamera, VideoMode
    has_cscore = True
except ImportError:
    has_cscore = False
    print("Warning: cscore package not found. Install with:")
    print("  pip install robotpy-cscore")

try:
    import numpy as np
    import cv2
    has_opencv = True
except ImportError:
    has_opencv = False
    print("Warning: OpenCV not found. Install with:")
    print("  pip install opencv-python")

def list_cameras():
    """List all available cameras using cscore."""
    if not has_cscore:
        print("Error: cscore package not available")
        return []
    
    print("Scanning for cameras...")
    cameras = []
    
    # Use cscore to enumerate cameras
    for i in range(10):  # Try camera indices 0-9
        try:
            camera = UsbCamera(f"camera{i}", i)
            if camera.isConnected():
                # Get camera info
                info = {
                    "dev_index": i,
                    "name": camera.getName(),
                    "path": camera.getPath()
                }
                
                # Try to get supported modes
                try:
                    modes = camera.enumerateVideoModes()
                    resolutions = []
                    for mode in modes:
                        if mode.pixelFormat == VideoMode.PixelFormat.kMJPEG or mode.pixelFormat == VideoMode.PixelFormat.kYUYV:
                            resolutions.append(f"{mode.width}x{mode.height} @{mode.fps}fps ({mode.pixelFormat})")
                    info["modes"] = resolutions
                except Exception as e:
                    info["modes"] = [f"Error getting modes: {e}"]
                
                cameras.append(info)
                print(f"  Found camera {i}: {info['name']}")
            
            # Always release camera
            camera.close()
            
        except Exception as e:
            # Silently continue if camera can't be opened
            pass
    
    return cameras

def start_capture_thread(camera_server, camera, width, height, fps):
    """Start a thread to process frames for the camera server."""
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    
    # Get the default output stream for the camera
    output_stream = camera_server.getVideo()
    
    # Create a blank output image
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    
    # Start the thread
    def process_frames():
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Get a frame from the camera
            ret, frame = camera.grabFrame(img)
            
            # Check if the frame is valid
            if ret:
                frame_count += 1
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    current_fps = frame_count / elapsed_time
                    
                    # Reset counters
                    frame_count = 0
                    start_time = time.time()
                    
                    # Print status
                    print(f"Streaming at {current_fps:.1f} FPS")
                
                # Put the frame on the output stream
                output_stream.putFrame(frame)
            else:
                # Failed to get a frame
                output_stream.notifyError(camera.getLastStatus())
                print(f"Error getting frame: {camera.getLastError()}")
                
                # Short delay before trying again
                time.sleep(0.1)
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
    
    return processing_thread

def test_camera_stream(camera_index=0, width=320, height=240, fps=30):
    """Test a camera using WPILib CameraServer."""
    if not has_cscore:
        print("Error: cscore package not available")
        return False
    
    try:
        # Initialize camera server (must be done only once)
        camera_server = CameraServer.getInstance()
        camera_server.enableLogging()
        
        # Create a camera
        print(f"Starting camera {camera_index} at {width}x{height} @{fps}fps...")
        camera = camera_server.startAutomaticCapture(dev=camera_index)
        
        # Configure camera
        camera.setResolution(width, height)
        camera.setFPS(fps)
        
        # Try to get current video mode
        try:
            mode = camera.getVideoMode()
            print(f"Camera configured with:")
            print(f"  Resolution: {mode.width}x{mode.height}")
            print(f"  FPS: {mode.fps}")
            print(f"  Pixel Format: {mode.pixelFormat}")
        except Exception as e:
            print(f"Could not get video mode: {e}")
        
        # Start a processing thread if OpenCV is available
        if has_opencv:
            print("Starting CV processing thread...")
            start_capture_thread(camera_server, camera, width, height, fps)
        
        # Display stream info
        server_address = "http://localhost:1181"
        mjpg_address = f"{server_address}/?action=stream"
        
        print("\nCamera stream is now available at:")
        print(f"  {mjpg_address}")
        print("\nView the stream with:")
        print("  - Any web browser")
        print("  - FRC Dashboard")
        print("  - Or using MJPG Streamer viewer\n")
        
        print("Press Ctrl+C to stop streaming.")
        
        # Keep the script alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping camera stream...")
            camera.close()
            print("Stream stopped.")
        
        return True
        
    except Exception as e:
        print(f"Error during camera streaming: {e}")
        return False

def test_all_camera_functions(camera_index=0, width=320, height=240, fps=30):
    """Run a complete test of WPILib camera functions."""
    if not has_cscore:
        print("Error: cscore package not available")
        return False
    
    try:
        # Initialize camera server
        camera_server = CameraServer.getInstance()
        camera_server.enableLogging()
        
        # Create a USB camera
        print(f"Testing camera {camera_index}...")
        camera = UsbCamera("test_camera", camera_index)
        
        # Connect to camera
        if not camera.isConnected():
            print(f"Error: Camera {camera_index} is not connected")
            return False
        
        # Get camera information
        print("\nCamera Information:")
        print(f"  Name: {camera.getName()}")
        print(f"  Path: {camera.getPath()}")
        
        # Configure camera
        camera.setResolution(width, height)
        camera.setFPS(fps)
        
        # Enumerate supported formats and modes
        print("\nSupported Video Modes:")
        try:
            modes = camera.enumerateVideoModes()
            for i, mode in enumerate(modes):
                print(f"  {i}: {mode.width}x{mode.height} @ {mode.fps}fps (PixelFormat: {mode.pixelFormat})")
        except Exception as e:
            print(f"  Error listing video modes: {e}")
        
        # Get current settings
        try:
            cur_mode = camera.getVideoMode()
            print("\nCurrent Video Mode:")
            print(f"  Resolution: {cur_mode.width}x{cur_mode.height}")
            print(f"  FPS: {cur_mode.fps}")
            print(f"  Pixel Format: {cur_mode.pixelFormat}")
        except Exception as e:
            print(f"  Error getting current video mode: {e}")
        
        # Test camera properties
        print("\nTesting Camera Properties:")
        properties = [
            ("brightness", VideoSource.kBrightness, 0, 100),
            ("contrast", VideoSource.kContrast, 0, 100),
            ("saturation", VideoSource.kSaturation, 0, 100),
            ("exposure", VideoSource.kExposureAuto, 0, 1),
            ("white balance", VideoSource.kWhiteBalanceAuto, 0, 1)
        ]
        
        for name, prop_id, min_val, max_val in properties:
            try:
                value = camera.getProperty(prop_id)
                print(f"  {name}: {value.value} (range: {min_val}-{max_val})")
            except Exception as e:
                print(f"  {name}: Error reading property ({e})")
        
        # Get video source to the camera server
        camera_server.addCamera(camera)
        
        # Create output stream
        mjpeg_server = camera_server.addServer(name="serve_" + camera.getName())
        mjpeg_server.setSource(camera)
        
        # Display stream info
        server_address = f"http://localhost:{mjpeg_server.getPort()}"
        mjpg_address = f"{server_address}/?action=stream"
        
        print("\nCamera stream is now available at:")
        print(f"  MJPEG Stream: {mjpg_address}")
        
        # Keep the script alive
        print("\nPress Ctrl+C to stop streaming.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping camera stream...")
        
        # Clean up
        camera.close()
        print("Camera test completed.")
        return True
        
    except Exception as e:
        print(f"Error during camera testing: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test camera with WPILib/cscore")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=320, help="Camera width")
    parser.add_argument("--height", type=int, default=240, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--list", action="store_true", help="List available cameras")
    parser.add_argument("--stream", action="store_true", help="Start a simple camera stream")
    parser.add_argument("--full-test", action="store_true", help="Run full camera function test")
    
    args = parser.parse_args()
    
    if not has_cscore:
        print("Error: This script requires the 'robotpy-cscore' package.")
        print("Install it with: pip install robotpy-cscore")
        return
    
    if args.list:
        print("Listing available cameras...")
        cameras = list_cameras()
        if not cameras:
            print("No cameras found")
        return
    
    if args.stream:
        test_camera_stream(args.camera_id, args.width, args.height, args.fps)
        return
    
    if args.full_test:
        test_all_camera_functions(args.camera_id, args.width, args.height, args.fps)
        return
    
    # Default action if no specific command is given
    print("No action specified. Use --list, --stream, or --full-test.")
    print("For help, use --help")

if __name__ == "__main__":
    main() 