#!/usr/bin/env python3
"""
Main Vision Processing System

This script integrates detection, tracking, selection and NetworkTables
communication for a complete vision processing pipeline.
"""

import os
import cv2
import time
import argparse
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Local imports
try:
    # Try importing from detection module
    from detection.detector import ObjectDetector
except ImportError:
    # Fallback if not available
    from detection import ObjectDetector

# Import tracking
try:
    from tracking.tracker_base import TrackedObject, BoundingBox
    from tracking.sort_tracker import SortTracker
    from tracking.kalman_tracker import KalmanTracker
    from tracking.iou_tracker import IoUTracker
    from tracking.opencv_tracker import OpenCVTracker
    AVAILABLE_TRACKERS = {
        "sort": SortTracker,
        "kalman": KalmanTracker,
        "iou": IoUTracker,
        "opencv": OpenCVTracker
    }
except ImportError:
    # Simple fallback if tracking module is not properly imported
    from tracking import SimpleTracker
    AVAILABLE_TRACKERS = {"simple": SimpleTracker}

# Import utilities
from util.networktables import NetworkTablesInterface, get_default_configuration
from util.selection import select_target, target_info_to_dict, filter_objects

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisionSystem")

# Default configuration
DEFAULT_CONFIG = {
    "camera": {
        "device": 0,           # Camera device ID or path to video file
        "width": 640,          # Camera width
        "height": 480,         # Camera height
        "fps": 30              # Camera FPS
    },
    "processing": {
        "display": True,       # Whether to display processed frames
        "display_detections": True,  # Whether to display detection boxes
        "display_tracking": True,   # Whether to display tracking info
        "display_selection": True,  # Whether to display target selection
        "max_fps": 30,         # Maximum processing FPS (0 = unlimited)
    },
    "detection": {
        "model_path": "model.tflite",  # Path to TFLite model
        "labels_path": "labels.txt",    # Path to labels file
        "threshold": 0.5,      # Detection threshold
        "use_coral": True,     # Whether to use Coral EdgeTPU
        "input_size": [300, 300]  # Model input size [width, height]
    },
    "tracking": {
        "algorithm": "kalman", # Tracking algorithm to use
        "max_age": 30,         # Maximum frames to keep lost tracks
        "min_hits": 3,         # Minimum hits to start tracking
        "iou_threshold": 0.3   # IoU threshold for tracking
    },
    "selection": {
        "algorithm": "lowest", # Target selection algorithm
        "min_confidence": 0.5, # Minimum confidence for target selection
        "max_lost_count": 3,   # Maximum lost count for target selection
        "min_size": 20,        # Minimum object size in pixels
        "max_size": 1000,      # Maximum object size in pixels
        "class_filter": ""     # Comma-separated list of classes to track (empty = all)
    },
    "networktables": {
        "team_number": 0,      # FRC team number (0 = disabled)
        "server_ip": "",       # NetworkTables server IP (empty = use team number)
        "table_name": "VisionTracking"  # NetworkTables table name
    }
}

def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """
    Load configuration from file, falling back to defaults for missing values.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Update config with user values
            for section in user_config:
                if section in config:
                    # Update existing section
                    config[section].update(user_config[section])
                else:
                    # Add new section
                    config[section] = user_config[section]
                    
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    else:
        logger.warning(f"Configuration file {config_path} not found, using defaults")
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write default config to file
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Created default configuration file at {config_path}")
        except Exception as e:
            logger.error(f"Error creating default configuration file: {e}")
    
    return config

def setup_camera(config: Dict[str, Any]) -> Any:
    """
    Set up camera or video source.
    
    Args:
        config: Camera configuration
        
    Returns:
        Camera object (either cv2.VideoCapture or CameraServer)
    """
    camera_config = config["camera"]
    use_camera_server = camera_config.get("use_camera_server", False)
    
    if use_camera_server:
        # Try to import CameraServer for FRC
        try:
            from cscore import CameraServer
            logger.info("Using CameraServer for camera input")
            
            # Initialize camera server
            cs = CameraServer.getInstance()
            cs.enableLogging()
            
            # Determine camera source
            camera_name = camera_config.get("name", "USB Camera")
            source = camera_config["device"]
            
            # Handle different source types
            if isinstance(source, str) and source.startswith("/dev/video"):
                # Linux video device
                camera = cs.startAutomaticCapture(name=camera_name, path=source)
            elif isinstance(source, str) and os.path.exists(source):
                # Video file (note: CameraServer doesn't directly support files)
                logger.warning("CameraServer doesn't directly support video files. Using OpenCV instead.")
                return cv2.VideoCapture(source)
            else:
                # Camera by index
                camera = cs.startAutomaticCapture(name=camera_name, dev=int(source))
            
            # Configure camera properties
            camera.setResolution(
                camera_config["width"],
                camera_config["height"]
            )
            camera.setFPS(camera_config["fps"])
            
            # Get CvSink for capturing frames
            sink = cs.getVideo()
            
            # Create output stream if needed
            if camera_config.get("create_output_stream", True):
                output_stream = cs.putVideo(
                    "Processed", 
                    camera_config["width"],
                    camera_config["height"]
                )
                
                # Store output stream in camera config for later use
                camera_config["output_stream"] = output_stream
            
            # Create frame to hold captured image
            img = np.zeros(shape=(
                camera_config["height"],
                camera_config["width"],
                3
            ), dtype=np.uint8)
            
            # Create a camera object-like interface
            camera_interface = {
                "sink": sink,
                "img": img,
                "read": lambda: _cs_read(sink, img)
            }
            
            logger.info(f"Initialized CameraServer camera: {camera_name}")
            return camera_interface
            
        except ImportError:
            logger.warning("CameraServer not available, falling back to OpenCV")
            # Fall back to OpenCV
            use_camera_server = False
    
    # OpenCV method
    if not use_camera_server:
        logger.info("Using OpenCV for camera input")
        # Open camera or video file
        source = camera_config["device"]
        if isinstance(source, str) and os.path.exists(source):
            logger.info(f"Opening video file: {source}")
            cap = cv2.VideoCapture(source)
        else:
            logger.info(f"Opening camera device: {source}")
            cap = cv2.VideoCapture(source)
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
            cap.set(cv2.CAP_PROP_FPS, camera_config["fps"])
        
        # Check if camera opened successfully
        if not cap.isOpened():
            logger.error("Error opening video source")
            return None
        
        # Read and discard a frame to get actual dimensions
        ret, frame = cap.read()
        if ret:
            logger.info(f"Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
            actual_width = frame.shape[1]
            actual_height = frame.shape[0]
            if actual_width != camera_config["width"] or actual_height != camera_config["height"]:
                logger.warning(f"Actual camera resolution ({actual_width}x{actual_height}) "
                              f"differs from configured ({camera_config['width']}x{camera_config['height']})")
                # Update config with actual dimensions
                camera_config["width"] = actual_width
                camera_config["height"] = actual_height
        
        return cap

def _cs_read(sink, img):
    """
    Helper function to read from CameraServer sink in a way compatible with OpenCV API.
    
    Args:
        sink: CameraServer CvSink
        img: Image array to store the captured frame
        
    Returns:
        Tuple of (success, frame)
    """
    frame_time = sink.grabFrame(img)
    if frame_time == 0:
        # Error getting frame
        error = sink.getError()
        logger.warning(f"Error getting frame: {error}")
        return False, None
    return True, img.copy()

def setup_detector(config: Dict[str, Any], model_path: Optional[str] = None, labels_path: Optional[str] = None) -> Optional[ObjectDetector]:
    """
    Set up object detector.
    
    Args:
        config: Detection configuration
        model_path: Override for model path from command line
        labels_path: Override for labels path from command line
        
    Returns:
        ObjectDetector object or None if setup failed
    """
    detection_config = config["detection"]
    
    # Override paths if provided
    if model_path:
        detection_config["model_path"] = model_path
    if labels_path:
        detection_config["labels_path"] = labels_path
    
    # Log which model and labels files we're using
    logger.info(f"Using model: {detection_config['model_path']}")
    logger.info(f"Using labels: {detection_config['labels_path']}")
    
    try:
        detector = ObjectDetector(
            model_path=detection_config["model_path"],
            labels_path=detection_config["labels_path"],
            threshold=detection_config["threshold"],
            use_coral=detection_config.get("use_coral", True),
            input_size=detection_config.get("input_size", [300, 300])
        )
        logger.info("Object detector initialized successfully")
        return detector
    except Exception as e:
        logger.error(f"Error initializing object detector: {e}")
        return None

def setup_tracker(config: Dict[str, Any]) -> Any:
    """
    Set up object tracker.
    
    Args:
        config: Tracking configuration
        
    Returns:
        Tracker object
    """
    tracking_config = config["tracking"]
    
    algorithm = tracking_config["algorithm"].lower()
    
    if algorithm in AVAILABLE_TRACKERS:
        # Initialize tracker with parameters
        tracker_class = AVAILABLE_TRACKERS[algorithm]
        tracker = tracker_class(
            max_age=tracking_config.get("max_age", 30),
            min_hits=tracking_config.get("min_hits", 3),
            iou_threshold=tracking_config.get("iou_threshold", 0.3)
        )
        logger.info(f"Initialized {algorithm} tracker")
        return tracker
    else:
        logger.warning(f"Tracker algorithm '{algorithm}' not found, using default")
        if "simple" in AVAILABLE_TRACKERS:
            return AVAILABLE_TRACKERS["simple"](
                max_age=tracking_config.get("max_age", 30),
                min_hits=tracking_config.get("min_hits", 3),
                iou_threshold=tracking_config.get("iou_threshold", 0.3)
            )
        # Fallback to first available tracker
        tracker_name = list(AVAILABLE_TRACKERS.keys())[0]
        return AVAILABLE_TRACKERS[tracker_name](
            max_age=tracking_config.get("max_age", 30),
            min_hits=tracking_config.get("min_hits", 3),
            iou_threshold=tracking_config.get("iou_threshold", 0.3)
        )

def setup_networktables(config: Dict[str, Any]) -> Optional[NetworkTablesInterface]:
    """
    Set up NetworkTables connection.
    
    Args:
        config: NetworkTables configuration
        
    Returns:
        NetworkTablesInterface object or None if disabled
    """
    nt_config = config["networktables"]
    
    team_number = nt_config.get("team_number", 0)
    server_ip = nt_config.get("server_ip", "")
    table_name = nt_config.get("table_name", "VisionTracking")
    
    if team_number <= 0 and not server_ip:
        logger.info("NetworkTables disabled (no team number or server IP)")
        return None
    
    try:
        if server_ip:
            # Connect to specific IP
            nt = NetworkTablesInterface(
                server_address=server_ip,
                table_name=table_name
            )
            logger.info(f"Initialized NetworkTables with server {server_ip}")
        else:
            # Connect using team number
            nt = NetworkTablesInterface(
                team_number=team_number,
                table_name=table_name
            )
            logger.info(f"Initialized NetworkTables for team {team_number}")
        
        # Set default configuration
        default_config = get_default_configuration()
        nt.set_default_configuration(default_config)
        
        return nt
    except Exception as e:
        logger.error(f"Error initializing NetworkTables: {e}")
        return None

def draw_detections(frame: np.ndarray, objects: List[TrackedObject], 
                   selected: Optional[TrackedObject] = None) -> np.ndarray:
    """
    Draw detection and tracking information on frame.
    
    Args:
        frame: Input frame
        objects: List of tracked objects
        selected: Selected target object (optional)
        
    Returns:
        Frame with annotations
    """
    # Make a copy of the frame
    output = frame.copy()
    
    # Draw all tracked objects
    for obj in objects:
        # Get bounding box coordinates
        x1, y1 = int(obj.bbox.xmin), int(obj.bbox.ymin)
        x2, y2 = int(obj.bbox.xmax), int(obj.bbox.ymax)
        
        # Draw box - green for tracked, blue for selected
        color = (0, 255, 0)  # Green by default
        thickness = 2
        
        if selected and obj.id == selected.id:
            color = (255, 0, 0)  # Blue for selected target
            thickness = 3
            
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Draw ID and class
        label = f"ID:{obj.id}"
        if obj.class_name:
            label += f" {obj.class_name}"
        if obj.score:
            label += f" {obj.score:.2f}"
            
        # Position label above the bounding box
        cv2.putText(output, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # If object has velocity, draw a line showing direction
        if hasattr(obj, 'velocity') and obj.velocity:
            vx, vy = obj.velocity
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Scale velocity for visualization
            scale = 10
            end_x = int(center_x + vx * scale)
            end_y = int(center_y + vy * scale)
            
            cv2.line(output, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
    
    # Draw information about selected target
    if selected:
        # Draw target info at the top left corner
        info_text = [
            f"Target: ID {selected.id}",
            f"Position: ({selected.bbox.center[0]}, {selected.bbox.center[1]})"
        ]
        
        if hasattr(selected, 'velocity') and selected.velocity:
            vx, vy = selected.velocity
            speed = np.sqrt(vx**2 + vy**2)
            info_text.append(f"Speed: {speed:.2f} px/frame")
        
        for i, text in enumerate(info_text):
            cv2.putText(output, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return output

def load_camera_calibration(calibration_file: str) -> Optional[Dict[str, Any]]:
    """
    Kamera kalibrasyon dosyasını yükler.
    
    Args:
        calibration_file: Kalibrasyon dosyasının yolu
        
    Returns:
        Kalibrasyon parametrelerini içeren sözlük veya dosya bulunamazsa None
    """
    if not os.path.exists(calibration_file):
        logger.warning(f"Kalibrasyon dosyası bulunamadı: {calibration_file}")
        return None
    
    try:
        # Dosya formatına göre okuma
        file_ext = os.path.splitext(calibration_file)[1].lower()
        
        if file_ext == '.json':
            # JSON formatı
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)
                
            # JSON'dan numpy array'a dönüştürme
            camera_matrix = np.array(calibration_data.get('camera_matrix', []), dtype=np.float32)
            dist_coeffs = np.array(calibration_data.get('dist_coeffs', []), dtype=np.float32)
            
        elif file_ext in ['.xml', '.yml', '.yaml']:
            # OpenCV dosya formatı
            fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode('camera_matrix').mat()
            dist_coeffs = fs.getNode('dist_coeffs').mat()
            fs.release()
            
        else:
            logger.error(f"Desteklenmeyen kalibrasyon dosyası formatı: {file_ext}")
            return None
        
        if camera_matrix.size == 0 or dist_coeffs.size == 0:
            logger.error("Kalibrasyon verileri boş veya geçersiz")
            return None
            
        logger.info(f"Kamera kalibrasyon dosyası başarıyla yüklendi: {calibration_file}")
        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs
        }
        
    except Exception as e:
        logger.error(f"Kalibrasyon dosyası yüklenirken hata oluştu: {e}")
        return None

def undistort_frame(frame: np.ndarray, calibration: Dict[str, Any]) -> np.ndarray:
    """
    Kamera kalibrasyon parametrelerini kullanarak görüntüyü düzeltir.
    
    Args:
        frame: Düzeltilecek görüntü
        calibration: Kalibrasyon parametreleri
        
    Returns:
        Düzeltilmiş görüntü
    """
    if frame is None or calibration is None:
        return frame
    
    try:
        camera_matrix = calibration.get('camera_matrix')
        dist_coeffs = calibration.get('dist_coeffs')
        
        if camera_matrix is None or dist_coeffs is None:
            return frame
            
        # Görüntüyü düzelt
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Görüntüyü düzelt
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # ROI'yi kes (isteğe bağlı)
        # x, y, w, h = roi
        # undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
        
    except Exception as e:
        logger.error(f"Görüntü düzeltilirken hata oluştu: {e}")
        return frame

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vision Processing System")
    parser.add_argument("--config", default="config/config.json", help="Path to configuration file")
    parser.add_argument("--model", help="Path to TFLite model file (overrides config)")
    parser.add_argument("--labels", help="Path to labels file (overrides config)")
    parser.add_argument("--camera", type=int, help="Camera device ID (overrides config)")
    parser.add_argument("--video", help="Path to video file (overrides config)")
    parser.add_argument("--team", type=int, help="FRC team number (overrides config)")
    parser.add_argument("--server", help="NetworkTables server IP (overrides config)")
    parser.add_argument("--use-cs", action="store_true", help="Use CameraServer instead of OpenCV")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.camera is not None:
        config["camera"]["device"] = args.camera
    if args.video:
        config["camera"]["device"] = args.video
    if args.team:
        config["networktables"]["team_number"] = args.team
    if args.server:
        config["networktables"]["server_ip"] = args.server
    if args.no_display:
        config["processing"]["display"] = False
    if args.use_cs:
        config["camera"]["use_camera_server"] = True
    
    # Kamera kalibrasyon verilerini yükle
    camera_calibration = None
    if config["camera"].get("calibration", {}).get("use_calibration", False):
        calibration_file = config["camera"]["calibration"].get("calibration_file", "")
        if calibration_file:
            camera_calibration = load_camera_calibration(calibration_file)
    
    # Setup components
    cap = setup_camera(config)
    if not cap:
        logger.error("Failed to set up camera, exiting")
        return
    
    detector = setup_detector(config, model_path=args.model, labels_path=args.labels)
    if not detector:
        logger.error("Failed to set up detector, exiting")
        return
    
    tracker = setup_tracker(config)
    if not tracker:
        logger.error("Failed to set up tracker, exiting")
        return
    
    nt = setup_networktables(config)
    if nt:
        logger.info("NetworkTables connection established")
        # Wait for connection
        if nt.wait_for_connection(timeout=5.0):
            logger.info("Connected to NetworkTables server")
        else:
            logger.warning("NetworkTables connection timed out")
    
    # Get frame dimensions
    frame_width = config["camera"]["width"]
    frame_height = config["camera"]["height"]
    
    # Check if we're using CameraServer
    using_camera_server = isinstance(cap, dict) and "sink" in cap
    
    # Main processing loop
    logger.info("Starting main processing loop")
    frame_count = 0
    start_time = time.time()
    last_fps_print = start_time
    
    while True:
        loop_start = time.time()
        
        # Read frame from camera
        if using_camera_server:
            # Using CameraServer
            ret, frame = cap["read"]()
        else:
            # Using OpenCV
            ret, frame = cap.read()
            
        if not ret or frame is None:
            logger.warning("Failed to read frame, retrying...")
            time.sleep(0.1)
            continue
            
        # Kamera kalibrasyon verileri varsa görüntüyü düzelt
        if camera_calibration is not None and config["camera"]["calibration"].get("undistort_frames", True):
            frame = undistort_frame(frame, camera_calibration)
        
        # Run object detection
        detections = detector.detect(frame)
        
        # Update tracker with new detections
        tracked_objects = tracker.update(detections, frame)
        
        # Get selection parameters
        selection_params = {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "min_confidence": config["selection"].get("min_confidence", 0.5),
            "max_lost_count": config["selection"].get("max_lost_count", 3),
            "min_size": config["selection"].get("min_size", 20),
            "max_size": config["selection"].get("max_size", 1000),
            "class_filter": config["selection"].get("class_filter", "")
        }
        
        # Select target
        selection_method = config["selection"].get("algorithm", "lowest")
        selected_target = select_target(tracked_objects, selection_method, selection_params)
        
        # Prepare output for display
        if config["processing"].get("display", True):
            output_frame = draw_detections(frame, tracked_objects, selected_target)
            
            # Add processing info
            frame_time = time.time() - loop_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display frame locally
            cv2.imshow("Vision Processing", output_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed, exiting...")
                break
            
            # If using CameraServer, also send processed frame to output stream
            if using_camera_server and "output_stream" in config["camera"]:
                try:
                    config["camera"]["output_stream"].putFrame(output_frame)
                except Exception as e:
                    logger.error(f"Error sending frame to output stream: {e}")
        
        # Publish data to NetworkTables
        if nt:
            # Convert selected target to dictionary
            if selected_target:
                target_info = target_info_to_dict(selected_target, frame_width, frame_height)
                nt.publish_target_info(target_info)
            
            # Convert all tracked objects to list of dictionaries for publishing
            tracked_objects_list = []
            for obj in tracked_objects:
                obj_dict = {
                    "id": obj.id,
                    "class_id": obj.class_id,
                    "class_name": obj.class_name,
                    "score": obj.score,
                    "bbox": {
                        "xmin": obj.bbox.xmin,
                        "ymin": obj.bbox.ymin,
                        "xmax": obj.bbox.xmax,
                        "ymax": obj.bbox.ymax
                    }
                }
                if hasattr(obj, 'velocity') and obj.velocity:
                    obj_dict["velocity"] = list(obj.velocity)
                tracked_objects_list.append(obj_dict)
            
            nt.publish_tracked_objects(tracked_objects_list)
        
        # Update frame count
        frame_count += 1
        
        # Print FPS every 5 seconds
        if time.time() - last_fps_print > 5:
            elapsed = time.time() - last_fps_print
            fps = frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"Processing at {fps:.1f} FPS ({frame_count} frames in {elapsed:.1f}s)")
            last_fps_print = time.time()
            frame_count = 0
        
        # Limit FPS if needed
        max_fps = config["processing"].get("max_fps", 30)
        if max_fps > 0:
            frame_time = time.time() - loop_start
            sleep_time = 1.0 / max_fps - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # Clean up
    if not using_camera_server:
        cap.release()
    cv2.destroyAllWindows()
    logger.info("Vision processing ended")

if __name__ == "__main__":
    main() 