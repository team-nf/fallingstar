#!/usr/bin/env python3
"""
PnP (Perspective-n-Point) Estimation for Circular Objects

This module provides functionality to estimate 3D position of circular objects (like algae)
using computer vision techniques and PnP algorithm.
"""

import cv2
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple, Union


class CircularObjectPnP:
    """Class for estimating 3D position of circular objects using PnP algorithm."""
    
    def __init__(self, 
                focal_length: Optional[float] = None,
                camera_matrix: Optional[np.ndarray] = None,
                dist_coeffs: Optional[np.ndarray] = None,
                real_diameter_cm: float = 39.0,
                min_confidence: float = 0.7,
                min_circularity: float = 0.8,
                samples_on_circle: int = 12):
        """
        Initialize the CircularObjectPnP estimator.
        
        Args:
            focal_length: Focal length of the camera (in pixels) if camera_matrix is not provided
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            real_diameter_cm: Actual diameter of the circular object in centimeters
            min_confidence: Minimum confidence score to consider a detection valid
            min_circularity: Minimum circularity ratio (width/height) to consider the object as circular
            samples_on_circle: Number of sample points to generate on the circle perimeter
        """
        self.real_diameter_cm = real_diameter_cm
        self.min_confidence = min_confidence
        self.min_circularity = min_circularity
        self.samples_on_circle = samples_on_circle
        
        # Set up camera parameters
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        elif focal_length is not None:
            # Create camera matrix with provided focal length
            self.camera_matrix = np.array([
                [focal_length, 0, 0],
                [0, focal_length, 0],
                [0, 0, 1]
            ])
        else:
            # Default camera matrix (will be updated later)
            self.camera_matrix = np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ])
            
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1))
        
    def update_camera_params(self, camera_matrix: np.ndarray, dist_coeffs: Optional[np.ndarray] = None):
        """
        Update camera parameters.
        
        Args:
            camera_matrix: New camera intrinsic matrix
            dist_coeffs: New distortion coefficients
        """
        self.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs
            
    def is_circular(self, bbox: Dict[str, float]) -> bool:
        """
        Check if a bounding box represents a circular object.
        
        Args:
            bbox: Bounding box with xmin, ymin, xmax, ymax
            
        Returns:
            True if the bounding box likely contains a circular object
        """
        width = bbox['xmax'] - bbox['xmin']
        height = bbox['ymax'] - bbox['ymin']
        
        # Avoid division by zero
        if width == 0 or height == 0:
            return False
            
        # Calculate aspect ratio
        aspect_ratio = min(width, height) / max(width, height)
        
        # If aspect ratio is close to 1, it's approximately circular
        return aspect_ratio >= self.min_circularity
    
    def find_circle_in_roi(self, image: np.ndarray, bbox: Dict[str, float]) -> Optional[Tuple[Tuple[int, int], int]]:
        """
        Find the largest circle within the bounding box region.
        
        Args:
            image: Input image
            bbox: Bounding box containing the circular object
            
        Returns:
            Tuple of (center_x, center_y), radius or None if circle not found
        """
        # Extract ROI from the image
        xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
        xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width - 1, xmax)
        ymax = min(height - 1, ymax)
        
        if xmax <= xmin or ymax <= ymin:
            return None
            
        roi = image[ymin:ymax, xmin:xmax]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        # Try to find circles using Hough Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(roi.shape),  # Only find the largest circle
            param1=50,
            param2=30,
            minRadius=min(roi.shape) // 4,  # Must be at least 1/4 of the ROI
            maxRadius=min(roi.shape) // 2   # Cannot be larger than 1/2 of the ROI
        )
        
        if circles is not None:
            # Get the largest circle
            best_circle = sorted(circles[0], key=lambda x: x[2], reverse=True)[0]
            
            # Convert coordinates relative to ROI back to full image coordinates
            center_x = int(best_circle[0]) + xmin
            center_y = int(best_circle[1]) + ymin
            radius = int(best_circle[2])
            
            return ((center_x, center_y), radius)
            
        # Alternative approach: assume the bounding box is a good approximation
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        radius = min(xmax - xmin, ymax - ymin) // 2
        
        return ((center_x, center_y), radius)
    
    def generate_circle_points(self, center: Tuple[int, int], radius: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D-2D point correspondences for a circle.
        
        Args:
            center: Center coordinates of the circle in the image (x, y)
            radius: Radius of the circle in pixels
            
        Returns:
            Tuple of (object_points_3d, image_points_2d)
        """
        # Generate points on the perimeter of the circle in 2D image space
        angle_step = 2 * math.pi / self.samples_on_circle
        image_points = []
        object_points = []
        
        # Add center point
        image_points.append([center[0], center[1]])
        object_points.append([0, 0, 0])
        
        # Generate points on the circle perimeter
        real_radius = self.real_diameter_cm / 2.0
        
        for i in range(self.samples_on_circle):
            angle = i * angle_step
            
            # 2D image points
            x = center[0] + int(radius * math.cos(angle))
            y = center[1] + int(radius * math.sin(angle))
            image_points.append([x, y])
            
            # Corresponding 3D object points (using a circle on the XY plane)
            x_3d = real_radius * math.cos(angle)
            y_3d = real_radius * math.sin(angle)
            z_3d = 0  # Circle on the XY plane (Z=0)
            object_points.append([x_3d, y_3d, z_3d])
        
        return np.array(object_points, dtype=np.float32), np.array(image_points, dtype=np.float32)
    
    def estimate_pose(self, image: np.ndarray, detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Estimate 3D pose of a circular object from its 2D detection.
        
        Args:
            image: Input image
            detection: Detection dictionary with bbox and confidence
            
        Returns:
            Dictionary with pose estimation results or None if estimation fails
        """
        # Check confidence threshold
        if detection.get('score', 0) < self.min_confidence:
            return None
            
        # Extract bounding box
        bbox_dict = detection.get('bbox', {})
        if not isinstance(bbox_dict, dict):
            # Handle case where bbox might be a BoundingBox object
            try:
                bbox_dict = {
                    'xmin': bbox_dict.xmin,
                    'ymin': bbox_dict.ymin,
                    'xmax': bbox_dict.xmax,
                    'ymax': bbox_dict.ymax
                }
            except:
                return None
        
        # Check if the object is roughly circular
        if not self.is_circular(bbox_dict):
            return None
            
        # Find circle parameters in the image
        circle_info = self.find_circle_in_roi(image, bbox_dict)
        if circle_info is None:
            return None
            
        # Generate point correspondences
        center, radius = circle_info
        object_points, image_points = self.generate_circle_points(center, radius)
        
        # Solve PnP to get object pose
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None
                
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate distance from camera (in cm)
            distance = np.linalg.norm(translation_vector)
            
            # Get angle of approach
            # Forward direction in camera coordinates is Z axis
            z_axis = np.array([0, 0, 1])
            direction_vector = translation_vector.flatten() / np.linalg.norm(translation_vector)
            angle = np.arccos(np.dot(z_axis, direction_vector)) * 180 / np.pi
            
            # Get horizontal and vertical angles
            horizontal_angle = math.atan2(translation_vector[0], translation_vector[2]) * 180 / math.pi
            vertical_angle = math.atan2(translation_vector[1], translation_vector[2]) * 180 / math.pi
            
            # Create result dictionary
            result = {
                'distance_cm': float(distance),
                'center': center,
                'radius_px': radius,
                'pose': {
                    'rotation_vector': rotation_vector.flatten().tolist(),
                    'translation_vector': translation_vector.flatten().tolist(),
                    'angles': {
                        'horizontal': float(horizontal_angle),
                        'vertical': float(vertical_angle),
                        'approach': float(angle)
                    }
                },
                'confidence': detection.get('score', 1.0)
            }
            
            return result
            
        except Exception as e:
            print(f"PnP estimation error: {e}")
            return None
    
    def draw_pose(self, image: np.ndarray, pose_result: Dict[str, Any], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw pose estimation result on the image.
        
        Args:
            image: Input image
            pose_result: Pose estimation result
            color: Color for drawing
            
        Returns:
            Image with pose information drawn on it
        """
        if pose_result is None:
            return image
            
        # Make a copy of the image to avoid modifying the original
        result_img = image.copy()
        
        # Draw circle
        center = pose_result['center']
        radius = pose_result['radius_px']
        cv2.circle(result_img, center, radius, color, 2)
        cv2.circle(result_img, center, 5, (0, 0, 255), -1)  # Center point
        
        # Draw distance
        distance = pose_result['distance_cm']
        cv2.putText(
            result_img,
            f"Dist: {distance:.1f} cm",
            (center[0] - radius, center[1] - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        # Draw coordinate axes
        rotation_vector = np.array(pose_result['pose']['rotation_vector'])
        translation_vector = np.array(pose_result['pose']['translation_vector'])
        
        # Calculate scale factor based on radius
        scale_factor = radius / 2
        
        # Draw coordinate system axes
        self.draw_axes(
            result_img,
            self.camera_matrix,
            self.dist_coeffs,
            rotation_vector,
            translation_vector,
            scale_factor
        )
        
        # Draw angles
        angles = pose_result['pose']['angles']
        cv2.putText(
            result_img,
            f"H: {angles['horizontal']:.1f}° V: {angles['vertical']:.1f}°",
            (center[0] - radius, center[1] - radius - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        return result_img
    
    def draw_axes(self, 
                 image: np.ndarray, 
                 camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray, 
                 rvec: np.ndarray, 
                 tvec: np.ndarray, 
                 length: float = 50) -> None:
        """
        Draw 3D coordinate axes on the image.
        
        Args:
            image: Input image
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            rvec: Rotation vector
            tvec: Translation vector
            length: Length of the axes in the 3D coordinate system
        """
        # Define the 3D coordinate system points
        origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
        x_axis = np.float32([[length, 0, 0]]).reshape(-1, 3)
        y_axis = np.float32([[0, length, 0]]).reshape(-1, 3)
        z_axis = np.float32([[0, 0, length]]).reshape(-1, 3)
        
        # Project 3D points to the image plane
        origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
        x_axis_2d, _ = cv2.projectPoints(x_axis, rvec, tvec, camera_matrix, dist_coeffs)
        y_axis_2d, _ = cv2.projectPoints(y_axis, rvec, tvec, camera_matrix, dist_coeffs)
        z_axis_2d, _ = cv2.projectPoints(z_axis, rvec, tvec, camera_matrix, dist_coeffs)
        
        # Convert to integer coordinates
        origin_2d = tuple(map(int, origin_2d.flatten()))
        x_axis_2d = tuple(map(int, x_axis_2d.flatten()))
        y_axis_2d = tuple(map(int, y_axis_2d.flatten()))
        z_axis_2d = tuple(map(int, z_axis_2d.flatten()))
        
        # Draw the axes
        cv2.line(image, origin_2d, x_axis_2d, (0, 0, 255), 2)  # X-axis in red
        cv2.line(image, origin_2d, y_axis_2d, (0, 255, 0), 2)  # Y-axis in green
        cv2.line(image, origin_2d, z_axis_2d, (255, 0, 0), 2)  # Z-axis in blue
        
        # Add axis labels
        cv2.putText(image, "X", x_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, "Y", y_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Z", z_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def load_camera_params_from_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera parameters from a file.
    
    Args:
        file_path: Path to the camera calibration file
        
    Returns:
        Tuple of (camera_matrix, dist_coeffs)
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            # Load JSON format
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            camera_matrix = np.array(data.get('camera_matrix', []), dtype=np.float32)
            dist_coeffs = np.array(data.get('dist_coeffs', []), dtype=np.float32)
            
        elif file_ext in ['.xml', '.yml', '.yaml']:
            # Load OpenCV format
            fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode('camera_matrix').mat()
            dist_coeffs = fs.getNode('dist_coeffs').mat()
            fs.release()
            
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        return camera_matrix, dist_coeffs
        
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        # Return default values
        return np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ]), np.zeros((5, 1))


def process_detections(image: np.ndarray, 
                     detections: List[Dict[str, Any]], 
                     pnp_estimator: CircularObjectPnP,
                     class_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Process a list of detections and estimate the 3D pose for circular objects.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        pnp_estimator: CircularObjectPnP estimator
        class_filter: Only process objects of this class (None for all)
        
    Returns:
        List of detections with added pose information
    """
    results = []
    
    for detection in detections:
        # Filter by class if needed
        if class_filter is not None and detection.get('class_name') != class_filter:
            continue
            
        # Try to estimate pose
        pose_result = pnp_estimator.estimate_pose(image, detection)
        
        if pose_result is not None:
            # Add pose information to the detection
            detection_with_pose = detection.copy()
            detection_with_pose['pose'] = pose_result
            results.append(detection_with_pose)
            
    return results


# Example usage:
if __name__ == "__main__":
    import os
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test PnP estimation on a circular object")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--calibration", type=str, default="config/camera_calibration.json",
                       help="Path to camera calibration file")
    parser.add_argument("--diameter", type=float, default=39.0,
                       help="Actual diameter of the circular object in cm")
    args = parser.parse_args()
    
    # Load camera parameters
    camera_matrix, dist_coeffs = load_camera_params_from_file(args.calibration)
    
    # Create PnP estimator
    pnp_estimator = CircularObjectPnP(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        real_diameter_cm=args.diameter
    )
    
    # Test on image if provided
    if args.image and os.path.exists(args.image):
        # Load image
        image = cv2.imread(args.image)
        
        # For testing, create a fake detection
        h, w = image.shape[:2]
        detection = {
            'bbox': {
                'xmin': w // 4,
                'ymin': h // 4,
                'xmax': 3 * w // 4,
                'ymax': 3 * h // 4
            },
            'score': 0.9,
            'class_name': 'algae'
        }
        
        # Estimate pose
        pose_result = pnp_estimator.estimate_pose(image, detection)
        
        if pose_result is not None:
            # Draw result
            result_image = pnp_estimator.draw_pose(image, pose_result)
            
            # Display result
            cv2.imshow("PnP Estimation", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Print pose information
            print(f"Distance: {pose_result['distance_cm']:.2f} cm")
            print(f"Horizontal angle: {pose_result['pose']['angles']['horizontal']:.2f} degrees")
            print(f"Vertical angle: {pose_result['pose']['angles']['vertical']:.2f} degrees")
        else:
            print("Failed to estimate pose")
    else:
        print("Please provide a valid image path") 