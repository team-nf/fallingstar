#!/usr/bin/env python3

import numpy as np
import cv2
import glob
import os
import json
import argparse
from datetime import datetime

def calibrate_camera(images_folder, checkerboard_size=(9, 6), square_size=25.0, save_dir='./'):
    """
    Calibrate camera using checkerboard images.
    
    Args:
        images_folder: Path to folder containing checkerboard images
        checkerboard_size: Tuple of (width, height) interior corners of checkerboard
        square_size: Size of checkerboard square in mm
        save_dir: Directory to save calibration results
    
    Returns:
        camera_matrix, dist_coeffs: Camera calibration parameters
    """
    # Prepare object points (3D points in real world space)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of images
    images = glob.glob(os.path.join(images_folder, '*.jpg')) + \
             glob.glob(os.path.join(images_folder, '*.png'))
    
    if not images:
        print(f"No images found in {images_folder}")
        return None, None
    
    # Read first image to get image size
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    
    # Process each image
    successful_images = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        
        # If corners found, refine and add to calibration data
        if ret:
            print(f"Found corners in {os.path.basename(fname)}")
            successful_images.append(fname)
            
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw and display the corners
            img_corners = img.copy()
            cv2.drawChessboardCorners(img_corners, checkerboard_size, corners2, ret)
            
            # Save image with corners drawn
            corners_dir = os.path.join(save_dir, 'corners_detected')
            os.makedirs(corners_dir, exist_ok=True)
            base_name = os.path.basename(fname)
            cv2.imwrite(os.path.join(corners_dir, f"corners_{base_name}"), img_corners)
            
            # Add to calibration data
            objpoints.append(objp)
            imgpoints.append(corners2)
    
    # Calibrate camera if we have enough images
    if len(objpoints) < 5:
        print(f"Not enough images with detected corners (found {len(objpoints)}, need at least 5)")
        return None, None
    
    print(f"Calibrating camera using {len(objpoints)} images...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    print(f"Calibration complete!")
    print(f"Mean reprojection error: {mean_error}")
    
    # Save calibration results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cal_results = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_size": img_size,
        "checkerboard_size": checkerboard_size,
        "square_size_mm": square_size,
        "reprojection_error": float(mean_error),
        "num_images_used": len(objpoints),
        "calibration_date": timestamp
    }
    
    # Save as JSON
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"camera_calibration_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(cal_results, f, indent=4)
    
    # Also save in OpenCV format (XML)
    xml_path = os.path.join(save_dir, f"camera_calibration_{timestamp}.xml")
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("image_size", np.array(img_size))
    fs.release()
    
    print(f"Calibration saved to {json_path} and {xml_path}")
    
    # Undistort and save a sample image
    if successful_images:
        img = cv2.imread(successful_images[0])
        h, w = img.shape[:2]
        
        # Get optimal new camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        
        # Undistort
        dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
        
        # Crop the image (optional)
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        
        # Save undistorted image
        undistort_dir = os.path.join(save_dir, 'undistorted')
        os.makedirs(undistort_dir, exist_ok=True)
        base_name = os.path.basename(successful_images[0])
        undist_path = os.path.join(undistort_dir, f"undistorted_{base_name}")
        cv2.imwrite(undist_path, dst)
        print(f"Saved undistorted sample image to {undist_path}")
    
    return camera_matrix, dist_coeffs

def capture_calibration_images(output_folder, num_images=20, delay=2):
    """
    Capture images from camera for calibration.
    
    Args:
        output_folder: Folder to save captured images
        num_images: Number of images to capture
        delay: Delay between captures in seconds
    """
    import time
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera opened successfully.")
    print(f"Will capture {num_images} images with {delay} seconds between captures.")
    print("Position the checkerboard in different orientations.")
    print("Press 'c' to capture an image manually, or wait for automatic capture.")
    print("Press 'q' to quit.")
    
    img_count = 0
    last_capture_time = time.time() - delay  # Allow immediate first capture
    
    while img_count < num_images:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Show the frame
        display_frame = frame.copy()
        cv2.putText(
            display_frame, 
            f"Captured: {img_count}/{num_images} | Press 'c' to capture, 'q' to quit", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.imshow('Calibration Capture', display_frame)
        
        # Capture logic
        current_time = time.time()
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        # Capture image on 'c' key press or after delay
        if key == ord('c') or (current_time - last_capture_time >= delay):
            img_path = os.path.join(output_folder, f"calib_img_{img_count:03d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Captured image {img_count+1}/{num_images}: {img_path}")
            img_count += 1
            last_capture_time = current_time
    
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_count} images.")

def main():
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture calibration images')
    capture_parser.add_argument('--output', default='./calibration_images', 
                              help='Output folder for calibration images')
    capture_parser.add_argument('--num-images', type=int, default=20, 
                              help='Number of images to capture')
    capture_parser.add_argument('--delay', type=int, default=2, 
                              help='Delay between automatic captures (seconds)')
    
    # Calibrate command
    calib_parser = subparsers.add_parser('calibrate', help='Calibrate camera using images')
    calib_parser.add_argument('--images', required=True, 
                            help='Folder containing calibration images')
    calib_parser.add_argument('--output', default='./calibration_results', 
                            help='Output folder for calibration results')
    calib_parser.add_argument('--width', type=int, default=9, 
                            help='Number of inner corners along width of checkerboard')
    calib_parser.add_argument('--height', type=int, default=6, 
                            help='Number of inner corners along height of checkerboard')
    calib_parser.add_argument('--square-size', type=float, default=25.0, 
                            help='Size of checkerboard square in millimeters')
    
    args = parser.parse_args()
    
    if args.command == 'capture':
        capture_calibration_images(args.output, args.num_images, args.delay)
    elif args.command == 'calibrate':
        calibrate_camera(
            args.images, 
            checkerboard_size=(args.width, args.height),
            square_size=args.square_size,
            save_dir=args.output
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 