# Camera Calibration for FRC Vision Processing

This directory contains tools for calibrating your global shutter camera for accurate vision processing.

## Why Camera Calibration is Important

Camera calibration corrects for:
- Lens distortion (barrel, pincushion, tangential)
- Inaccurate measurements due to perspective and distortion
- Inconsistent detection of game objects

For FRC vision tasks, proper calibration ensures:
1. More accurate distance calculations
2. Better object detection, especially at the edges of the frame
3. More reliable pose estimation with AprilTags

## Requirements

- A printed checkerboard pattern (you can print one from [here](https://docs.opencv.org/3.4/pattern.png))
- Good, consistent lighting
- Your global shutter camera mounted in its final position on the robot

## Calibration Process

### 1. Print the Checkerboard Pattern

Print a checkerboard pattern (typically 9x6 inner corners) on a rigid board. Measure the square size in millimeters accurately.

### 2. Capture Calibration Images

Use the provided script to capture images of the checkerboard in different orientations:

```bash
python3 camera_calibration.py capture --output ./calibration_images --num-images 20
```

Guidelines for good calibration images:
- Capture the checkerboard in various orientations and positions
- Include images where the checkerboard is at the edges of the frame
- Make sure the entire checkerboard is visible in each image
- Avoid motion blur by holding the checkerboard steady
- Maintain consistent lighting

### 3. Run the Calibration

Process the captured images to calculate the camera calibration parameters:

```bash
python3 camera_calibration.py calibrate --images ./calibration_images --output ./calibration_results --width 9 --height 6 --square-size 25.0
```

Replace:
- `--width` and `--height` with the number of inner corners of your checkerboard
- `--square-size` with the actual size of each checkerboard square in millimeters

### 4. Review the Results

After calibration, check:
1. The reprojection error (lower is better, typically < 1.0 is good)
2. The undistorted sample image to visually verify calibration quality

The calibration outputs:
- `camera_calibration_YYYYMMDD_HHMMSS.json`: Human-readable calibration parameters
- `camera_calibration_YYYYMMDD_HHMMSS.xml`: OpenCV format calibration file

### 5. Use the Calibration in Your Vision Pipeline

Update your vision processing code to use the calibration parameters:

```python
import cv2
import json

# Load calibration from JSON
with open('path/to/calibration.json', 'r') as f:
    calib_data = json.load(f)
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeffs = np.array(calib_data['dist_coeffs'])

# Or load from XML
fs = cv2.FileStorage('path/to/calibration.xml', cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('dist_coeffs').mat()
fs.release()

# Undistort an image
def undistort_image(img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    return undistorted
```

## Troubleshooting

- **Corners not detected**: Ensure good lighting and that the checkerboard is fully visible
- **High reprojection error**: Try capturing more images with the checkerboard in different orientations
- **Undistorted images look strange**: Double-check your checkerboard dimensions and square size

## Additional Resources

- [OpenCV Camera Calibration Documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [WPILib Vision Processing](https://docs.wpilib.org/en/stable/docs/software/vision-processing/introduction/index.html) 