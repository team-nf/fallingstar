import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Button,
  Typography,
  Slider,
  List,
  ListItem,
  ListItemText,
  IconButton
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  FileUpload as UploadIcon,
  FileDownload as DownloadIcon
} from '@mui/icons-material';

// Mock data
const mockCameras = [
  { id: 'camera1', name: 'USB Camera', resolution: '640x480' },
  { id: 'camera2', name: 'IP Camera', resolution: '1280x720' },
  { id: 'camera3', name: 'Raspberry Pi Camera', resolution: '1920x1080' }
];

const mockModels = [
  { id: 'model1', name: 'YOLO v5', description: 'General purpose object detection' },
  { id: 'model2', name: 'SSD MobileNet', description: 'Lightweight model for edge devices' },
  { id: 'model3', name: 'Cone Detector', description: 'Specialized for FRC field cones' }
];

const mockDetections = [
  { id: 1, label: 'Cone', confidence: 0.92, bbox: { x: 120, y: 80, width: 100, height: 150 } },
  { id: 2, label: 'Cube', confidence: 0.87, bbox: { x: 320, y: 220, width: 80, height: 80 } },
  { id: 3, label: 'Robot', confidence: 0.75, bbox: { x: 450, y: 150, width: 140, height: 120 } }
];

const Detection = () => {
  const [selectedCamera, setSelectedCamera] = useState('');
  const [cameras, setCameras] = useState(mockCameras);
  const [selectedModel, setSelectedModel] = useState('');
  const [models, setModels] = useState(mockModels);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [detections, setDetections] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);

  useEffect(() => {
    document.title = 'Detection - FallingStar';
  }, []);

  const handleCameraChange = (event) => {
    setSelectedCamera(event.target.value);
    setIsDetecting(false);
    setDetections([]);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
    setIsDetecting(false);
    setDetections([]);
  };

  const handleConfidenceChange = (event, newValue) => {
    setConfidenceThreshold(newValue);
  };

  const startDetection = () => {
    setIsDetecting(true);
    setDetections(mockDetections.filter(d => d.confidence >= confidenceThreshold));
  };

  const stopDetection = () => {
    setIsDetecting(false);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Object Detection
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card elevation={3}>
            <CardHeader title="Camera & Model Selection" />
            <Divider />
            <CardContent>
              <FormControl fullWidth margin="normal">
                <InputLabel id="detection-camera-select-label">Select Camera</InputLabel>
                <Select
                  labelId="detection-camera-select-label"
                  id="detection-camera-select"
                  value={selectedCamera}
                  label="Select Camera"
                  onChange={handleCameraChange}
                >
                  {cameras.map((camera) => (
                    <MenuItem key={camera.id} value={camera.id}>
                      {camera.name} ({camera.resolution})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth margin="normal">
                <InputLabel id="model-select-label">Select Model</InputLabel>
                <Select
                  labelId="model-select-label"
                  id="model-select"
                  value={selectedModel}
                  label="Select Model"
                  onChange={handleModelChange}
                >
                  {models.map((model) => (
                    <MenuItem key={model.id} value={model.id}>
                      {model.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedModel && (
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1, mb: 2 }}>
                  {models.find(m => m.id === selectedModel)?.description}
                </Typography>
              )}

              <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                <Button
                  variant="outlined"
                  startIcon={<UploadIcon />}
                >
                  Upload Model
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  disabled={!selectedModel}
                >
                  Download
                </Button>
              </Box>
            </CardContent>
          </Card>

          <Card elevation={3} sx={{ mt: 3 }}>
            <CardHeader title="Detection Settings" />
            <Divider />
            <CardContent>
              <Typography gutterBottom>
                Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
              </Typography>
              <Slider
                value={confidenceThreshold}
                onChange={handleConfidenceChange}
                min={0.1}
                max={1.0}
                step={0.05}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
              />

              <Button
                variant="contained"
                color={isDetecting ? "error" : "primary"}
                fullWidth
                sx={{ mt: 2 }}
                onClick={isDetecting ? stopDetection : startDetection}
                disabled={!selectedCamera || !selectedModel}
              >
                {isDetecting ? "Stop Detection" : "Start Detection"}
              </Button>
            </CardContent>
          </Card>

          <Card elevation={3} sx={{ mt: 3 }}>
            <CardHeader title="Detected Objects" />
            <Divider />
            <CardContent>
              {detections.length > 0 ? (
                <List dense>
                  {detections.map((detection) => (
                    <ListItem
                      key={detection.id}
                      secondaryAction={
                        <IconButton edge="end" aria-label="view">
                          <VisibilityIcon />
                        </IconButton>
                      }
                    >
                      <ListItemText
                        primary={detection.label}
                        secondary={`Confidence: ${(detection.confidence * 100).toFixed(0)}%`}
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="textSecondary" align="center">
                  No objects detected
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card elevation={3} sx={{ height: '100%' }}>
            <CardHeader title="Detection Preview" />
            <Divider />
            <CardContent sx={{ position: 'relative', height: 'calc(100% - 56px)', minHeight: '500px' }}>
              {selectedCamera ? (
                <Box
                  sx={{
                    width: '100%',
                    height: '100%',
                    bgcolor: 'rgba(0, 0, 0, 0.1)',
                    position: 'relative',
                    overflow: 'hidden',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}
                >
                  <Typography variant="body2" color="textSecondary">
                    Camera Feed with Detection
                  </Typography>
                  
                  {/* Detection bounding boxes would be rendered here */}
                  {isDetecting && detections.map(detection => (
                    <Box
                      key={detection.id}
                      sx={{
                        position: 'absolute',
                        left: `${detection.bbox.x}px`,
                        top: `${detection.bbox.y}px`,
                        width: `${detection.bbox.width}px`,
                        height: `${detection.bbox.height}px`,
                        border: '2px solid #00C853',
                        borderRadius: '4px',
                        '&::before': {
                          content: `"${detection.label}: ${(detection.confidence * 100).toFixed(0)}%"`,
                          position: 'absolute',
                          top: '-20px',
                          left: '-2px',
                          backgroundColor: 'rgba(0, 200, 83, 0.8)',
                          color: 'white',
                          padding: '2px 4px',
                          fontSize: '0.75rem',
                          borderRadius: '2px'
                        }
                      }}
                    />
                  ))}
                </Box>
              ) : (
                <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center' }}>
                  Select a camera and model to begin detection
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Detection; 