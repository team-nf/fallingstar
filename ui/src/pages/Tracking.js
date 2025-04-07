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
  Settings as SettingsIcon
} from '@mui/icons-material';

// Mock data
const mockCameras = [
  { id: 'camera1', name: 'USB Camera', resolution: '640x480' },
  { id: 'camera2', name: 'IP Camera', resolution: '1280x720' },
  { id: 'camera3', name: 'Raspberry Pi Camera', resolution: '1920x1080' }
];

const trackingAlgorithms = [
  { id: 'kalman', name: 'Kalman Filter', description: 'Good for smooth motion tracking' },
  { id: 'sort', name: 'SORT', description: 'Simple Online and Realtime Tracking' },
  { id: 'deep_sort', name: 'DeepSORT', description: 'Deep learning enhanced tracking' },
  { id: 'kcf', name: 'KCF Tracker', description: 'Kernelized Correlation Filters' }
];

// Mock tracked objects with historical positions for trails
const mockTrackedObjects = [
  { 
    id: 1, 
    label: 'Cone', 
    confidence: 0.92, 
    currentPosition: { x: 120, y: 80, width: 100, height: 150 },
    history: [
      { x: 80, y: 60, width: 100, height: 150, timestamp: Date.now() - 3000 },
      { x: 95, y: 65, width: 100, height: 150, timestamp: Date.now() - 2500 },
      { x: 105, y: 70, width: 100, height: 150, timestamp: Date.now() - 2000 },
      { x: 110, y: 75, width: 100, height: 150, timestamp: Date.now() - 1500 },
      { x: 115, y: 78, width: 100, height: 150, timestamp: Date.now() - 1000 },
      { x: 118, y: 79, width: 100, height: 150, timestamp: Date.now() - 500 },
      { x: 120, y: 80, width: 100, height: 150, timestamp: Date.now() }
    ]
  },
  { 
    id: 2, 
    label: 'Cube', 
    confidence: 0.87, 
    currentPosition: { x: 320, y: 220, width: 80, height: 80 },
    history: [
      { x: 370, y: 180, width: 80, height: 80, timestamp: Date.now() - 3000 },
      { x: 360, y: 190, width: 80, height: 80, timestamp: Date.now() - 2500 },
      { x: 350, y: 200, width: 80, height: 80, timestamp: Date.now() - 2000 },
      { x: 340, y: 210, width: 80, height: 80, timestamp: Date.now() - 1500 },
      { x: 330, y: 215, width: 80, height: 80, timestamp: Date.now() - 1000 },
      { x: 325, y: 218, width: 80, height: 80, timestamp: Date.now() - 500 },
      { x: 320, y: 220, width: 80, height: 80, timestamp: Date.now() }
    ]
  },
  { 
    id: 3, 
    label: 'Robot', 
    confidence: 0.75, 
    currentPosition: { x: 450, y: 150, width: 140, height: 120 },
    history: [
      { x: 350, y: 250, width: 140, height: 120, timestamp: Date.now() - 3000 },
      { x: 370, y: 230, width: 140, height: 120, timestamp: Date.now() - 2500 },
      { x: 390, y: 210, width: 140, height: 120, timestamp: Date.now() - 2000 },
      { x: 410, y: 190, width: 140, height: 120, timestamp: Date.now() - 1500 },
      { x: 430, y: 170, width: 140, height: 120, timestamp: Date.now() - 1000 },
      { x: 440, y: 160, width: 140, height: 120, timestamp: Date.now() - 500 },
      { x: 450, y: 150, width: 140, height: 120, timestamp: Date.now() }
    ]
  }
];

const Tracking = () => {
  const [selectedCamera, setSelectedCamera] = useState('');
  const [cameras, setCameras] = useState(mockCameras);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [algorithms, setAlgorithms] = useState(trackingAlgorithms);
  const [isTracking, setIsTracking] = useState(false);
  const [trackedObjects, setTrackedObjects] = useState([]);
  const [trailDuration, setTrailDuration] = useState(3); // in seconds
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    document.title = 'Tracking - FallingStar';
  }, []);

  // Update current time for trail rendering
  useEffect(() => {
    if (isTracking) {
      const interval = setInterval(() => {
        setNow(Date.now());
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isTracking]);

  const handleCameraChange = (event) => {
    setSelectedCamera(event.target.value);
    setIsTracking(false);
    setTrackedObjects([]);
  };

  const handleAlgorithmChange = (event) => {
    setSelectedAlgorithm(event.target.value);
    setIsTracking(false);
    setTrackedObjects([]);
  };

  const handleTrailDurationChange = (event, newValue) => {
    setTrailDuration(newValue);
  };

  const startTracking = () => {
    setIsTracking(true);
    setTrackedObjects(mockTrackedObjects);
  };

  const stopTracking = () => {
    setIsTracking(false);
  };

  // Function to render the object trails
  const renderObjectTrails = (object) => {
    const currentTime = now;
    const minTime = currentTime - (trailDuration * 1000);
    
    // Filter history points based on trail duration
    const trailPoints = object.history.filter(point => point.timestamp >= minTime);
    
    // Skip if not enough points for a trail
    if (trailPoints.length < 2) return null;
    
    // Calculate center points of each bounding box
    const pathPoints = trailPoints.map(point => ({
      x: point.x + (point.width / 2),
      y: point.y + (point.height / 2),
      timestamp: point.timestamp
    }));
    
    // Create SVG path
    const pathData = `M ${pathPoints[0].x} ${pathPoints[0].y} ` + 
      pathPoints.slice(1).map(point => `L ${point.x} ${point.y}`).join(' ');
    
    return (
      <svg
        key={`trail-${object.id}`}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          zIndex: 1
        }}
      >
        <path
          d={pathData}
          stroke="#00C853"
          strokeWidth="2"
          fill="none"
          strokeOpacity="0.7"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Object Tracking
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card elevation={3}>
            <CardHeader title="Camera & Algorithm Selection" />
            <Divider />
            <CardContent>
              <FormControl fullWidth margin="normal">
                <InputLabel id="tracking-camera-select-label">Select Camera</InputLabel>
                <Select
                  labelId="tracking-camera-select-label"
                  id="tracking-camera-select"
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
                <InputLabel id="algorithm-select-label">Tracking Algorithm</InputLabel>
                <Select
                  labelId="algorithm-select-label"
                  id="algorithm-select"
                  value={selectedAlgorithm}
                  label="Tracking Algorithm"
                  onChange={handleAlgorithmChange}
                >
                  {algorithms.map((algorithm) => (
                    <MenuItem key={algorithm.id} value={algorithm.id}>
                      {algorithm.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedAlgorithm && (
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1, mb: 2 }}>
                  {algorithms.find(a => a.id === selectedAlgorithm)?.description}
                </Typography>
              )}
            </CardContent>
          </Card>

          <Card elevation={3} sx={{ mt: 3 }}>
            <CardHeader title="Tracking Settings" />
            <Divider />
            <CardContent>
              <Typography gutterBottom>
                Trail Duration: {trailDuration} seconds
              </Typography>
              <Slider
                value={trailDuration}
                onChange={handleTrailDurationChange}
                min={1}
                max={10}
                step={0.5}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}s`}
              />

              <Button
                variant="contained"
                color={isTracking ? "error" : "primary"}
                fullWidth
                sx={{ mt: 2 }}
                onClick={isTracking ? stopTracking : startTracking}
                disabled={!selectedCamera || !selectedAlgorithm}
              >
                {isTracking ? "Stop Tracking" : "Start Tracking"}
              </Button>
            </CardContent>
          </Card>

          <Card elevation={3} sx={{ mt: 3 }}>
            <CardHeader title="Tracked Objects" />
            <Divider />
            <CardContent>
              {trackedObjects.length > 0 ? (
                <List dense>
                  {trackedObjects.map((object) => (
                    <ListItem
                      key={object.id}
                      secondaryAction={
                        <IconButton edge="end" aria-label="settings">
                          <SettingsIcon />
                        </IconButton>
                      }
                    >
                      <ListItemText
                        primary={object.label}
                        secondary={`ID: ${object.id} | Confidence: ${(object.confidence * 100).toFixed(0)}%`}
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="textSecondary" align="center">
                  No objects tracked
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card elevation={3} sx={{ height: '100%' }}>
            <CardHeader title="Tracking Preview" />
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
                    Camera Feed with Tracking
                  </Typography>
                  
                  {/* Render object trails */}
                  {isTracking && trackedObjects.map(object => renderObjectTrails(object))}
                  
                  {/* Render current object boxes */}
                  {isTracking && trackedObjects.map(object => (
                    <Box
                      key={object.id}
                      sx={{
                        position: 'absolute',
                        left: `${object.currentPosition.x}px`,
                        top: `${object.currentPosition.y}px`,
                        width: `${object.currentPosition.width}px`,
                        height: `${object.currentPosition.height}px`,
                        border: '2px solid #00C853',
                        borderRadius: '4px',
                        '&::before': {
                          content: `"${object.label} #${object.id}"`,
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
                  Select a camera and algorithm to begin tracking
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Tracking; 