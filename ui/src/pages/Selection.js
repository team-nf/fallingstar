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
  Radio,
  RadioGroup,
  FormControlLabel,
  IconButton
} from '@mui/material';
import {
  Flag as FlagIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';

// Reuse the same mock data from Tracking
const mockCameras = [
  { id: 'camera1', name: 'USB Camera', resolution: '640x480' },
  { id: 'camera2', name: 'IP Camera', resolution: '1280x720' },
  { id: 'camera3', name: 'Raspberry Pi Camera', resolution: '1920x1080' }
];

const selectionAlgorithms = [
  { id: 'nearest', name: 'Nearest Target', description: 'Select the closest target to the center' },
  { id: 'largest', name: 'Largest Target', description: 'Select the largest target by area' },
  { id: 'highest_conf', name: 'Highest Confidence', description: 'Select the target with highest detection confidence' },
  { id: 'manual', name: 'Manual Selection', description: 'Allow user to select a target' }
];

// Same tracked objects as in Tracking.js but with color property
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
    ],
    color: '#00C853' // Default tracking color
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
    ],
    color: '#00C853'
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
    ],
    color: '#00C853'
  }
];

const Selection = () => {
  const [selectedCamera, setSelectedCamera] = useState('');
  const [cameras, setCameras] = useState(mockCameras);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [algorithms, setAlgorithms] = useState(selectionAlgorithms);
  const [isActive, setIsActive] = useState(false);
  const [objects, setObjects] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [trailDuration, setTrailDuration] = useState(3); // in seconds
  const [now, setNow] = useState(Date.now());
  const selectedColor = '#FF4081'; // Pink color for selected target

  useEffect(() => {
    document.title = 'Target Selection - FallingStar';
  }, []);

  // Update current time for trail rendering
  useEffect(() => {
    if (isActive) {
      const interval = setInterval(() => {
        setNow(Date.now());
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isActive]);

  const handleCameraChange = (event) => {
    setSelectedCamera(event.target.value);
    setIsActive(false);
    setObjects([]);
    setSelectedTarget(null);
  };

  const handleAlgorithmChange = (event) => {
    setSelectedAlgorithm(event.target.value);
    setIsActive(false);
    setObjects([]);
    setSelectedTarget(null);
  };

  const handleTrailDurationChange = (event, newValue) => {
    setTrailDuration(newValue);
  };

  const startSelection = () => {
    setIsActive(true);
    
    // Clone the objects to avoid mutating the original
    const objCopy = JSON.parse(JSON.stringify(mockTrackedObjects));
    setObjects(objCopy);
    
    // Select target based on algorithm
    selectTarget(objCopy);
  };

  const stopSelection = () => {
    setIsActive(false);
    setSelectedTarget(null);
  };

  const selectTarget = (objects) => {
    if (!objects || objects.length === 0) return;
    
    let target = null;
    
    switch (selectedAlgorithm) {
      case 'nearest':
        // Select the closest to center (simplified)
        target = objects.reduce((prev, current) => {
          const prevCenter = {
            x: prev.currentPosition.x + (prev.currentPosition.width / 2),
            y: prev.currentPosition.y + (prev.currentPosition.height / 2)
          };
          const currentCenter = {
            x: current.currentPosition.x + (current.currentPosition.width / 2),
            y: current.currentPosition.y + (current.currentPosition.height / 2)
          };
          
          // Distance from center of view (assumed to be at 320, 240)
          const viewCenter = { x: 320, y: 240 };
          const prevDist = Math.sqrt(Math.pow(prevCenter.x - viewCenter.x, 2) + Math.pow(prevCenter.y - viewCenter.y, 2));
          const currentDist = Math.sqrt(Math.pow(currentCenter.x - viewCenter.x, 2) + Math.pow(currentCenter.y - viewCenter.y, 2));
          
          return prevDist < currentDist ? prev : current;
        });
        break;
        
      case 'largest':
        // Select the largest target by area
        target = objects.reduce((prev, current) => {
          const prevArea = prev.currentPosition.width * prev.currentPosition.height;
          const currentArea = current.currentPosition.width * current.currentPosition.height;
          return prevArea > currentArea ? prev : current;
        });
        break;
        
      case 'highest_conf':
        // Select the target with highest confidence
        target = objects.reduce((prev, current) => {
          return prev.confidence > current.confidence ? prev : current;
        });
        break;
        
      case 'manual':
        // For manual, just select the first one initially
        target = objects[0];
        break;
        
      default:
        target = objects[0];
    }
    
    if (target) {
      // Set the selected target's color
      const updatedObjects = objects.map(obj => {
        if (obj.id === target.id) {
          return { ...obj, color: selectedColor };
        }
        return obj;
      });
      
      setObjects(updatedObjects);
      setSelectedTarget(target.id);
    }
  };

  const handleManualSelection = (objectId) => {
    if (selectedAlgorithm === 'manual' && isActive) {
      // Update colors
      const updatedObjects = objects.map(obj => {
        if (obj.id === objectId) {
          return { ...obj, color: selectedColor };
        } else {
          return { ...obj, color: '#00C853' };
        }
      });
      
      setObjects(updatedObjects);
      setSelectedTarget(objectId);
    }
  };

  // Function to render the object trails (similar to Tracking.js)
  const renderObjectTrails = (object) => {
    const currentTime = now;
    const minTime = currentTime - (trailDuration * 1000);
    
    // Filter history points based on trail duration
    const trailPoints = object.history.filter(point => point.timestamp >= minTime);
    
    // Skip if not enough points
    if (trailPoints.length < 2) return null;
    
    // Calculate center points
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
          stroke={object.color || '#00C853'}
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
        Target Selection
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card elevation={3}>
            <CardHeader title="Camera & Algorithm Selection" />
            <Divider />
            <CardContent>
              <FormControl fullWidth margin="normal">
                <InputLabel id="selection-camera-select-label">Select Camera</InputLabel>
                <Select
                  labelId="selection-camera-select-label"
                  id="selection-camera-select"
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
                <InputLabel id="selection-algorithm-select-label">Selection Algorithm</InputLabel>
                <Select
                  labelId="selection-algorithm-select-label"
                  id="selection-algorithm-select"
                  value={selectedAlgorithm}
                  label="Selection Algorithm"
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
            <CardHeader title="Selection Settings" />
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
                color={isActive ? "error" : "primary"}
                fullWidth
                sx={{ mt: 2 }}
                onClick={isActive ? stopSelection : startSelection}
                disabled={!selectedCamera || !selectedAlgorithm}
              >
                {isActive ? "Stop Selection" : "Start Selection"}
              </Button>
            </CardContent>
          </Card>

          <Card elevation={3} sx={{ mt: 3 }}>
            <CardHeader title="Tracked Objects" />
            <Divider />
            <CardContent>
              {objects.length > 0 ? (
                <RadioGroup
                  value={selectedTarget ? selectedTarget.toString() : ''}
                  onChange={(e) => handleManualSelection(Number(e.target.value))}
                >
                  {objects.map((object) => (
                    <FormControlLabel
                      key={object.id}
                      value={object.id.toString()}
                      control={<Radio disabled={selectedAlgorithm !== 'manual'} />}
                      label={
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                          <Typography>
                            {object.label} {object.id === selectedTarget ? "(Selected)" : ""}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            Confidence: {(object.confidence * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      }
                    />
                  ))}
                </RadioGroup>
              ) : (
                <Typography variant="body2" color="textSecondary" align="center">
                  No objects available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card elevation={3} sx={{ height: '100%' }}>
            <CardHeader title="Selection Preview" />
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
                    Camera Feed with Target Selection
                  </Typography>
                  
                  {/* Crosshair at center for reference */}
                  <Box
                    sx={{
                      position: 'absolute',
                      width: '100%',
                      height: '100%',
                      pointerEvents: 'none',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: '50%',
                        left: 0,
                        right: 0,
                        borderTop: '1px dashed rgba(255,255,255,0.3)',
                      },
                      '&::after': {
                        content: '""',
                        position: 'absolute',
                        left: '50%',
                        top: 0,
                        bottom: 0,
                        borderLeft: '1px dashed rgba(255,255,255,0.3)',
                      }
                    }}
                  />
                  
                  {/* Render object trails */}
                  {isActive && objects.map(object => renderObjectTrails(object))}
                  
                  {/* Render current object boxes */}
                  {isActive && objects.map(object => (
                    <Box
                      key={object.id}
                      sx={{
                        position: 'absolute',
                        left: `${object.currentPosition.x}px`,
                        top: `${object.currentPosition.y}px`,
                        width: `${object.currentPosition.width}px`,
                        height: `${object.currentPosition.height}px`,
                        border: `2px solid ${object.color || '#00C853'}`,
                        borderRadius: '4px',
                        cursor: selectedAlgorithm === 'manual' ? 'pointer' : 'default',
                        '&::before': {
                          content: `"${object.label} #${object.id}"`,
                          position: 'absolute',
                          top: '-20px',
                          left: '-2px',
                          backgroundColor: object.id === selectedTarget ? 'rgba(255, 64, 129, 0.8)' : 'rgba(0, 200, 83, 0.8)',
                          color: 'white',
                          padding: '2px 4px',
                          fontSize: '0.75rem',
                          borderRadius: '2px'
                        }
                      }}
                      onClick={() => handleManualSelection(object.id)}
                    />
                  ))}
                  
                  {/* Target indicator for selected target */}
                  {isActive && selectedTarget && objects.find(o => o.id === selectedTarget) && (
                    <Box
                      sx={{
                        position: 'absolute',
                        left: `${objects.find(o => o.id === selectedTarget).currentPosition.x + objects.find(o => o.id === selectedTarget).currentPosition.width/2 - 15}px`,
                        top: `${objects.find(o => o.id === selectedTarget).currentPosition.y + objects.find(o => o.id === selectedTarget).currentPosition.height/2 - 15}px`,
                        width: '30px',
                        height: '30px',
                        border: '2px solid white',
                        borderRadius: '50%',
                        zIndex: 2,
                        pointerEvents: 'none',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: '50%',
                          left: '-10px',
                          right: '-10px',
                          borderTop: '2px solid white'
                        },
                        '&::after': {
                          content: '""',
                          position: 'absolute',
                          left: '50%',
                          top: '-10px',
                          bottom: '-10px',
                          borderLeft: '2px solid white'
                        }
                      }}
                    />
                  )}
                </Box>
              ) : (
                <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center' }}>
                  Select a camera and algorithm to begin
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Selection; 