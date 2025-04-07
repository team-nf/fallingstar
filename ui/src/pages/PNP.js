import React, { useState, useEffect, useRef } from 'react';
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
  TextField,
  List,
  ListItem,
  ListItemText,
  Tab,
  Tabs,
  Paper
} from '@mui/material';
import { ThreeDRotation as PoseIcon } from '@mui/icons-material';
import * as THREE from 'three';

// Mock data
const mockCameras = [
  { id: 'camera1', name: 'USB Camera', resolution: '640x480' },
  { id: 'camera2', name: 'IP Camera', resolution: '1280x720' },
  { id: 'camera3', name: 'Raspberry Pi Camera', resolution: '1920x1080' }
];

// Mock pre-defined targets
const predefinedTargets = [
  { id: 'apriltag', name: 'AprilTag', description: 'Standard AprilTag markers' },
  { id: 'retroreflective', name: 'Retroreflective Tape', description: 'FRC field retroreflective tape' },
  { id: 'cube', name: 'Cube', description: '3D cube with known dimensions' },
  { id: 'custom', name: 'Custom Target', description: 'User-defined custom target' }
];

// Mock pose estimation results
const mockPoseData = {
  translation: { x: 1.2, y: 0.7, z: 3.5 }, // meters
  rotation: { roll: 5, pitch: 2, yaw: 15 }, // degrees
  timestamp: Date.now()
};

// Tab panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`pnp-tabpanel-${index}`}
      aria-labelledby={`pnp-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && (
        <Box sx={{ p: 3, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const PNP = () => {
  const [selectedCamera, setSelectedCamera] = useState('');
  const [cameras, setCameras] = useState(mockCameras);
  const [selectedTarget, setSelectedTarget] = useState('');
  const [targets, setTargets] = useState(predefinedTargets);
  const [isPoseEstimating, setIsPoseEstimating] = useState(false);
  const [poseData, setPoseData] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [customDimensions, setCustomDimensions] = useState({ width: 0.2, height: 0.2, depth: 0.2 });
  const [fieldPosition, setFieldPosition] = useState({ x: 0, y: 0 });
  const threeContainer = useRef(null);
  const scene = useRef(null);
  const camera = useRef(null);
  const renderer = useRef(null);
  const cube = useRef(null);

  // Initialize Three.js scene
  useEffect(() => {
    if (threeContainer.current && isPoseEstimating && poseData) {
      // Clear previous scene
      while (threeContainer.current.firstChild) {
        threeContainer.current.removeChild(threeContainer.current.firstChild);
      }

      // Initialize scene
      scene.current = new THREE.Scene();
      camera.current = new THREE.PerspectiveCamera(75, threeContainer.current.clientWidth / threeContainer.current.clientHeight, 0.1, 1000);
      renderer.current = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.current.setSize(threeContainer.current.clientWidth, threeContainer.current.clientHeight);
      threeContainer.current.appendChild(renderer.current.domElement);

      // Add coordinate axes
      const axesHelper = new THREE.AxesHelper(5);
      scene.current.add(axesHelper);

      // Create a ground grid
      const gridHelper = new THREE.GridHelper(10, 10);
      gridHelper.rotation.x = Math.PI / 2;
      scene.current.add(gridHelper);

      // Create a robot representation (simple cube)
      const robotGeometry = new THREE.BoxGeometry(1, 0.5, 1);
      const robotMaterial = new THREE.MeshBasicMaterial({ color: 0x3f51b5, wireframe: true });
      cube.current = new THREE.Mesh(robotGeometry, robotMaterial);
      
      // Position based on PnP data
      cube.current.position.set(
        poseData.translation.x,
        poseData.translation.y,
        poseData.translation.z
      );
      
      // Rotation based on PnP data (convert degrees to radians)
      cube.current.rotation.x = THREE.MathUtils.degToRad(poseData.rotation.roll);
      cube.current.rotation.y = THREE.MathUtils.degToRad(poseData.rotation.yaw);
      cube.current.rotation.z = THREE.MathUtils.degToRad(poseData.rotation.pitch);
      
      scene.current.add(cube.current);

      // Position camera
      camera.current.position.set(5, 5, 5);
      camera.current.lookAt(0, 0, 0);

      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate);
        renderer.current.render(scene.current, camera.current);
      };
      animate();

      // Add window resize handler
      const handleResize = () => {
        if (camera.current && renderer.current && threeContainer.current) {
          camera.current.aspect = threeContainer.current.clientWidth / threeContainer.current.clientHeight;
          camera.current.updateProjectionMatrix();
          renderer.current.setSize(threeContainer.current.clientWidth, threeContainer.current.clientHeight);
        }
      };
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        if (renderer.current && renderer.current.domElement) {
          threeContainer.current?.removeChild(renderer.current.domElement);
        }
      };
    }
  }, [isPoseEstimating, poseData]);

  useEffect(() => {
    document.title = 'PnP - FallingStar';
  }, []);

  const handleCameraChange = (event) => {
    setSelectedCamera(event.target.value);
    setIsPoseEstimating(false);
    setPoseData(null);
  };

  const handleTargetChange = (event) => {
    setSelectedTarget(event.target.value);
    setIsPoseEstimating(false);
    setPoseData(null);
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleDimensionChange = (dimension) => (event) => {
    setCustomDimensions({
      ...customDimensions,
      [dimension]: parseFloat(event.target.value)
    });
  };

  const handleFieldPositionChange = (axis) => (event) => {
    setFieldPosition({
      ...fieldPosition,
      [axis]: parseFloat(event.target.value)
    });
  };

  const startPoseEstimation = () => {
    setIsPoseEstimating(true);
    // In a real application, this would actually compute PnP
    // For the mockup, we just use the mock data
    setPoseData(mockPoseData);
  };

  const stopPoseEstimation = () => {
    setIsPoseEstimating(false);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Perspective-n-Point (PnP)
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card elevation={3}>
            <CardHeader title="Camera & Target Selection" />
            <Divider />
            <CardContent>
              <FormControl fullWidth margin="normal">
                <InputLabel id="pnp-camera-select-label">Select Camera</InputLabel>
                <Select
                  labelId="pnp-camera-select-label"
                  id="pnp-camera-select"
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
                <InputLabel id="target-select-label">Select Target</InputLabel>
                <Select
                  labelId="target-select-label"
                  id="target-select"
                  value={selectedTarget}
                  label="Select Target"
                  onChange={handleTargetChange}
                >
                  {targets.map((target) => (
                    <MenuItem key={target.id} value={target.id}>
                      {target.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedTarget && (
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1, mb: 2 }}>
                  {targets.find(t => t.id === selectedTarget)?.description}
                </Typography>
              )}

              {selectedTarget === 'custom' && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Custom Target Dimensions (meters)
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <TextField
                        label="Width"
                        type="number"
                        value={customDimensions.width}
                        onChange={handleDimensionChange('width')}
                        inputProps={{ step: 0.01, min: 0.01 }}
                        fullWidth
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={4}>
                      <TextField
                        label="Height"
                        type="number"
                        value={customDimensions.height}
                        onChange={handleDimensionChange('height')}
                        inputProps={{ step: 0.01, min: 0.01 }}
                        fullWidth
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={4}>
                      <TextField
                        label="Depth"
                        type="number"
                        value={customDimensions.depth}
                        onChange={handleDimensionChange('depth')}
                        inputProps={{ step: 0.01, min: 0.01 }}
                        fullWidth
                        size="small"
                      />
                    </Grid>
                  </Grid>
                </Box>
              )}
            </CardContent>
          </Card>

          <Card elevation={3} sx={{ mt: 3 }}>
            <CardHeader title="PnP Settings" />
            <Divider />
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Field Position (meters)
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="X Position"
                    type="number"
                    value={fieldPosition.x}
                    onChange={handleFieldPositionChange('x')}
                    inputProps={{ step: 0.1 }}
                    fullWidth
                    size="small"
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Y Position"
                    type="number"
                    value={fieldPosition.y}
                    onChange={handleFieldPositionChange('y')}
                    inputProps={{ step: 0.1 }}
                    fullWidth
                    size="small"
                    margin="normal"
                  />
                </Grid>
              </Grid>

              <Button
                variant="contained"
                color={isPoseEstimating ? "error" : "primary"}
                fullWidth
                sx={{ mt: 2 }}
                onClick={isPoseEstimating ? stopPoseEstimation : startPoseEstimation}
                disabled={!selectedCamera || !selectedTarget}
              >
                {isPoseEstimating ? "Stop PnP" : "Start PnP"}
              </Button>
            </CardContent>
          </Card>

          {isPoseEstimating && poseData && (
            <Card elevation={3} sx={{ mt: 3 }}>
              <CardHeader 
                title="Pose Estimation Results" 
                avatar={<PoseIcon />}
              />
              <Divider />
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Translation (meters)
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={4}>
                    <Typography variant="body2">
                      X: {poseData.translation.x.toFixed(2)}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2">
                      Y: {poseData.translation.y.toFixed(2)}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2">
                      Z: {poseData.translation.z.toFixed(2)}
                    </Typography>
                  </Grid>
                </Grid>

                <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                  Rotation (degrees)
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={4}>
                    <Typography variant="body2">
                      Roll: {poseData.rotation.roll.toFixed(1)}°
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2">
                      Pitch: {poseData.rotation.pitch.toFixed(1)}°
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2">
                      Yaw: {poseData.rotation.yaw.toFixed(1)}°
                    </Typography>
                  </Grid>
                </Grid>

                <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                  Last updated: {new Date(poseData.timestamp).toLocaleTimeString()}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        <Grid item xs={12} md={8}>
          <Card elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardHeader title="PnP Visualization" />
            <Divider />
            <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={tabValue} onChange={handleTabChange} aria-label="PnP visualization tabs">
                  <Tab label="3D View" id="pnp-tab-0" />
                  <Tab label="Camera View" id="pnp-tab-1" />
                  <Tab label="Field View" id="pnp-tab-2" />
                </Tabs>
              </Box>
              
              <Box sx={{ flexGrow: 1, position: 'relative', minHeight: '500px' }}>
                <TabPanel value={tabValue} index={0} style={{ height: '100%' }}>
                  {selectedCamera && selectedTarget ? (
                    <Box 
                      ref={threeContainer}
                      sx={{ 
                        width: '100%', 
                        height: '100%', 
                        bgcolor: 'rgba(0, 0, 0, 0.05)',
                        borderRadius: 1
                      }}
                    >
                      {!isPoseEstimating && (
                        <Box 
                          sx={{ 
                            position: 'absolute', 
                            top: 0, 
                            left: 0, 
                            width: '100%', 
                            height: '100%',
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center'
                          }}
                        >
                          <Typography variant="body1" color="textSecondary">
                            Start PnP to view 3D visualization
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  ) : (
                    <Box 
                      sx={{ 
                        width: '100%', 
                        height: '100%', 
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center'
                      }}
                    >
                      <Typography variant="body1" color="textSecondary">
                        Select a camera and target to begin
                      </Typography>
                    </Box>
                  )}
                </TabPanel>
                
                <TabPanel value={tabValue} index={1}>
                  <Box 
                    sx={{ 
                      width: '100%', 
                      height: '100%', 
                      bgcolor: 'rgba(0, 0, 0, 0.1)',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      position: 'relative',
                      borderRadius: 1
                    }}
                  >
                    <Typography variant="body2" color="textSecondary">
                      Camera View with Pose Overlay
                    </Typography>
                    
                    {isPoseEstimating && poseData && (
                      <Box 
                        sx={{ 
                          position: 'absolute',
                          bottom: 16,
                          left: 16,
                          padding: 1,
                          bgcolor: 'rgba(0, 0, 0, 0.7)',
                          color: 'white',
                          borderRadius: 1
                        }}
                      >
                        <Typography variant="body2">
                          Distance: {poseData.translation.z.toFixed(2)}m
                        </Typography>
                        <Typography variant="body2">
                          Angle: {poseData.rotation.yaw.toFixed(1)}°
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </TabPanel>
                
                <TabPanel value={tabValue} index={2}>
                  <Box 
                    sx={{ 
                      width: '100%', 
                      height: '100%', 
                      bgcolor: 'rgba(0, 0, 0, 0.1)',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      position: 'relative',
                      borderRadius: 1
                    }}
                  >
                    <Typography variant="body2" color="textSecondary">
                      Field View with Robot Position
                    </Typography>
                    
                    {isPoseEstimating && poseData && (
                      <Box 
                        sx={{ 
                          position: 'absolute',
                          width: 20,
                          height: 20,
                          bgcolor: 'primary.main',
                          borderRadius: '50%',
                          left: `calc(50% + ${fieldPosition.x * 30}px)`,
                          top: `calc(50% - ${fieldPosition.y * 30}px)`,
                          transform: `rotate(${poseData.rotation.yaw}deg)`,
                          '&::after': {
                            content: '""',
                            position: 'absolute',
                            width: 0,
                            height: 0,
                            borderLeft: '10px solid transparent',
                            borderRight: '10px solid transparent',
                            borderBottom: '20px solid primary.main',
                            top: -15,
                            left: 0
                          }
                        }}
                      />
                    )}
                  </Box>
                </TabPanel>
              </Box>
            </Box>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PNP; 