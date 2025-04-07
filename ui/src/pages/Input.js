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
  TextField,
  Typography,
  Slider,
  Button,
  Paper,
  CircularProgress,
  Snackbar,
  Alert
} from '@mui/material';
import { useSettings } from '../context/SettingsContext';

const Input = () => {
  const { settings, updateSettings } = useSettings();
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const videoRef = useRef(null);

  // Load camera settings from context
  const {
    selectedCamera = '',
    brightness = 50,
    contrast = 50,
    exposure = 50
  } = settings.camera || {};

  // Fetch camera list from API
  const fetchCameras = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:9029/api/cameras');
      if (!response.ok) {
        throw new Error(`Failed to fetch cameras: ${response.statusText}`);
      }
      const data = await response.json();
      setCameras(data);
      setSuccess('Camera list refreshed successfully');
    } catch (error) {
      console.error('Error fetching cameras:', error);
      setError(error.message || 'Failed to fetch camera list');
      // Fallback to mock data in case of error
      setCameras([
        { id: 'camera0', name: 'USB Camera', resolution: '640x480' },
        { id: 'camera1', name: 'Built-in Camera', resolution: '1280x720' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Fetch cameras on component mount
  useEffect(() => {
    fetchCameras();
    document.title = 'Input - FallingStar';
  }, []);

  // Handle camera selection
  const handleCameraChange = (event) => {
    const cameraId = event.target.value;
    updateSettings('camera', { selectedCamera: cameraId });
    
    // Update resolution info
    const selectedCam = cameras.find(cam => cam.id === cameraId);
    if (selectedCam) {
      const [width, height] = selectedCam.resolution.split('x').map(Number);
      updateSettings('camera', { 
        resolution: { width, height }
      });
    }
  };

  // Handle camera settings changes
  const handleSettingChange = (setting) => (event, newValue) => {
    updateSettings('camera', { [setting]: newValue });
  };

  // Apply camera settings to the device
  const applySettings = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:9029/api/camera/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          camera: selectedCamera,
          settings: {
            brightness,
            contrast,
            exposure
          }
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to apply settings: ${response.statusText}`);
      }
      
      setSuccess('Camera settings applied successfully');
    } catch (error) {
      console.error('Error applying camera settings:', error);
      setError(error.message || 'Failed to apply camera settings');
    } finally {
      setLoading(false);
    }
  };

  // Handle error or success alert close
  const handleAlertClose = () => {
    setError('');
    setSuccess('');
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Camera Input
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card elevation={3}>
            <CardHeader title="Camera Selection" />
            <Divider />
            <CardContent>
              <FormControl fullWidth margin="normal">
                <InputLabel id="camera-select-label">Select Camera</InputLabel>
                <Select
                  labelId="camera-select-label"
                  id="camera-select"
                  value={selectedCamera}
                  label="Select Camera"
                  onChange={handleCameraChange}
                  disabled={loading}
                >
                  {cameras.map((camera) => (
                    <MenuItem key={camera.id} value={camera.id}>
                      {camera.name} ({camera.resolution})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedCamera && (
                <Typography variant="subtitle2" sx={{ mt: 2 }}>
                  Resolution: {cameras.find(c => c.id === selectedCamera)?.resolution || 'Unknown'}
                </Typography>
              )}

              <Button 
                variant="outlined" 
                color="primary" 
                fullWidth 
                sx={{ mt: 2 }}
                onClick={fetchCameras}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                {loading ? 'Refreshing...' : 'Refresh Camera List'}
              </Button>
            </CardContent>
          </Card>

          <Card elevation={3} sx={{ mt: 3 }}>
            <CardHeader title="Camera Settings" />
            <Divider />
            <CardContent>
              <Typography gutterBottom>Brightness</Typography>
              <Slider
                value={brightness}
                onChange={handleSettingChange('brightness')}
                valueLabelDisplay="auto"
                disabled={!selectedCamera || loading}
              />

              <Typography gutterBottom sx={{ mt: 2 }}>Contrast</Typography>
              <Slider
                value={contrast}
                onChange={handleSettingChange('contrast')}
                valueLabelDisplay="auto"
                disabled={!selectedCamera || loading}
              />

              <Typography gutterBottom sx={{ mt: 2 }}>Exposure</Typography>
              <Slider
                value={exposure}
                onChange={handleSettingChange('exposure')}
                valueLabelDisplay="auto"
                disabled={!selectedCamera || loading}
              />

              <Button 
                variant="contained" 
                color="primary" 
                fullWidth 
                sx={{ mt: 2 }}
                onClick={applySettings}
                disabled={!selectedCamera || loading}
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
              >
                {loading ? 'Applying...' : 'Apply Settings'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card elevation={3} sx={{ height: '100%' }}>
            <CardHeader title="Camera Stream" />
            <Divider />
            <CardContent sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100% - 56px)' }}>
              {selectedCamera ? (
                <Box 
                  ref={videoRef}
                  component="div"
                  sx={{
                    width: '100%',
                    height: '100%',
                    backgroundColor: 'rgba(0, 0, 0, 0.1)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    borderRadius: 1,
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                >
                  {/* Placeholder for actual video stream - in a real app we would connect to a WebRTC or WebSocket stream */}
                  <Paper 
                    elevation={0} 
                    sx={{ 
                      bgcolor: 'background.paper', 
                      p: 2, 
                      textAlign: 'center',
                      borderRadius: 2
                    }}
                  >
                    <Typography variant="h6">
                      Camera Preview
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Streaming from {cameras.find(cam => cam.id === selectedCamera)?.name || 'unknown camera'}
                    </Typography>
                    <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                      Brightness: {brightness}% | Contrast: {contrast}% | Exposure: {exposure}%
                    </Typography>
                  </Paper>
                </Box>
              ) : (
                <Typography variant="body1" color="textSecondary">
                  Select a camera to view the stream
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Error Snackbar */}
      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleAlertClose}>
        <Alert onClose={handleAlertClose} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>

      {/* Success Snackbar */}
      <Snackbar open={!!success} autoHideDuration={6000} onClose={handleAlertClose}>
        <Alert onClose={handleAlertClose} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Input; 