import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Grid,
  Button,
  Typography,
  TextField,
  Paper,
  Chip,
  LinearProgress,
  Alert,
  CircularProgress,
  Snackbar
} from '@mui/material';
import { 
  GridOn as ChessboardIcon,
  Save as SaveIcon,
  CameraEnhance as CalibrateIcon,
  Refresh as ResetIcon
} from '@mui/icons-material';
import { useSettings } from '../context/SettingsContext';
import { io } from 'socket.io-client';

const Calibration = () => {
  const { settings, updateSettings } = useSettings();
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [captureCount, setCaptureCount] = useState(0);
  const [calibrationStatus, setCalibrationStatus] = useState('');
  const [calibrated, setCalibrated] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [socket, setSocket] = useState(null);

  // Get camera settings
  const { selectedCamera } = settings.camera || {};
  const { chessboardSize = { width: 9, height: 6 }, squareSize = 25 } = settings.calibration || {};

  useEffect(() => {
    document.title = 'Calibration - FallingStar';
    
    // Connect to socket
    const newSocket = io(window.location.origin);
    setSocket(newSocket);

    // Listen for calibration events
    newSocket.on('calibration_result', (data) => {
      console.log('Calibration data:', data);
      if (data.progress) {
        setCalibrationProgress(data.progress);
      }
      if (data.status) {
        setCalibrationStatus(data.status);
      }
      if (data.result) {
        setCalibrated(true);
        updateSettings('calibration', { 
          result: data.result,
          calibrationFile: data.filename
        });
        setSuccess('Calibration completed successfully');
      }
    });

    newSocket.on('calibration_error', (data) => {
      console.error('Calibration error:', data);
      setError(`Calibration error: ${data}`);
      setIsCalibrating(false);
    });

    newSocket.on('calibration_exit', (data) => {
      console.log('Calibration process exited:', data);
      setIsCalibrating(false);
    });

    return () => {
      if (newSocket) {
        newSocket.disconnect();
      }
    };
  }, [updateSettings]);

  const handleChessboardSizeChange = (dimension) => (e) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      updateSettings('calibration', { 
        chessboardSize: {
          ...chessboardSize,
          [dimension]: value
        }
      });
    }
  };

  const handleSquareSizeChange = (e) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      updateSettings('calibration', { squareSize: value });
    }
  };

  const handleAutoCalibrate = async () => {
    if (!selectedCamera) {
      setError('Please select a camera in the Input tab first');
      return;
    }

    setIsCalibrating(true);
    setCalibrationStatus('Initializing calibration...');
    setCalibrationProgress(5);
    setCaptureCount(0);
    
    try {
      // Start calibration process on the server
      const response = await fetch('/api/vision/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode: 'calibration'
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to start calibration: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error starting calibration:', error);
      setError(error.message || 'Failed to start calibration');
      setIsCalibrating(false);
    }
  };

  const saveCalibration = () => {
    // In a real app, this would update network tables or some other configuration
    const timestamp = new Date().toISOString().replace(/:/g, '-');
    setSuccess(`Calibration saved and applied for camera: ${selectedCamera}`);
  };

  const resetCalibration = async () => {
    setCalibrated(false);
    setCaptureCount(0);
    setCalibrationProgress(0);
    setCalibrationStatus('');
    
    try {
      // Stop any running calibration process
      if (isCalibrating) {
        await fetch('/api/vision/stop', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            mode: 'calibration'
          }),
        });
        setIsCalibrating(false);
      }
      
      // Clear calibration result
      updateSettings('calibration', { result: null, calibrationFile: null });
    } catch (error) {
      console.error('Error resetting calibration:', error);
      setError(error.message || 'Failed to reset calibration');
    }
  };

  const handleAlertClose = () => {
    setError('');
    setSuccess('');
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Camera Calibration
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card elevation={3}>
            <CardHeader title="Camera Status" />
            <Divider />
            <CardContent>
              {selectedCamera ? (
                <Alert severity="info" sx={{ mb: 2 }}>
                  Using camera: {selectedCamera}
                </Alert>
              ) : (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  No camera selected. Please select a camera in the Input tab first.
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card elevation={3} sx={{ mt: 3 }}>
            <CardHeader 
              title="Calibration Settings" 
              avatar={<ChessboardIcon />}
            />
            <Divider />
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Chessboard Configuration
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Width"
                    type="number"
                    value={chessboardSize.width}
                    onChange={handleChessboardSizeChange('width')}
                    fullWidth
                    variant="outlined"
                    size="small"
                    margin="normal"
                    inputProps={{ min: 3, max: 20 }}
                    helperText="Interior corners"
                    disabled={isCalibrating}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Height"
                    type="number"
                    value={chessboardSize.height}
                    onChange={handleChessboardSizeChange('height')}
                    fullWidth
                    variant="outlined"
                    size="small"
                    margin="normal"
                    inputProps={{ min: 3, max: 20 }}
                    helperText="Interior corners"
                    disabled={isCalibrating}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    label="Square Size (mm)"
                    type="number"
                    value={squareSize}
                    onChange={handleSquareSizeChange}
                    fullWidth
                    variant="outlined"
                    size="small"
                    margin="normal"
                    inputProps={{ min: 5, max: 100 }}
                    disabled={isCalibrating}
                  />
                </Grid>
              </Grid>

              <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                Calibration Actions
              </Typography>

              <Button 
                variant="contained" 
                color="primary"
                fullWidth
                startIcon={isCalibrating ? <CircularProgress size={24} color="inherit" /> : <CalibrateIcon />}
                onClick={handleAutoCalibrate}
                disabled={!selectedCamera || isCalibrating}
                sx={{ mt: 1 }}
              >
                {isCalibrating ? "Calibrating..." : "Auto Calibrate"}
              </Button>

              <Button 
                variant="outlined" 
                color="primary"
                fullWidth
                startIcon={<SaveIcon />}
                onClick={saveCalibration}
                disabled={!calibrated || isCalibrating}
                sx={{ mt: 1 }}
              >
                Save Calibration
              </Button>

              <Button 
                variant="outlined" 
                color="secondary"
                fullWidth
                startIcon={<ResetIcon />}
                onClick={resetCalibration}
                disabled={(!calibrated && calibrationProgress === 0) || isCalibrating}
                sx={{ mt: 1 }}
              >
                Reset
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card elevation={3}>
            <CardHeader 
              title="Calibration Status" 
              subheader={
                calibrationProgress > 0 && calibrationProgress < 100 
                  ? `Calibrating... ${calibrationProgress}%` 
                  : calibrated ? "Calibration Complete" : "Not Calibrated"
              }
            />
            <Divider />
            <CardContent>
              {isCalibrating && (
                <Box sx={{ width: '100%', mb: 2 }}>
                  <LinearProgress variant="determinate" value={calibrationProgress} />
                </Box>
              )}

              {calibrationStatus && (
                <Alert severity={calibrated ? "success" : "info"} sx={{ mb: 2 }}>
                  {calibrationStatus}
                </Alert>
              )}

              {captureCount > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Chip 
                    label={`${captureCount} images captured`} 
                    color="primary" 
                    variant="outlined" 
                  />
                </Box>
              )}

              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom align="center">
                    Original Image
                  </Typography>
                  <Paper 
                    sx={{ 
                      height: 300, 
                      bgcolor: 'background.paper',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      position: 'relative'
                    }}
                  >
                    {selectedCamera ? (
                      <Box 
                        sx={{ 
                          width: '100%', 
                          height: '100%', 
                          bgcolor: 'rgba(0, 0, 0, 0.1)',
                          display: 'flex',
                          justifyContent: 'center',
                          alignItems: 'center'
                        }}
                      >
                        <Typography variant="body2" color="textSecondary">
                          {isCalibrating ? 'Calibration in progress...' : 'Live Camera Feed'}
                        </Typography>
                      </Box>
                    ) : (
                      <Typography variant="body2" color="textSecondary">
                        Select a camera in the Input tab to view
                      </Typography>
                    )}
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom align="center">
                    Undistorted Image
                  </Typography>
                  <Paper 
                    sx={{ 
                      height: 300, 
                      bgcolor: 'background.paper',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center'
                    }}
                  >
                    {selectedCamera && calibrated ? (
                      <Box 
                        sx={{ 
                          width: '100%', 
                          height: '100%', 
                          bgcolor: 'rgba(0, 0, 0, 0.1)',
                          display: 'flex',
                          justifyContent: 'center',
                          alignItems: 'center'
                        }}
                      >
                        <Typography variant="body2" color="textSecondary">
                          Undistorted Camera Feed
                        </Typography>
                      </Box>
                    ) : (
                      <Typography variant="body2" color="textSecondary">
                        {selectedCamera 
                          ? "Calibrate camera to see undistorted view" 
                          : "Select a camera and calibrate"}
                      </Typography>
                    )}
                  </Paper>
                </Grid>
              </Grid>
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

export default Calibration;