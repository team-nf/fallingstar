const express = require('express');
const path = require('path');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
const fs = require('fs');
const axios = require('axios');

// Create Express app
const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

// Vision API URL
const VISION_API_URL = process.env.VISION_API_URL || 'http://vision:5000';

// Middleware
app.use(cors());
app.use(express.json());

// Create necessary directories if they don't exist
const configDir = path.join(__dirname, 'config');

if (!fs.existsSync(configDir)) {
  fs.mkdirSync(configDir);
}

// Global settings object
let globalSettings = {
  camera: {
    selectedCamera: '',
    brightness: 50,
    contrast: 50,
    exposure: 50
  },
  calibration: {},
  detection: {},
  tracking: {},
  selection: {},
  pnp: {}
};

// Save settings
app.post('/api/settings', (req, res) => {
  try {
    const newSettings = req.body;
    globalSettings = { ...globalSettings, ...newSettings };
    
    // Save to file
    fs.writeFileSync(
      path.join(configDir, 'settings.json'),
      JSON.stringify(globalSettings, null, 2)
    );
    
    res.json({ success: true });
  } catch (error) {
    console.error('Error saving settings:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get settings
app.get('/api/settings', (req, res) => {
  try {
    if (fs.existsSync(path.join(configDir, 'settings.json'))) {
      const settings = JSON.parse(
        fs.readFileSync(path.join(configDir, 'settings.json'), 'utf8')
      );
      res.json(settings);
    } else {
      res.json(globalSettings);
    }
  } catch (error) {
    console.error('Error getting settings:', error);
    res.status(500).json({ error: error.message });
  }
});

// Proxy requests to vision API endpoints
// Get available cameras
app.get('/api/cameras', async (req, res) => {
  try {
    console.log(`Fetching cameras from ${VISION_API_URL}/api/cameras`);
    const response = await axios.get(`${VISION_API_URL}/api/cameras`);
    console.log('Camera response:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Error getting cameras:', error);
    res.status(500).json({ error: error.message || 'Failed to get cameras' });
  }
});

// Get camera stream
app.get('/api/camera/:id/stream', async (req, res) => {
  try {
    const response = await axios.get(`${VISION_API_URL}/api/camera/${req.params.id}/stream`);
    res.json(response.data);

    // Emit to socket clients if we got frame data
    if (response.data && response.data.frame) {
      io.emit('camera_frame', response.data);
    }
  } catch (error) {
    console.error('Error getting camera stream:', error);
    res.status(500).json({ error: error.message || 'Failed to get camera stream' });
  }
});

// Calibrate camera
app.post('/api/camera/:id/calibrate', async (req, res) => {
  try {
    const response = await axios.post(
      `${VISION_API_URL}/api/camera/${req.params.id}/calibrate`,
      { ...req.body, ...globalSettings.calibration }
    );
    res.json(response.data);

    // Emit to socket clients
    io.emit('calibration_result', response.data);
  } catch (error) {
    console.error('Error calibrating camera:', error);
    res.status(500).json({ error: error.message || 'Failed to calibrate camera' });
  }
});

// Detect targets
app.post('/api/detect', async (req, res) => {
  try {
    const response = await axios.post(
      `${VISION_API_URL}/api/detect`,
      { ...req.body, ...globalSettings.detection }
    );
    res.json(response.data);

    // Emit to socket clients
    io.emit('detection_result', response.data);
  } catch (error) {
    console.error('Error detecting targets:', error);
    res.status(500).json({ error: error.message || 'Failed to detect targets' });
  }
});

// Estimate pose using PnP
app.post('/api/pnp', async (req, res) => {
  try {
    const response = await axios.post(
      `${VISION_API_URL}/api/pnp`,
      { ...req.body, ...globalSettings.pnp }
    );
    res.json(response.data);

    // Emit to socket clients
    io.emit('pnp_result', response.data);
  } catch (error) {
    console.error('Error estimating pose:', error);
    res.status(500).json({ error: error.message || 'Failed to estimate pose' });
  }
});

// Vision API health check
app.get('/api/vision-health', async (req, res) => {
  try {
    const response = await axios.get(`${VISION_API_URL}/api/health`);
    res.json(response.data);
  } catch (error) {
    console.error('Error checking vision API health:', error);
    res.status(500).json({ error: error.message || 'Vision API unavailable' });
  }
});

// Socket.io
io.on('connection', (socket) => {
  console.log('Client connected');
  
  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

// This app is an API server, not serving React static files
// The React app is served by the dev server on a different port

// Start server
const PORT = process.env.PORT || 9029;
server.listen(PORT, () => {
  console.log(`API Server running on port ${PORT}`);
}); 