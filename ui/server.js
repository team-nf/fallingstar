const express = require('express');
const http = require('http');
const path = require('path');
const { Server } = require('socket.io');
const { spawn } = require('child_process');
const cors = require('cors');

// Create Express app
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(express.json());

// In production, serve the React app
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, 'build')));
  
  app.get('/*', function (req, res) {
    res.sendFile(path.join(__dirname, 'build', 'index.html'));
  });
}

// API endpoints
app.get('/api/config', (req, res) => {
  // Return the current configuration
  res.json({
    teamNumber: process.env.TEAM_NUMBER || '9029',
    enableUI: process.env.ENABLE_UI === 'true',
    configPath: process.env.CONFIG_PATH || 'config/pc_config.json'
  });
});

app.post('/api/config', (req, res) => {
  // Update configuration
  // This is a placeholder - actual implementation would update the config file
  console.log('Received config update:', req.body);
  res.json({ success: true });
});

// Camera stream endpoint
app.get('/api/stream', (req, res) => {
  // This is a placeholder - in a real implementation, you would
  // configure a proper video streaming endpoint using something like
  // MJPEG, WebRTC, or a websocket-based solution
  res.redirect('/static/images/stream_placeholder.jpg');
});

// Generate some random data for demo purposes
function generateRandomData() {
  return {
    fps: Math.floor(Math.random() * 10) + 25,
    targets: Math.floor(Math.random() * 3),
    latency: Math.floor(Math.random() * 20) + 10,
    position: {
      x: parseFloat((Math.random() * 10 - 5).toFixed(2)),
      y: parseFloat((Math.random() * 10 - 5).toFixed(2)),
      z: parseFloat((Math.random() * 10).toFixed(2))
    }
  };
}

// Socket.IO connection
io.on('connection', (socket) => {
  console.log('Client connected');
  
  // Send initial data
  socket.emit('connection_status', { connected: true });
  
  // Simulate sending stream data
  const streamDataInterval = setInterval(() => {
    socket.emit('stream_data', generateRandomData());
  }, 1000);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected');
    clearInterval(streamDataInterval);
  });
});

// Start the server
const PORT = process.env.REACT_APP_SERVER_PORT || 9029;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`React Server running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV}`);
}); 