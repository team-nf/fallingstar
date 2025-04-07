const express = require('express');
const path = require('path');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
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

// Vision API URL with a more reliable connection when using host networking
const VISION_API_URL = process.env.VISION_API_URL || 'http://localhost:5000';

// Configure axios with longer timeouts and retry logic
const apiClient = axios.create({
  baseURL: VISION_API_URL,
  timeout: 5000,  // 5 second timeout
  maxRetries: 3,
  retryDelay: 500
});

// Add retry logic to axios
apiClient.interceptors.response.use(undefined, async (error) => {
  const { config } = error;
  if (!config || !config.maxRetries) return Promise.reject(error);
  
  config.retryCount = config.retryCount || 0;
  if (config.retryCount >= config.maxRetries) {
    return Promise.reject(error);
  }
  
  config.retryCount += 1;
  console.log(`Retrying request to ${config.url}, attempt ${config.retryCount}/${config.maxRetries}`);
  
  // Wait before retrying
  await new Promise(resolve => setTimeout(resolve, config.retryDelay || 1000));
  return apiClient(config);
});

// Middleware
app.use(cors());
app.use(express.json());

// Special handling for MJPEG streams
app.get('/api/camera/:id/stream/mjpeg', async (req, res) => {
  try {
    console.log(`Piping MJPEG stream from ${VISION_API_URL}/api/camera/${req.params.id}/stream/mjpeg`);
    
    // Set appropriate headers for MJPEG streaming
    res.writeHead(200, {
      'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
      'Cache-Control': 'no-cache',
      'Connection': 'close',
      'Pragma': 'no-cache'
    });
    
    // Get the raw request from vision API
    const response = await axios({
      method: 'GET',
      url: `${VISION_API_URL}/api/camera/${req.params.id}/stream/mjpeg`,
      responseType: 'stream',
      timeout: 30000  // Longer timeout for streams
    });
    
    // Pipe the stream directly to the client
    response.data.pipe(res);
    
    // Handle client disconnect
    req.on('close', () => {
      if (response.data) {
        response.data.destroy();
      }
    });
    
  } catch (error) {
    console.error(`Error streaming MJPEG: ${error.message}`);
    if (!res.headersSent) {
      res.status(500).send(`Error streaming MJPEG: ${error.message}`);
    }
  }
});

// Proxy API requests to vision API (except MJPEG streams which are handled separately)
app.use('/api', async (req, res) => {
  // Skip MJPEG stream requests as they're handled by a dedicated endpoint
  if (req.url.includes('/camera/') && req.url.includes('/stream/mjpeg')) {
    return;
  }
  
  try {
    const visionUrl = `${VISION_API_URL}/api${req.url}`;
    console.log(`Proxying request to: ${visionUrl}`);
    
    const method = req.method.toLowerCase();
    let response;
    
    if (method === 'get') {
      response = await apiClient.get(req.url);
    } else if (method === 'post') {
      response = await apiClient.post(req.url, req.body);
    } else if (method === 'put') {
      response = await apiClient.put(req.url, req.body);
    } else if (method === 'delete') {
      response = await apiClient.delete(req.url);
    } else {
      return res.status(405).json({ error: 'Method not allowed' });
    }
    
    // Forward the response
    res.status(response.status).json(response.data);
    
    // For camera stream - send to socket.io clients
    if (req.url.includes('/camera/') && req.url.includes('/stream') && response.data && response.data.frame) {
      io.emit('camera_frame', response.data);
    }
    
    // For other result types, emit appropriate events
    if (req.url.includes('/calibrate') && method === 'post') {
      io.emit('calibration_result', response.data);
    } else if (req.url.includes('/detect') && method === 'post') {
      io.emit('detection_result', response.data);
    } else if (req.url.includes('/pnp') && method === 'post') {
      io.emit('pnp_result', response.data);
    }
    
  } catch (error) {
    console.error(`Error proxying to ${req.url}: ${error.message}`);
    if (error.response) {
      // Forward the error status and data from the vision API
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ 
        error: error.message || 'Internal server error',
        url: req.url,
        method: req.method
      });
    }
  }
});

// Add a health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Socket.io
io.on('connection', (socket) => {
  console.log('Client connected');
  
  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

// Start server
const PORT = process.env.PORT || 9029;
server.listen(PORT, () => {
  console.log(`API Server running on port ${PORT}`);
  console.log(`Using vision API at: ${VISION_API_URL}`);
}); 