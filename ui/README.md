# FallingStar UI Component

This directory contains the UI code for FallingStar.

## Technology Stack

- React for the frontend
- Material-UI for component styling
- Express.js for the backend server
- Socket.IO for real-time updates

## Directory Structure

- `src/` - React source code
  - `components/` - Reusable UI components
  - `context/` - React context for state management
  - `pages/` - Main application pages
- `public/` - Static assets
- `server.js` - Express server that serves the UI and proxies API requests

## Features

The UI provides interfaces for:

- Team settings and UI theme configuration
- Camera selection and settings management
- Camera calibration with visualization
- Target detection with configurable parameters
- Real-time tracking visualization
- Target selection and filtering
- 3D pose estimation with visualization

## Development

To start the development server:

```bash
npm run dev
```

To build for production:

```bash
npm run build
```

## Server.js

The `server.js` file serves several purposes:

1. Serves the React application
2. Proxies requests to the vision API
3. Manages settings persistence
4. Provides Socket.IO for real-time updates

## Communication with Vision Component

The UI communicates with the vision component via REST API calls. The responses are served to the frontend and, when appropriate, also broadcasted via Socket.IO for real-time updates.

## Pages

- **Home**: Configure team settings and UI theme
- **Input**: Manage camera selection and settings
- **Calibration**: Calibrate cameras for accurate measurements
- **Detection**: Detect objects using ML models
- **Tracking**: Track objects across frames with trails
- **Selection**: Select specific targets using various algorithms
- **PnP**: Perspective-n-Point for 3D pose estimation 