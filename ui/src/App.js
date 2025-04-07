import React, { useState, useEffect } from 'react';
import { 
  Box, 
  CssBaseline, 
  ThemeProvider, 
  createTheme,
  AppBar,
  Tabs,
  Tab,
  Typography,
  Paper,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Slider,
  Switch,
  FormControlLabel,
  IconButton,
  Tooltip
} from '@mui/material';
import { io } from 'socket.io-client';
import './App.css';
import CameraStream from './components/CameraStream';
import TargetingPanel from './components/TargetingPanel';
import MegaTagPanel from './components/MegaTagPanel';
import SettingsIcon from '@mui/icons-material/Settings';
import TuneIcon from '@mui/icons-material/Tune';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import GpsFixedIcon from '@mui/icons-material/GpsFixed';

// Create a dark theme with green accents (like the Limelight UI)
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff00',
    },
    secondary: {
      main: '#ff0000',
    },
    background: {
      default: '#1e1e1e',
      paper: '#2d2d2d',
    },
  },
  components: {
    MuiTextField: {
      defaultProps: {
        size: 'small',
        variant: 'outlined',
      },
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: '#444',
            },
            '&:hover fieldset': {
              borderColor: '#666',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#00ff00',
            },
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          minHeight: 48,
          textTransform: 'none',
          '&.Mui-selected': {
            color: '#00ff00',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [streamTabValue, setStreamTabValue] = useState(0);
  const [socketConnected, setSocketConnected] = useState(false);
  const [streamData, setStreamData] = useState(null);
  
  useEffect(() => {
    // Connect to websocket
    const socket = io(window.location.origin);
    
    socket.on('connect', () => {
      console.log('Socket connected');
      setSocketConnected(true);
    });
    
    socket.on('disconnect', () => {
      console.log('Socket disconnected');
      setSocketConnected(false);
    });
    
    socket.on('stream_data', (data) => {
      setStreamData(data);
    });
    
    return () => {
      socket.disconnect();
    };
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleStreamTabChange = (event, newValue) => {
    setStreamTabValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
        <AppBar 
          position="static" 
          sx={{ 
            bgcolor: 'background.paper', 
            boxShadow: 1,
            borderBottom: '1px solid rgba(255, 255, 255, 0.12)'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', px: 2, py: 1 }}>
            <Typography variant="h6" component="div" sx={{ color: '#00ff00', fontWeight: 'bold' }}>
              FallingStar Vision
            </Typography>
            <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box 
                  sx={{ 
                    width: 8, 
                    height: 8, 
                    borderRadius: '50%', 
                    bgcolor: socketConnected ? '#00ff00' : '#ff0000'
                  }} 
                />
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  {socketConnected ? 'Connected' : 'Disconnected'}
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ color: '#00ff00', fontFamily: 'Roboto Mono' }}>
                {streamData?.fps ? `${streamData.fps} FPS` : '0 FPS'}
              </Typography>
            </Box>
          </Box>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            sx={{ 
              borderBottom: 1, 
              borderColor: 'divider',
              '& .MuiTabs-indicator': {
                backgroundColor: '#00ff00',
              }
            }}
          >
            <Tab icon={<CameraAltIcon />} label="Input" />
            <Tab icon={<TuneIcon />} label="Standard" />
            <Tab icon={<GpsFixedIcon />} label="Advanced" />
            <Tab icon={<SettingsIcon />} label="Output & Crosshair" />
          </Tabs>
        </AppBar>
        
        <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Main content area */}
          <Box sx={{ flex: 1, p: 2, overflow: 'auto', backgroundColor: 'background.default' }}>
            {/* Tab content */}
            <TabPanel value={tabValue} index={0}>
              <TargetingPanel />
            </TabPanel>
            <TabPanel value={tabValue} index={1}>
              <Typography>Standard settings will go here</Typography>
            </TabPanel>
            <TabPanel value={tabValue} index={2}>
              <MegaTagPanel />
            </TabPanel>
            <TabPanel value={tabValue} index={3}>
              <Typography>Output & Crosshair settings will go here</Typography>
            </TabPanel>
          </Box>
          
          {/* Camera stream panel */}
          <Paper 
            sx={{ 
              width: '40%', 
              p: 2, 
              display: 'flex', 
              flexDirection: 'column',
              bgcolor: 'background.default',
              borderLeft: 1,
              borderColor: 'divider'
            }}
          >
            <Tabs 
              value={streamTabValue} 
              onChange={handleStreamTabChange}
              sx={{ 
                mb: 2,
                '& .MuiTabs-indicator': {
                  backgroundColor: '#00ff00',
                }
              }}
            >
              <Tab label="Stream" />
            </Tabs>
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <CameraStream />
              <Typography variant="caption" sx={{ mt: 2, textAlign: 'center', color: 'text.secondary' }}>
                http://10.47.79.11:5800
              </Typography>
            </Box>
          </Paper>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 1 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default App; 