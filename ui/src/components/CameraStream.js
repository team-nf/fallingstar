import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Button, 
  Typography,
  IconButton,
  Tooltip,
  Paper
} from '@mui/material';
import { PhotoCamera, Fullscreen, FullscreenExit } from '@mui/icons-material';

const CameraStream = () => {
  const [streamUrl, setStreamUrl] = useState('');
  const [connected, setConnected] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const streamRef = useRef(null);
  const containerRef = useRef(null);
  
  useEffect(() => {
    // Get the stream URL from the server
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const baseUrl = window.location.host;
    
    // This is a placeholder - replace with actual stream URL
    setStreamUrl(`/api/stream`);
    
    // Simulate connection status
    setConnected(true);
    
    // Handle fullscreen changes
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);
  
  const takeSnapshot = () => {
    // Implement snapshot functionality
    console.log('Taking snapshot');
  };
  
  const toggleFullscreen = () => {
    if (!isFullscreen) {
      containerRef.current?.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };
  
  return (
    <Box 
      ref={containerRef}
      sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100%',
        position: 'relative',
        ...(isFullscreen && {
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          zIndex: 9999,
          backgroundColor: 'background.default',
          p: 2
        })
      }}
    >
      <Box 
        sx={{ 
          flex: 1, 
          border: '1px solid #444', 
          backgroundColor: '#000',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          overflow: 'hidden',
          borderRadius: 1
        }}
        ref={streamRef}
      >
        {connected ? (
          <img 
            src={`${window.location.origin}/api/stream?t=${Date.now()}`} 
            alt="Camera Stream"
            style={{ 
              maxWidth: '100%', 
              maxHeight: '100%',
              objectFit: 'contain'
            }}
          />
        ) : (
          <Typography variant="body2" color="text.secondary">
            Waiting for stream...
          </Typography>
        )}
        
        {/* Crosshair */}
        <Box 
          sx={{ 
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none',
            zIndex: 10
          }}
        >
          <Box sx={{ 
            width: 30, 
            height: 1, 
            backgroundColor: '#00ff00',
            position: 'absolute',
            top: 0,
            left: -15
          }} />
          <Box sx={{ 
            width: 1, 
            height: 30, 
            backgroundColor: '#00ff00',
            position: 'absolute',
            top: -15,
            left: 0
          }} />
        </Box>
        
        {/* Target boxes */}
        <Box 
          sx={{ 
            position: 'absolute', 
            border: '2px solid #00ff00',
            width: 100,
            height: 80,
            top: '60%',
            left: '60%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'flex-start',
            color: '#00ff00',
            pointerEvents: 'none'
          }}
        >
          <Paper sx={{ 
            backgroundColor: 'rgba(0,0,0,0.7)', 
            px: 0.5,
            color: '#00ff00',
            position: 'absolute',
            top: -20
          }}>
            <Typography variant="caption">
              5
            </Typography>
          </Paper>
        </Box>
        
        {/* Stream controls */}
        <Box 
          sx={{ 
            position: 'absolute',
            top: 8,
            right: 8,
            display: 'flex',
            gap: 1
          }}
        >
          <Tooltip title="Take Snapshot">
            <IconButton 
              size="small"
              onClick={takeSnapshot}
              sx={{ 
                bgcolor: 'rgba(0,0,0,0.5)',
                '&:hover': {
                  bgcolor: 'rgba(0,0,0,0.7)'
                }
              }}
            >
              <PhotoCamera fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
            <IconButton 
              size="small"
              onClick={toggleFullscreen}
              sx={{ 
                bgcolor: 'rgba(0,0,0,0.5)',
                '&:hover': {
                  bgcolor: 'rgba(0,0,0,0.7)'
                }
              }}
            >
              {isFullscreen ? (
                <FullscreenExit fontSize="small" />
              ) : (
                <Fullscreen fontSize="small" />
              )}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      {/* Stream metrics */}
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          mt: 2,
          px: 1
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="caption" sx={{ color: '#00ff00', fontFamily: 'Roboto Mono' }}>
            +3.50°
          </Typography>
          <Typography variant="caption" sx={{ color: '#00ff00', fontFamily: 'Roboto Mono' }}>
            +9.45°
          </Typography>
          <Typography variant="caption" sx={{ color: '#00ff00', fontFamily: 'Roboto Mono' }}>
            +0.296%
          </Typography>
          <Typography variant="caption" sx={{ color: '#00ff00', fontFamily: 'Roboto Mono' }}>
            +78.9
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default CameraStream; 