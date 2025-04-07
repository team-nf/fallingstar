import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  TextField, 
  FormControl, 
  Select, 
  MenuItem,
  Grid,
  IconButton,
  Tooltip,
  Button
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import FileDownloadIcon from '@mui/icons-material/FileDownload';

const MegaTagPanel = () => {
  const [llForward, setLlForward] = useState('0.5');
  const [llRight, setLlRight] = useState('-0.1');
  const [llUp, setLlUp] = useState('0.25');
  const [llRoll, setLlRoll] = useState('0');
  const [llPitch, setLlPitch] = useState('0');
  const [llYaw, setLlYaw] = useState('0');
  const [snapToFloor, setSnapToFloor] = useState('Yes');
  
  return (
    <Box sx={{ p: 1 }}>
      {/* MegaTag Field-Space Localization Setup */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box 
            sx={{ 
              width: 30, 
              borderBottom: '2px solid #666',
              mr: 2
            }}
          />
          <Typography variant="subtitle1">
            MegaTag Field-Space Localization Setup
          </Typography>
          <Tooltip title="Configure the Field-Space localization parameters">
            <IconButton size="small" sx={{ ml: 1, color: 'text.secondary' }}>
              <InfoOutlinedIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Box 
            sx={{ 
              flexGrow: 1,
              borderBottom: '2px solid #666',
              ml: 2
            }}
          />
        </Box>
        
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                LL Forward
              </Typography>
              <TextField
                value={llForward}
                onChange={(e) => setLlForward(e.target.value)}
                InputProps={{
                  sx: { 
                    height: 35,
                    backgroundColor: 'background.paper'
                  }
                }}
                size="small"
              />
            </Box>
          </Grid>
          
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                LL Right
              </Typography>
              <TextField
                value={llRight}
                onChange={(e) => setLlRight(e.target.value)}
                InputProps={{
                  sx: { 
                    height: 35,
                    backgroundColor: 'background.paper'
                  }
                }}
                size="small"
              />
            </Box>
          </Grid>
          
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                LL Up
              </Typography>
              <TextField
                value={llUp}
                onChange={(e) => setLlUp(e.target.value)}
                InputProps={{
                  sx: { 
                    height: 35,
                    backgroundColor: 'background.paper'
                  }
                }}
                size="small"
              />
            </Box>
          </Grid>
          
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                LL Roll
              </Typography>
              <TextField
                value={llRoll}
                onChange={(e) => setLlRoll(e.target.value)}
                InputProps={{
                  sx: { 
                    height: 35,
                    backgroundColor: 'background.paper'
                  }
                }}
                size="small"
              />
            </Box>
          </Grid>
          
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                LL Pitch
              </Typography>
              <TextField
                value={llPitch}
                onChange={(e) => setLlPitch(e.target.value)}
                InputProps={{
                  sx: { 
                    height: 35,
                    backgroundColor: 'background.paper'
                  }
                }}
                size="small"
              />
            </Box>
          </Grid>
          
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                LL Yaw
              </Typography>
              <TextField
                value={llYaw}
                onChange={(e) => setLlYaw(e.target.value)}
                InputProps={{
                  sx: { 
                    height: 35,
                    backgroundColor: 'background.paper'
                  }
                }}
                size="small"
              />
            </Box>
          </Grid>
        </Grid>
        
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Snap Robot to Floor
              </Typography>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <Select
                  value={snapToFloor}
                  onChange={(e) => setSnapToFloor(e.target.value)}
                  sx={{ 
                    height: 35,
                    backgroundColor: 'background.paper'
                  }}
                >
                  <MenuItem value="Yes">Yes</MenuItem>
                  <MenuItem value="No">No</MenuItem>
                </Select>
              </FormControl>
            </Box>
          </Grid>
          
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Field Map File:
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Box sx={{ 
                  color: '#00ff00', 
                  mr: 1, 
                  display: 'flex', 
                  alignItems: 'center'
                }}>
                  ✓
                </Box>
                <Button
                  variant="outlined"
                  size="small"
                  sx={{ minWidth: 'auto', p: 1 }}
                >
                  <FileDownloadIcon fontSize="small" />
                </Button>
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default MegaTagPanel; 