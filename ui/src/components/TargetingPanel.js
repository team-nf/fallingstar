import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  TextField, 
  FormControl, 
  Select, 
  MenuItem,
  FormHelperText,
  Grid,
  Paper,
  IconButton,
  Tooltip
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

const TargetingPanel = () => {
  const [targeting3d, setTargeting3d] = useState('Yes');
  const [offsetForward, setOffsetForward] = useState('0');
  const [offsetRight, setOffsetRight] = useState('0');
  const [offsetUp, setOffsetUp] = useState('0');
  
  return (
    <Box sx={{ p: 1 }}>
      {/* 3D Targeting Section */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box 
            sx={{ 
              width: 30, 
              borderBottom: '2px solid #666',
              mr: 2
            }}
          />
          <Typography variant="subtitle1">Full 3D Targeting</Typography>
          <Box 
            sx={{ 
              flexGrow: 1,
              borderBottom: '2px solid #666',
              ml: 2
            }}
          />
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="body2" sx={{ width: 120 }}>
            Full 3D Targeting
          </Typography>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <Select
              value={targeting3d}
              onChange={(e) => setTargeting3d(e.target.value)}
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
      </Box>
      
      {/* 3D Point-of-Interest Offset */}
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
            3D Point-of-Interest Offset
          </Typography>
          <Tooltip title="Configure the 3D offset for your robot's point of interest">
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
        
        <Grid container spacing={4} sx={{ mt: 1 }}>
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Forward (m)
              </Typography>
              <TextField
                value={offsetForward}
                onChange={(e) => setOffsetForward(e.target.value)}
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
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Right (m)
              </Typography>
              <TextField
                value={offsetRight}
                onChange={(e) => setOffsetRight(e.target.value)}
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
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Up (m)
              </Typography>
              <TextField
                value={offsetUp}
                onChange={(e) => setOffsetUp(e.target.value)}
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
      </Box>
    </Box>
  );
};

export default TargetingPanel; 