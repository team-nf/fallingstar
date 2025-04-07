import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  TextField, 
  FormControlLabel, 
  Switch,
  Grid,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider
} from '@mui/material';
import { useTheme } from '../context/ThemeContext';

const Home = () => {
  const { darkMode, toggleTheme } = useTheme();
  const [teamNumber, setTeamNumber] = useState(localStorage.getItem('teamNumber') || '');
  const [saved, setSaved] = useState(false);

  const handleTeamNumberChange = (e) => {
    setTeamNumber(e.target.value);
    setSaved(false);
  };

  const saveSettings = () => {
    localStorage.setItem('teamNumber', teamNumber);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  useEffect(() => {
    document.title = 'Home - FallingStar';
  }, []);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Welcome to FallingStar
      </Typography>
      <Typography variant="subtitle1" sx={{ mb: 4 }}>
        An open-source computer vision platform for robotics
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardHeader title="Team Settings" />
            <Divider />
            <CardContent>
              <TextField
                fullWidth
                label="Team Number"
                variant="outlined"
                value={teamNumber}
                onChange={handleTeamNumberChange}
                margin="normal"
                type="number"
                helperText="Your FIRST Robotics Competition team number"
              />
              <Box sx={{ mt: 2 }}>
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={saveSettings}
                  disabled={saved}
                >
                  {saved ? "Saved!" : "Save Settings"}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardHeader title="Appearance Settings" />
            <Divider />
            <CardContent>
              <FormControlLabel
                control={
                  <Switch
                    checked={darkMode}
                    onChange={toggleTheme}
                    color="primary"
                  />
                }
                label={darkMode ? "Dark Mode" : "Light Mode"}
              />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                {darkMode 
                  ? "Using black and dark green theme" 
                  : "Using smoky gray and dark navy blue theme"}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card elevation={3}>
            <CardHeader title="About FallingStar" />
            <Divider />
            <CardContent>
              <Typography variant="body1" paragraph>
                FallingStar is an open-source alternative to vision processing systems like Limelight, designed for robotics competitions and computer vision applications.
              </Typography>
              <Typography variant="body1">
                Navigate through the tabs to set up your cameras, calibrate them, detect objects, track targets, and more.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Home; 