import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, 
  Box, 
  Drawer, 
  IconButton, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Toolbar, 
  Typography,
  useMediaQuery,
  useTheme as useMuiTheme
} from '@mui/material';
import { 
  Menu as MenuIcon,
  Home as HomeIcon,
  VideoSettings as InputIcon,
  CameraEnhance as CalibrationIcon,
  ViewInAr as DetectionIcon,
  Timeline as TrackingIcon,
  RadioButtonChecked as SelectionIcon,
  ThreeDRotation as PNPIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon
} from '@mui/icons-material';
import { useTheme } from '../context/ThemeContext';

const drawerWidth = 240;

const menuItems = [
  { text: 'Home', icon: <HomeIcon />, path: '/' },
  { text: 'Input', icon: <InputIcon />, path: '/input' },
  { text: 'Calibration', icon: <CalibrationIcon />, path: '/calibration' },
  { text: 'Detection', icon: <DetectionIcon />, path: '/detection' },
  { text: 'Tracking', icon: <TrackingIcon />, path: '/tracking' },
  { text: 'Selection', icon: <SelectionIcon />, path: '/selection' },
  { text: 'PnP', icon: <PNPIcon />, path: '/pnp' }
];

const Layout = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { darkMode, toggleTheme } = useTheme();
  const muiTheme = useMuiTheme();
  const isSmallScreen = useMediaQuery(muiTheme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const drawer = (
    <>
      <Toolbar>
        <Typography variant="h6" noWrap>
          FallingStar
        </Typography>
      </Toolbar>
      <List>
        {menuItems.map((item) => (
          <ListItem 
            button 
            key={item.text} 
            onClick={() => {
              navigate(item.path);
              if (isSmallScreen) {
                setMobileOpen(false);
              }
            }}
            selected={location.pathname === item.path}
          >
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </>
  );

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {menuItems.find(item => item.path === location.pathname)?.text || 'FallingStar'}
          </Typography>
          <IconButton color="inherit" onClick={toggleTheme}>
            {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        {/* Mobile Drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        {/* Desktop Drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          height: '100%',
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default Layout; 