import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from './context/ThemeContext';
import { SettingsProvider } from './context/SettingsContext';
import { darkTheme, lightTheme } from './theme';
import { useTheme } from './context/ThemeContext';
import Layout from './components/Layout';
import Home from './pages/Home';
import Input from './pages/Input';
import Calibration from './pages/Calibration';
import Detection from './pages/Detection';
import Tracking from './pages/Tracking';
import Selection from './pages/Selection';
import PNP from './pages/PNP';

const ThemedApp = () => {
  const { darkMode } = useTheme();
  const theme = darkMode ? darkTheme : lightTheme;

  return (
    <MuiThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/input" element={<Input />} />
            <Route path="/calibration" element={<Calibration />} />
            <Route path="/detection" element={<Detection />} />
            <Route path="/tracking" element={<Tracking />} />
            <Route path="/selection" element={<Selection />} />
            <Route path="/pnp" element={<PNP />} />
          </Routes>
        </Layout>
      </Router>
    </MuiThemeProvider>
  );
};

const App = () => {
  return (
    <ThemeProvider>
      <SettingsProvider>
        <ThemedApp />
      </SettingsProvider>
    </ThemeProvider>
  );
};

export default App; 