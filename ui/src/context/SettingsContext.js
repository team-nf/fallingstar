import React, { createContext, useState, useContext, useEffect } from 'react';

// Default settings
const defaultSettings = {
  camera: {
    selectedCamera: '',
    brightness: 50,
    contrast: 50,
    exposure: 50
  },
  calibration: {
    chessboardSize: { width: 9, height: 6 },
    squareSize: 25
  },
  detection: {
    selectedModel: '',
    confidenceThreshold: 0.5
  },
  tracking: {
    selectedAlgorithm: '',
    trailDuration: 3
  },
  selection: {
    selectedAlgorithm: '',
  },
  pnp: {
    selectedTarget: '',
    customDimensions: { width: 0.2, height: 0.2, depth: 0.2 },
    fieldPosition: { x: 0, y: 0 }
  }
};

const SettingsContext = createContext();

export const SettingsProvider = ({ children }) => {
  // Try to load settings from localStorage, or use defaults
  const [settings, setSettings] = useState(() => {
    const savedSettings = localStorage.getItem('fallingstarSettings');
    return savedSettings ? JSON.parse(savedSettings) : defaultSettings;
  });

  // Save settings to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('fallingstarSettings', JSON.stringify(settings));
    
    // Also send settings to the server for Python integration
    fetch('http://localhost:9029/api/settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    }).catch(error => {
      console.error('Error saving settings to server:', error);
    });
  }, [settings]);

  // Update specific settings section
  const updateSettings = (section, updates) => {
    setSettings(prevSettings => ({
      ...prevSettings,
      [section]: {
        ...prevSettings[section],
        ...updates
      }
    }));
  };

  // Reset settings to defaults
  const resetSettings = () => {
    setSettings(defaultSettings);
  };

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, resetSettings }}>
      {children}
    </SettingsContext.Provider>
  );
};

export const useSettings = () => useContext(SettingsContext);

export default SettingsContext; 