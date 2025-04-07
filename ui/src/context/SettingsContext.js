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
  const [loading, setLoading] = useState(true);
  const [settings, setSettings] = useState(defaultSettings);

  // Load settings from vision API on component mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        // Try to load from vision API first
        const response = await fetch('http://localhost:9029/api/settings');
        if (response.ok) {
          const data = await response.json();
          setSettings(data);
        } else {
          // If API fails, try localStorage
          const savedSettings = localStorage.getItem('fallingstarSettings');
          if (savedSettings) {
            setSettings(JSON.parse(savedSettings));
          }
        }
      } catch (error) {
        console.error('Error loading settings:', error);
        // If all fails, use defaults and try localStorage
        const savedSettings = localStorage.getItem('fallingstarSettings');
        if (savedSettings) {
          setSettings(JSON.parse(savedSettings));
        }
      } finally {
        setLoading(false);
      }
    };

    loadSettings();
  }, []);

  // Save settings whenever they change
  useEffect(() => {
    if (loading) return; // Skip initial load
    
    // Save to localStorage as a backup
    localStorage.setItem('fallingstarSettings', JSON.stringify(settings));
    
    // Save to vision API
    fetch('http://localhost:9029/api/settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    }).catch(error => {
      console.error('Error saving settings to vision API:', error);
    });
  }, [settings, loading]);

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

  if (loading) {
    // You could return a loading indicator here if needed
    return null;
  }

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, resetSettings }}>
      {children}
    </SettingsContext.Provider>
  );
};

export const useSettings = () => useContext(SettingsContext);

export default SettingsContext; 