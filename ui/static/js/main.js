/**
 * Vision Processing UI - Main JavaScript
 * This script handles UI interactions, Socket.IO communication, and configuration updates.
 */

// Global variables
let socket;
let configData = {};
let uiSettings = {};
let processingStatus = "stopped";
let lastFrameTime = 0;
let fpsUpdateInterval;
let selectedTab = "uiSettings";

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Socket.IO connection
    initSocketIO();
    
    // Initialize UI event listeners
    initUIControls();
    
    // Load initial data
    loadConfigData();
    loadUISettings();
    
    // Start FPS counter
    startFPSCounter();
    
    // Update status display
    updateStatusDisplay();
});

// Initialize Socket.IO connection
function initSocketIO() {
    try {
        // Connect to the Socket.IO server (same host as the web page)
        socket = io();
        
        // Connection event handlers
        socket.on('connect', function() {
            console.log('Connected to server');
            updateStatusDisplay();
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            updateStatusDisplay();
        });
        
        // Frame update event handler
        socket.on('frame_update', function(data) {
            updateFrameDisplay(data);
        });
        
        // Get initial frame
        socket.emit('get_frame', function(data) {
            if (data && data.frame) {
                updateFrameDisplay(data);
            }
        });
    } catch (error) {
        console.error('Error initializing Socket.IO:', error);
    }
}

// Initialize UI controls and event listeners
function initUIControls() {
    // Buttons
    document.getElementById('startBtn').addEventListener('click', startProcessing);
    document.getElementById('stopBtn').addEventListener('click', stopProcessing);
    document.getElementById('settingsBtn').addEventListener('click', openSettingsModal);
    document.getElementById('closeSettingsBtn').addEventListener('click', closeSettingsModal);
    document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
    document.getElementById('resetSettingsBtn').addEventListener('click', resetSettings);
    document.getElementById('snapshotBtn').addEventListener('click', takeSnapshot);
    
    // View mode selector
    document.getElementById('viewMode').addEventListener('change', function() {
        // Update view mode (handled by server)
        updateUISetting('view_mode', this.value);
    });
    
    // Tab switching
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            switchTab(this.dataset.tab);
        });
    });
    
    // UI Settings controls
    document.getElementById('theme').addEventListener('change', function() {
        updateUISetting('theme', this.value);
        applyTheme(this.value);
    });
    
    document.getElementById('showProcessingSteps').addEventListener('change', function() {
        updateUISetting('show_processing_steps', this.checked);
    });
    
    document.getElementById('showDetectionBoxes').addEventListener('change', function() {
        updateUISetting('show_detection_boxes', this.checked);
    });
    
    document.getElementById('showTrackingInfo').addEventListener('change', function() {
        updateUISetting('show_tracking_info', this.checked);
    });
    
    document.getElementById('showSelectionInfo').addEventListener('change', function() {
        updateUISetting('show_selection_info', this.checked);
    });
    
    document.getElementById('streamResolution').addEventListener('change', function() {
        updateUISetting('stream_resolution', this.value);
    });
}

// Load configuration data from the server
function loadConfigData() {
    fetch('/config')
        .then(response => response.json())
        .then(data => {
            configData = data;
            
            // Populate settings forms
            populateCameraSettings();
            populateDetectionSettings();
            populateTrackingSettings();
            populateSelectionSettings();
        })
        .catch(error => {
            console.error('Error loading configuration data:', error);
        });
}

// Load UI settings from the server
function loadUISettings() {
    fetch('/ui_settings')
        .then(response => response.json())
        .then(data => {
            uiSettings = data;
            
            // Apply theme
            applyTheme(uiSettings.theme);
        })
        .catch(error => {
            console.error('Error loading UI settings:', error);
        });
}

// Apply theme to the document
function applyTheme(theme) {
    document.body.className = `theme-${theme}`;
}

// Update a UI setting
function updateUISetting(key, value) {
    const data = {};
    data[key] = value;
    
    fetch('/ui_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            uiSettings = data.settings;
        }
    })
    .catch(error => {
        console.error('Error updating UI setting:', error);
    });
}

// Switch between settings tabs
function switchTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.style.display = 'none';
    });
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content and mark button as active
    document.getElementById(tabName).style.display = 'block';
    document.querySelector(`.tab-btn[data-tab="${tabName}"]`).classList.add('active');
    
    // Store selected tab
    selectedTab = tabName;
}

// Populate camera settings form
function populateCameraSettings() {
    if (!configData.camera) return;
    
    const container = document.getElementById('cameraSettingsForm');
    container.innerHTML = '';
    
    // Create form elements for each camera setting
    for (const [key, value] of Object.entries(configData.camera)) {
        if (typeof value === 'object') continue; // Skip nested objects for simplicity
        
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = formatLabel(key);
        label.htmlFor = `camera_${key}`;
        
        let input;
        
        if (typeof value === 'boolean') {
            // Create toggle switch for boolean values
            const switchLabel = document.createElement('label');
            switchLabel.className = 'switch';
            
            input = document.createElement('input');
            input.type = 'checkbox';
            input.id = `camera_${key}`;
            input.checked = value;
            
            const slider = document.createElement('span');
            slider.className = 'slider';
            
            switchLabel.appendChild(input);
            switchLabel.appendChild(slider);
            
            formGroup.appendChild(label);
            formGroup.appendChild(switchLabel);
        } else if (typeof value === 'number') {
            // Create number input for numeric values
            input = document.createElement('input');
            input.type = 'number';
            input.id = `camera_${key}`;
            input.value = value;
            input.className = 'text-input';
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
        } else {
            // Create text input for other values
            input = document.createElement('input');
            input.type = 'text';
            input.id = `camera_${key}`;
            input.value = value;
            input.className = 'text-input';
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
        }
        
        // Add event listener to update config on change
        input.addEventListener('change', function() {
            updateConfigValue('camera', key, this.type === 'checkbox' ? this.checked : this.value);
        });
        
        container.appendChild(formGroup);
    }
}

// Populate detection settings form
function populateDetectionSettings() {
    if (!configData.detection) return;
    
    const container = document.getElementById('detectionSettingsForm');
    container.innerHTML = '';
    
    // Create form elements for each detection setting
    for (const [key, value] of Object.entries(configData.detection)) {
        if (key === 'input_size') continue; // Skip array for simplicity
        
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = formatLabel(key);
        label.htmlFor = `detection_${key}`;
        
        let input;
        
        if (typeof value === 'boolean') {
            // Create toggle switch for boolean values
            const switchLabel = document.createElement('label');
            switchLabel.className = 'switch';
            
            input = document.createElement('input');
            input.type = 'checkbox';
            input.id = `detection_${key}`;
            input.checked = value;
            
            const slider = document.createElement('span');
            slider.className = 'slider';
            
            switchLabel.appendChild(input);
            switchLabel.appendChild(slider);
            
            formGroup.appendChild(label);
            formGroup.appendChild(switchLabel);
        } else if (typeof value === 'number') {
            // Create number input for numeric values
            input = document.createElement('input');
            input.type = 'number';
            input.id = `detection_${key}`;
            input.value = value;
            input.step = key === 'threshold' ? '0.1' : '1';
            input.className = 'text-input';
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
        } else {
            // Create text input for other values
            input = document.createElement('input');
            input.type = 'text';
            input.id = `detection_${key}`;
            input.value = value;
            input.className = 'text-input';
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
        }
        
        // Add event listener to update config on change
        input.addEventListener('change', function() {
            updateConfigValue('detection', key, this.type === 'checkbox' ? this.checked : this.value);
        });
        
        container.appendChild(formGroup);
    }
}

// Populate tracking settings form
function populateTrackingSettings() {
    if (!configData.tracking) return;
    
    const container = document.getElementById('trackingSettingsForm');
    container.innerHTML = '';
    
    // Algorithm selector
    const algorithmGroup = document.createElement('div');
    algorithmGroup.className = 'form-group';
    
    const algorithmLabel = document.createElement('label');
    algorithmLabel.textContent = 'Algorithm';
    algorithmLabel.htmlFor = 'tracking_algorithm';
    
    const algorithmSelect = document.createElement('select');
    algorithmSelect.id = 'tracking_algorithm';
    algorithmSelect.className = 'select-input';
    
    const algorithms = ['sort', 'kalman', 'iou', 'opencv'];
    algorithms.forEach(algorithm => {
        const option = document.createElement('option');
        option.value = algorithm;
        option.textContent = algorithm.charAt(0).toUpperCase() + algorithm.slice(1);
        option.selected = configData.tracking.algorithm === algorithm;
        algorithmSelect.appendChild(option);
    });
    
    algorithmSelect.addEventListener('change', function() {
        updateConfigValue('tracking', 'algorithm', this.value);
    });
    
    algorithmGroup.appendChild(algorithmLabel);
    algorithmGroup.appendChild(algorithmSelect);
    container.appendChild(algorithmGroup);
    
    // Other tracking settings
    for (const [key, value] of Object.entries(configData.tracking)) {
        if (key === 'algorithm') continue; // Skip algorithm since we already handled it
        
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = formatLabel(key);
        label.htmlFor = `tracking_${key}`;
        
        let input;
        
        if (typeof value === 'boolean') {
            // Create toggle switch for boolean values
            const switchLabel = document.createElement('label');
            switchLabel.className = 'switch';
            
            input = document.createElement('input');
            input.type = 'checkbox';
            input.id = `tracking_${key}`;
            input.checked = value;
            
            const slider = document.createElement('span');
            slider.className = 'slider';
            
            switchLabel.appendChild(input);
            switchLabel.appendChild(slider);
            
            formGroup.appendChild(label);
            formGroup.appendChild(switchLabel);
        } else if (typeof value === 'number') {
            // Create number input for numeric values
            input = document.createElement('input');
            input.type = 'number';
            input.id = `tracking_${key}`;
            input.value = value;
            input.step = key === 'iou_threshold' ? '0.1' : '1';
            input.className = 'text-input';
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
        } else {
            // Create text input for other values
            input = document.createElement('input');
            input.type = 'text';
            input.id = `tracking_${key}`;
            input.value = value;
            input.className = 'text-input';
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
        }
        
        // Add event listener to update config on change
        input.addEventListener('change', function() {
            let value = this.type === 'checkbox' ? this.checked : this.value;
            if (this.type === 'number') value = parseFloat(value);
            updateConfigValue('tracking', key, value);
        });
        
        container.appendChild(formGroup);
    }
}

// Populate selection settings form
function populateSelectionSettings() {
    if (!configData.selection) return;
    
    const container = document.getElementById('selectionSettingsForm');
    container.innerHTML = '';
    
    // Algorithm selector
    const algorithmGroup = document.createElement('div');
    algorithmGroup.className = 'form-group';
    
    const algorithmLabel = document.createElement('label');
    algorithmLabel.textContent = 'Algorithm';
    algorithmLabel.htmlFor = 'selection_algorithm';
    
    const algorithmSelect = document.createElement('select');
    algorithmSelect.id = 'selection_algorithm';
    algorithmSelect.className = 'select-input';
    
    const algorithms = ['lowest', 'closest', 'largest', 'highest_confidence'];
    algorithms.forEach(algorithm => {
        const option = document.createElement('option');
        option.value = algorithm;
        option.textContent = formatLabel(algorithm);
        option.selected = configData.selection.algorithm === algorithm;
        algorithmSelect.appendChild(option);
    });
    
    algorithmSelect.addEventListener('change', function() {
        updateConfigValue('selection', 'algorithm', this.value);
    });
    
    algorithmGroup.appendChild(algorithmLabel);
    algorithmGroup.appendChild(algorithmSelect);
    container.appendChild(algorithmGroup);
    
    // Other selection settings
    for (const [key, value] of Object.entries(configData.selection)) {
        if (key === 'algorithm') continue; // Skip algorithm since we already handled it
        
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = formatLabel(key);
        label.htmlFor = `selection_${key}`;
        
        let input;
        
        if (typeof value === 'boolean') {
            // Create toggle switch for boolean values
            const switchLabel = document.createElement('label');
            switchLabel.className = 'switch';
            
            input = document.createElement('input');
            input.type = 'checkbox';
            input.id = `selection_${key}`;
            input.checked = value;
            
            const slider = document.createElement('span');
            slider.className = 'slider';
            
            switchLabel.appendChild(input);
            switchLabel.appendChild(slider);
            
            formGroup.appendChild(label);
            formGroup.appendChild(switchLabel);
        } else if (typeof value === 'number') {
            // Create number input for numeric values
            input = document.createElement('input');
            input.type = 'number';
            input.id = `selection_${key}`;
            input.value = value;
            input.step = key === 'min_confidence' ? '0.1' : '1';
            input.className = 'text-input';
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
        } else {
            // Create text input for other values
            input = document.createElement('input');
            input.type = 'text';
            input.id = `selection_${key}`;
            input.value = value;
            input.className = 'text-input';
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
        }
        
        // Add event listener to update config on change
        input.addEventListener('change', function() {
            let value = this.type === 'checkbox' ? this.checked : this.value;
            if (this.type === 'number') value = parseFloat(value);
            updateConfigValue('selection', key, value);
        });
        
        container.appendChild(formGroup);
    }
}

// Format a camelCase or snake_case label for display
function formatLabel(str) {
    return str
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, function(str) { return str.toUpperCase(); });
}

// Update a configuration value
function updateConfigValue(section, key, value) {
    const data = {};
    data['value'] = value;
    
    fetch(`/config/${section}/${key}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`Updated ${section}.${key} to ${value}`);
            
            // Update local config data
            if (!configData[section]) configData[section] = {};
            configData[section][key] = value;
        } else {
            console.error(`Failed to update ${section}.${key}:`, data.error);
        }
    })
    .catch(error => {
        console.error(`Error updating ${section}.${key}:`, error);
    });
}

// Update the frame display with new image data
function updateFrameDisplay(data) {
    if (!data || !data.frame) return;
    
    // Calculate FPS
    const now = performance.now();
    const elapsed = now - lastFrameTime;
    lastFrameTime = now;
    
    const fps = 1000 / elapsed;
    document.getElementById('fpsValue').textContent = fps.toFixed(1);
    
    // Get camera feed element
    const cameraFeed = document.getElementById('cameraFeed');
    
    // Only update if using Socket.IO mode (not HTTP streaming)
    if (cameraFeed.src.indexOf('video_feed') === -1) {
        cameraFeed.src = `data:image/jpeg;base64,${data.frame}`;
    }
}

// Start the FPS counter
function startFPSCounter() {
    // Clear existing interval if any
    if (fpsUpdateInterval) {
        clearInterval(fpsUpdateInterval);
    }
    
    // Update FPS display every second if not receiving frames
    fpsUpdateInterval = setInterval(function() {
        const now = performance.now();
        if (now - lastFrameTime > 1000) {
            document.getElementById('fpsValue').textContent = '0.0';
        }
    }, 1000);
}

// Open the settings modal
function openSettingsModal() {
    const modal = document.getElementById('settingsModal');
    modal.classList.add('visible');
}

// Close the settings modal
function closeSettingsModal() {
    const modal = document.getElementById('settingsModal');
    modal.classList.remove('visible');
}

// Save current settings
function saveSettings() {
    // Reload config data to verify settings were saved
    loadConfigData();
    
    // Close the modal
    closeSettingsModal();
}

// Reset settings to defaults
function resetSettings() {
    if (!confirm('Reset all settings to defaults?')) return;
    
    // Implementation would depend on server API
    
    // Reload config data
    loadConfigData();
}

// Start vision processing
function startProcessing() {
    // Implementation would depend on server API
    processingStatus = "running";
    updateStatusDisplay();
}

// Stop vision processing
function stopProcessing() {
    // Implementation would depend on server API
    processingStatus = "stopped";
    updateStatusDisplay();
}

// Take a snapshot of the current frame
function takeSnapshot() {
    // Get the current frame
    const cameraFeed = document.getElementById('cameraFeed');
    
    // Create a download link
    const link = document.createElement('a');
    link.download = `snapshot_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
    
    // If using direct data URL (Socket.IO)
    if (cameraFeed.src.startsWith('data:image/jpeg;base64,')) {
        link.href = cameraFeed.src;
        link.click();
    } else {
        // If using HTTP streaming, capture the frame
        const canvas = document.createElement('canvas');
        canvas.width = cameraFeed.naturalWidth;
        canvas.height = cameraFeed.naturalHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraFeed, 0, 0);
        
        link.href = canvas.toDataURL('image/jpeg');
        link.click();
    }
}

// Update the status display
function updateStatusDisplay() {
    const statusElement = document.getElementById('processingStatus');
    statusElement.textContent = processingStatus.charAt(0).toUpperCase() + processingStatus.slice(1);
    
    // Set status color
    if (processingStatus === 'running') {
        statusElement.style.color = 'var(--success)';
    } else if (processingStatus === 'stopped') {
        statusElement.style.color = 'var(--danger)';
    } else {
        statusElement.style.color = 'inherit';
    }
} 