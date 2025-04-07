import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Add console logs to debug build issues
console.log('Starting React application...');

try {
  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  console.log('React application rendered successfully');
} catch (error) {
  console.error('Error rendering React application:', error);
} 