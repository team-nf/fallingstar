#!/usr/bin/env python3
"""
NetworkTables integration for vision processing.

This module provides utilities for connecting to and communicating with
a RoboRIO or other NetworkTables server, primarily for FRC robotics.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union

# Try to import pynetworktables
try:
    import ntcore
    from ntcore import NetworkTableInstance, PubSubOptions
    HAS_NETWORKTABLES = True
except ImportError:
    try:
        from networktables import NetworkTables
        HAS_NETWORKTABLES = True
        LEGACY_NT = True
    except ImportError:
        HAS_NETWORKTABLES = False
        LEGACY_NT = False

# Configure logging
logger = logging.getLogger("networktables")

class NetworkTablesInterface:
    """Interface for communicating with NetworkTables."""
    
    def __init__(self, team_number: Optional[int] = None, server_address: Optional[str] = None,
                 table_name: str = "VisionTracking"):
        """
        Initialize NetworkTables connection.
        
        Args:
            team_number: FRC team number (for automatic server address)
            server_address: Manual server address (overrides team_number)
            table_name: Name of the NetworkTables table to use
        """
        self.connected = False
        self.table_name = table_name
        self.connection_listener = None
        self.table = None
        self.inst = None
        
        if not HAS_NETWORKTABLES:
            logger.warning("NetworkTables package not found. Running in simulation mode.")
            return
            
        # Initialize NetworkTables
        if LEGACY_NT:
            # Legacy NetworkTables API
            NetworkTables.initialize(server=server_address)
            if team_number and not server_address:
                server = f"roborio-{team_number}-frc.local"
                NetworkTables.initialize(server=server)
            
            # Set up connection listener
            self.connection_listener = ConnectionListener()
            NetworkTables.addConnectionListener(self.connection_listener.connected, immediateNotify=True)
            
            # Get table
            self.table = NetworkTables.getTable(table_name)
            self.connected = NetworkTables.isConnected()
            
        else:
            # NT4 API
            self.inst = NetworkTableInstance.getDefault()
            
            if server_address:
                self.inst.setServer(server_address)
            elif team_number:
                self.inst.setServerTeam(team_number)
            else:
                # Default to localhost if no team number or server specified
                self.inst.setServer("localhost")
            
            # Start client
            self.inst.startClient4("VisionProcessingClient")
            
            # Get table
            self.table = self.inst.getTable(table_name)
            self.connected = self.inst.isConnected()
            
            # Set up connection listener
            def _on_connect(connected, conn_info):
                self.connected = connected
                if connected:
                    logger.info(f"Connected to NetworkTables server at {conn_info.remote_id}")
                else:
                    logger.warning("Disconnected from NetworkTables server")
            
            self.inst.addConnectionListener(_on_connect, immediateNotify=True)
    
    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """
        Wait for connection to NetworkTables server.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if connected, False if timed out
        """
        if not HAS_NETWORKTABLES:
            return False
            
        start_time = time.time()
        while not self.connected:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                logger.warning(f"Timed out waiting for NetworkTables connection after {timeout} seconds")
                return False
                
            if LEGACY_NT:
                self.connected = NetworkTables.isConnected()
            else:
                self.connected = self.inst.isConnected()
                
        return True
    
    def publish_tracked_objects(self, tracked_objects: List[Dict[str, Any]]):
        """
        Publish tracked objects data to NetworkTables.
        
        Args:
            tracked_objects: List of tracked object dictionaries
        """
        if not HAS_NETWORKTABLES or not self.connected or not self.table:
            return
            
        # Convert to JSON and publish
        try:
            json_data = json.dumps(tracked_objects)
            
            if LEGACY_NT:
                self.table.putString("trackedObjects", json_data)
                self.table.putNumber("lastUpdateTime", time.time())
                self.table.putNumber("objectCount", len(tracked_objects))
            else:
                self.table.putString("trackedObjects", json_data)
                self.table.putDouble("lastUpdateTime", time.time())
                self.table.putInteger("objectCount", len(tracked_objects))
                
        except Exception as e:
            logger.error(f"Error publishing tracked objects: {e}")
    
    def publish_target_info(self, target_info: Dict[str, Any]):
        """
        Publish information about the selected target.
        
        Args:
            target_info: Dictionary with target information
        """
        if not HAS_NETWORKTABLES or not self.connected or not self.table:
            return
            
        try:
            # Publish individual values for more efficient access
            for key, value in target_info.items():
                if isinstance(value, bool):
                    self.table.putBoolean(f"target/{key}", value)
                elif isinstance(value, (int, float)):
                    if LEGACY_NT:
                        self.table.putNumber(f"target/{key}", value)
                    else:
                        if isinstance(value, int):
                            self.table.putInteger(f"target/{key}", value)
                        else:
                            self.table.putDouble(f"target/{key}", value)
                elif isinstance(value, str):
                    self.table.putString(f"target/{key}", value)
                else:
                    # Convert complex types to JSON
                    self.table.putString(f"target/{key}", json.dumps(value))
            
            # Update timestamp
            if LEGACY_NT:
                self.table.putNumber("target/timestamp", time.time())
                self.table.putBoolean("target/available", True)
            else:
                self.table.putDouble("target/timestamp", time.time())
                self.table.putBoolean("target/available", True)
                
        except Exception as e:
            logger.error(f"Error publishing target info: {e}")
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get configuration values from NetworkTables.
        
        Returns:
            Dictionary of configuration values
        """
        if not HAS_NETWORKTABLES or not self.connected or not self.table:
            return {}
            
        try:
            config_table = self.table.getSubTable("config")
            
            # Get all entries in the config table
            if LEGACY_NT:
                keys = config_table.getKeys()
                config = {}
                
                for key in keys:
                    entry_value = config_table.getValue(key, None)
                    if entry_value is not None:
                        config[key] = entry_value
                        
            else:
                entries = config_table.getEntries()
                config = {}
                
                for entry in entries:
                    key = entry.getName().split('/')[-1]
                    value = entry.getValue()
                    config[key] = value
            
            return config
            
        except Exception as e:
            logger.error(f"Error getting configuration: {e}")
            return {}
    
    def set_default_configuration(self, default_config: Dict[str, Any]):
        """
        Set default configuration values if they don't already exist.
        
        Args:
            default_config: Dictionary of default configuration values
        """
        if not HAS_NETWORKTABLES or not self.connected or not self.table:
            return
            
        try:
            config_table = self.table.getSubTable("config")
            
            for key, value in default_config.items():
                if isinstance(value, bool):
                    if LEGACY_NT:
                        if not config_table.containsKey(key):
                            config_table.putBoolean(key, value)
                    else:
                        config_table.getEntry(key).setDefaultBoolean(value)
                elif isinstance(value, int):
                    if LEGACY_NT:
                        if not config_table.containsKey(key):
                            config_table.putNumber(key, value)
                    else:
                        config_table.getEntry(key).setDefaultInteger(value)
                elif isinstance(value, float):
                    if LEGACY_NT:
                        if not config_table.containsKey(key):
                            config_table.putNumber(key, value)
                    else:
                        config_table.getEntry(key).setDefaultDouble(value)
                elif isinstance(value, str):
                    if LEGACY_NT:
                        if not config_table.containsKey(key):
                            config_table.putString(key, value)
                    else:
                        config_table.getEntry(key).setDefaultString(value)
                else:
                    # Convert complex types to JSON
                    if LEGACY_NT:
                        if not config_table.containsKey(key):
                            config_table.putString(key, json.dumps(value))
                    else:
                        config_table.getEntry(key).setDefaultString(json.dumps(value))
                        
        except Exception as e:
            logger.error(f"Error setting default configuration: {e}")
    
    def add_value_listener(self, key: str, callback: Callable[[str, Any], None]):
        """
        Add a listener for a specific NetworkTables value.
        
        Args:
            key: Key to listen for
            callback: Function to call when value changes
        """
        if not HAS_NETWORKTABLES or not self.connected or not self.table:
            return
            
        try:
            if LEGACY_NT:
                self.table.addEntryListener(
                    lambda table, key, value, is_new, flags: callback(key, value),
                    key=key
                )
            else:
                entry = self.table.getEntry(key)
                listener = entry.addListener(
                    lambda event: callback(event.name, event.value),
                    PubSubOptions.kValueRemote | PubSubOptions.kValueLocal
                )
                return listener  # Return listener ID for potential later removal
                
        except Exception as e:
            logger.error(f"Error adding value listener: {e}")


# For legacy NetworkTables API
class ConnectionListener:
    """NetworkTables connection listener for the legacy API."""
    
    def __init__(self):
        self.connected = False
        
    def connected(self, connected, info):
        """Called when connection status changes."""
        self.connected = connected
        if connected:
            logger.info(f"Connected to NetworkTables server at {info}")
        else:
            logger.warning("Disconnected from NetworkTables server")


def get_default_configuration() -> Dict[str, Any]:
    """
    Get default configuration values for vision tracking.
    
    Returns:
        Dictionary of default configuration values
    """
    return {
        "targetSelection": "lowest",  # Selection algorithm
        "publishRate": 10,  # How many times per second to publish data
        "minConfidence": 0.5,  # Minimum confidence for tracked objects
        "enableProcessing": True,  # Whether to enable vision processing
        "debug": False,  # Enable debug output
        "classFilter": "",  # Comma-separated list of classes to track
        "minSize": 20,  # Minimum object size in pixels
        "maxSize": 1000,  # Maximum object size in pixels
    } 