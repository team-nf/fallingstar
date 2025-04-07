#!/usr/bin/env python3

import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
import os

class NetworkTablesManager:
    """
    Manager for NetworkTables communication with FRC robot.
    
    This class handles:
    1. Connection to NetworkTables
    2. Publishing detection results
    3. Receiving configuration updates
    4. Monitoring connection status
    """
    
    def __init__(self, team_number: int = None, server_ip: str = None, 
                 table_name: str = "Vision", connection_listener: bool = True):
        """
        Initialize NetworkTables manager.
        
        Args:
            team_number: FRC team number (optional if server_ip is provided)
            server_ip: Robot IP address (optional if team_number is provided)
            table_name: NetworkTables table name for vision data
            connection_listener: Whether to add connection listeners
        """
        self.table_name = table_name
        self.is_connected = False
        self.connection_listeners = []
        self._lock = threading.RLock()
        self._thread = None
        self._running = False
        
        # Handle environment variables
        if team_number is None and "TEAM_NUMBER" in os.environ:
            try:
                team_number = int(os.environ["TEAM_NUMBER"])
                print(f"Using team number from environment: {team_number}")
            except ValueError:
                print(f"Invalid team number in environment: {os.environ['TEAM_NUMBER']}")
                
        # Initialize NetworkTables
        try:
            from networktables import NetworkTablesInstance
            self.nt = NetworkTablesInstance.getDefault()
            
            # Set up connection
            if server_ip:
                # Connect to a specific IP
                self.nt.startClient(server_ip)
                print(f"NetworkTables client started, connecting to {server_ip}")
            elif team_number:
                # Connect to team number
                self.nt.startClientTeam(team_number)
                self.nt.startDSClient()
                print(f"NetworkTables client started for team {team_number}")
            else:
                # Start as server (for testing)
                self.nt.startServer()
                print("NetworkTables started in server mode (no team number or server IP provided)")
            
            # Get vision table
            self.table = self.nt.getTable(self.table_name)
            
            # Add connection listener
            if connection_listener:
                self._setup_connection_listener()
                
            # Start heartbeat thread
            self._start_heartbeat()
            
        except ImportError:
            print("WARNING: NetworkTables not available, running in offline mode")
            self.nt = None
            self.table = None
    
    def _setup_connection_listener(self):
        """Set up connection listener for NetworkTables."""
        try:
            from networktables import NetworkTablesInstance
            
            def _on_connection_change(connected, info):
                """Handle connection status changes."""
                self.is_connected = connected
                if connected:
                    print(f"Connected to NetworkTables at {info.remote_ip}")
                    # Call all registered connection listeners
                    for listener in self.connection_listeners:
                        try:
                            listener(True)
                        except Exception as e:
                            print(f"Error in connection listener: {e}")
                else:
                    print("Disconnected from NetworkTables")
                    # Call all registered connection listeners
                    for listener in self.connection_listeners:
                        try:
                            listener(False)
                        except Exception as e:
                            print(f"Error in connection listener: {e}")
            
            # Add connection listener
            self.nt.addConnectionListener(_on_connection_change, immediateNotify=True)
        except Exception as e:
            print(f"Error setting up connection listener: {e}")
    
    def _start_heartbeat(self):
        """
        Start a background thread that sends a heartbeat to NetworkTables.
        This helps detect disconnections faster.
        """
        if self.nt is None:
            return
            
        self._running = True
        
        def _heartbeat_thread():
            """Thread function for sending heartbeat."""
            counter = 0
            while self._running:
                with self._lock:
                    try:
                        if self.table:
                            # Update timestamp
                            self.table.putNumber("heartbeat", counter)
                            self.table.putNumber("timestamp", time.time())
                            counter += 1
                    except Exception as e:
                        print(f"Error in heartbeat thread: {e}")
                time.sleep(0.25)  # Update 4 times per second
        
        # Start heartbeat thread
        self._thread = threading.Thread(target=_heartbeat_thread, daemon=True)
        self._thread.start()
    
    def add_connection_listener(self, listener: Callable[[bool], None]):
        """
        Add a listener that will be called when connection status changes.
        
        Args:
            listener: Function that takes a boolean (connected status)
        """
        self.connection_listeners.append(listener)
        
        # Call immediately with current status
        if self.is_connected:
            listener(True)
    
    def publish_detection_results(self, tracked_objects: Dict[int, Dict[str, Any]], 
                                 fps: float = 0.0, latency_ms: float = 0.0):
        """
        Publish detection and tracking results to NetworkTables.
        
        Args:
            tracked_objects: Dictionary of tracked objects
            fps: Current FPS
            latency_ms: Processing latency in milliseconds
        """
        if self.table is None:
            return
            
        with self._lock:
            try:
                # Update performance metrics
                self.table.putNumber("fps", fps)
                self.table.putNumber("latency_ms", latency_ms)
                
                # Clear previous targets
                for key in self.table.getKeys():
                    if key.startswith("target_") and not any(f"target_{id}" == key for id in tracked_objects):
                        self.table.delete(key)
                
                # Update detected targets
                for track_id, track_data in tracked_objects.items():
                    # Extract bounding box
                    bbox = track_data.get('bbox', None)
                    if not bbox:
                        continue
                        
                    # Get position and dimensions
                    x, y, w, h = bbox.get_coords()
                    
                    # Create target info
                    target_info = {
                        "x": x + w/2,  # Center X
                        "y": y + h/2,  # Center Y
                        "width": w,
                        "height": h,
                        "area": w * h,
                        "confidence": bbox.confidence,
                        "class_id": bbox.class_id,
                        "class_name": track_data.get('class_name', "unknown")
                    }
                    
                    # Add track-specific data
                    if 'age' in track_data:
                        target_info['age'] = track_data['age']
                    if 'hits' in track_data:
                        target_info['hits'] = track_data['hits']
                        
                    # Publish to NetworkTables
                    self.table.putString(f"target_{track_id}", json.dumps(target_info))
                
                # Update number of targets
                self.table.putNumber("num_targets", len(tracked_objects))
                
            except Exception as e:
                print(f"Error publishing to NetworkTables: {e}")
    
    def get_config_value(self, key: str, default_value: Any) -> Any:
        """
        Get a configuration value from NetworkTables.
        
        Args:
            key: Configuration key
            default_value: Default value if key doesn't exist
            
        Returns:
            The configuration value
        """
        if self.table is None:
            return default_value
            
        with self._lock:
            try:
                # Get entry
                entry = self.table.getEntry(f"config/{key}")
                
                # Return appropriate type based on default_value
                if isinstance(default_value, bool):
                    return entry.getBoolean(default_value)
                elif isinstance(default_value, int):
                    return entry.getNumber(default_value)
                elif isinstance(default_value, float):
                    return entry.getNumber(default_value)
                elif isinstance(default_value, str):
                    return entry.getString(default_value)
                else:
                    # Try to parse as JSON
                    json_str = entry.getString(None)
                    if json_str:
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            return default_value
                    return default_value
            except Exception as e:
                print(f"Error getting config value {key}: {e}")
                return default_value
    
    def set_config_value(self, key: str, value: Any):
        """
        Set a configuration value in NetworkTables.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        if self.table is None:
            return
            
        with self._lock:
            try:
                # Get entry
                entry = self.table.getEntry(f"config/{key}")
                
                # Set appropriate type
                if isinstance(value, bool):
                    entry.setBoolean(value)
                elif isinstance(value, (int, float)):
                    entry.setNumber(value)
                elif isinstance(value, str):
                    entry.setString(value)
                else:
                    # Convert to JSON
                    entry.setString(json.dumps(value))
            except Exception as e:
                print(f"Error setting config value {key}: {e}")
    
    def get_all_config_values(self) -> Dict[str, Any]:
        """
        Get all configuration values from NetworkTables.
        
        Returns:
            Dictionary of configuration values
        """
        if self.table is None:
            return {}
            
        with self._lock:
            try:
                config = {}
                for key in self.table.getKeys():
                    if key.startswith("config/"):
                        config_key = key[7:]  # Remove "config/" prefix
                        entry = self.table.getEntry(key)
                        
                        # Try to get different types
                        value = entry.getValue()
                        
                        # Handle JSON strings
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                pass
                                
                        config[config_key] = value
                return config
            except Exception as e:
                print(f"Error getting all config values: {e}")
                return {}
    
    def listen_for_config_changes(self, callback: Callable[[str, Any], None]):
        """
        Listen for changes to configuration values.
        
        Args:
            callback: Function that takes key and value
        """
        if self.table is None:
            return
            
        try:
            def _value_changed(table, key, entry, value, isNew):
                """Handle value changed events."""
                if key.startswith("config/"):
                    config_key = key[7:]  # Remove "config/" prefix
                    
                    # Handle JSON strings
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                            
                    # Call callback
                    callback(config_key, value)
            
            # Add listener
            self.table.addEntryListener(_value_changed, immediateNotify=True, prefix="config/")
        except Exception as e:
            print(f"Error setting up config change listener: {e}")
    
    def shutdown(self):
        """
        Shutdown NetworkTables connection and threads.
        """
        self._running = False
        
        if self._thread:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None
        
        if self.nt:
            try:
                self.nt.stopClient()
            except Exception as e:
                print(f"Error stopping NetworkTables client: {e}")


# For testing
if __name__ == "__main__":
    # Create manager
    ntm = NetworkTablesManager(team_number=0000)  # Replace with your team number
    
    # Test connection listener
    def connection_listener(connected):
        print(f"Connection listener: {'Connected' if connected else 'Disconnected'}")
    
    ntm.add_connection_listener(connection_listener)
    
    # Test configuration
    ntm.set_config_value("test_bool", True)
    ntm.set_config_value("test_int", 42)
    ntm.set_config_value("test_float", 3.14)
    ntm.set_config_value("test_string", "Hello, NetworkTables!")
    ntm.set_config_value("test_json", {"foo": "bar", "baz": 123})
    
    # Test configuration listener
    def config_listener(key, value):
        print(f"Config changed: {key} = {value}")
    
    ntm.listen_for_config_changes(config_listener)
    
    # Test publishing detection results
    from detection.tracker import BoundingBox
    
    try:
        for i in range(10):
            # Create fake detection results
            tracked_objects = {
                1: {
                    "bbox": BoundingBox(100, 100, 50, 50, 0.9, 0),
                    "class_name": "cube"
                },
                2: {
                    "bbox": BoundingBox(200, 200, 60, 40, 0.8, 1),
                    "class_name": "cone"
                }
            }
            
            # Publish
            ntm.publish_detection_results(tracked_objects, fps=30.0, latency_ms=20.0)
            
            # Wait
            print(f"Published detection results {i+1}/10")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        # Shutdown
        ntm.shutdown() 