#!/usr/bin/env python3
"""
NetworkTables connection test utility.
Tests connecting to a NetworkTables server and publishing/receiving data.
"""

import sys
import time
import argparse
import logging
import os

try:
    import ntcore
    from ntcore import NetworkTableInstance
    NT_VERSION = 4
except ImportError:
    try:
        from networktables import NetworkTables
        NT_VERSION = 3
    except ImportError:
        print("ERROR: Neither ntcore (NT4) nor networktables (NT3) package found.")
        print("Please install one of them:")
        print("  pip install robotpy-cscore")  # Includes ntcore
        print("  pip install pynetworktables")  # For NT3
        sys.exit(1)

def setup_logging():
    """Configure logging for NetworkTables debugging."""
    logging.basicConfig(level=logging.DEBUG)

def connect_nt3(server_ip="10.0.0.2", client_name="Debug_Client"):
    """Connect to NetworkTables v3."""
    # Set up NetworkTables
    NetworkTables.startClientTeam(0)  # Use team number or IP
    if server_ip:
        NetworkTables.setServer(server_ip)
    
    NetworkTables.setUpdateRate(0.05)  # 50ms update rate
    NetworkTables.initialize(server=server_ip)
    
    # Set identity
    NetworkTables.setClientMode()
    NetworkTables.setNetworkIdentity(client_name)
    
    # Wait for connection
    print(f"Attempting to connect to NetworkTables at {server_ip}")
    count = 0
    while not NetworkTables.isConnected():
        time.sleep(0.1)
        count += 1
        if count >= 100:  # Timeout after 10 seconds
            print("ERROR: Failed to connect to NetworkTables server")
            return None
        if count % 10 == 0:
            print(f"Waiting for connection... ({count/10:.1f}s)")
    
    print(f"Connected to NetworkTables server at {server_ip}")
    return NetworkTables

def connect_nt4(server_ip="10.0.0.2", client_name="Debug_Client"):
    """Connect to NetworkTables v4."""
    # Create instance
    inst = NetworkTableInstance.getDefault()
    
    # Set identity and server
    inst.setServerTeam(0)  # Use team number or IP
    if server_ip:
        inst.setServer(server_ip)
    
    inst.setNetworkIdentity(client_name)
    inst.startClient4(client_name)
    
    # Wait for connection
    print(f"Attempting to connect to NetworkTables at {server_ip}")
    count = 0
    while not inst.isConnected():
        time.sleep(0.1)
        count += 1
        if count >= 100:  # Timeout after 10 seconds
            print("ERROR: Failed to connect to NetworkTables server")
            return None
        if count % 10 == 0:
            print(f"Waiting for connection... ({count/10:.1f}s)")
    
    print(f"Connected to NetworkTables server at {server_ip}")
    return inst

def test_nt3(nt, table_name="VisionData"):
    """Test NetworkTables v3 functionality."""
    # Get the vision table
    table = nt.getTable(table_name)
    
    # Test publishing values
    test_value = 0
    
    print("\nPublishing test values (press Ctrl+C to quit)...")
    try:
        while True:
            # Update test value
            test_value += 1
            timestamp = time.time()
            
            # Publish values
            table.putNumber("TestValue", test_value)
            table.putNumber("Timestamp", timestamp)
            table.putBoolean("IsConnected", True)
            table.putString("Status", "Debug Test Active")
            
            # Read back values to verify
            read_value = table.getNumber("TestValue", -1)
            
            print(f"Published: {test_value}, Read back: {read_value}")
            
            # Small delay
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    
    # Clean up
    table.putBoolean("IsConnected", False)
    table.putString("Status", "Debug Test Completed")

def test_nt4(inst, table_name="VisionData"):
    """Test NetworkTables v4 functionality."""
    # Create publishers
    table = inst.getTable(table_name)
    
    test_pub = table.getDoubleTopic("TestValue").publish()
    timestamp_pub = table.getDoubleTopic("Timestamp").publish()
    connected_pub = table.getBooleanTopic("IsConnected").publish()
    status_pub = table.getStringTopic("Status").publish()
    
    # Initialize values
    test_value = 0
    
    # Test publishing values
    print("\nPublishing test values (press Ctrl+C to quit)...")
    try:
        while True:
            # Update test value
            test_value += 1
            timestamp = time.time()
            
            # Publish values
            test_pub.set(test_value)
            timestamp_pub.set(timestamp)
            connected_pub.set(True)
            status_pub.set("Debug Test Active")
            
            # Read back values to verify
            sub = table.getDoubleTopic("TestValue").subscribe(-1)
            read_value = sub.get()
            
            print(f"Published: {test_value}, Read back: {read_value}")
            
            # Small delay
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    
    # Clean up
    connected_pub.set(False)
    status_pub.set("Debug Test Completed")

def main():
    parser = argparse.ArgumentParser(description="Test NetworkTables connection")
    parser.add_argument("--server", default="127.0.0.1", help="NetworkTables server IP address")
    parser.add_argument("--name", default="Vision_Debug_Client", help="Client name")
    parser.add_argument("--table", default="VisionData", help="Table name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        setup_logging()
    
    print(f"NetworkTables Test Utility (NT version: {NT_VERSION})")
    print("------------------------------------------")
    
    if NT_VERSION == 4:
        inst = connect_nt4(args.server, args.name)
        if inst:
            test_nt4(inst, args.table)
    else:
        nt = connect_nt3(args.server, args.name)
        if nt:
            test_nt3(nt, args.table)

if __name__ == "__main__":
    main() 