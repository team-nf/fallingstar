#!/usr/bin/env python3

"""
Simple NetworkTables test script using ntcore
"""

import time
from ntcore import NetworkTableInstance, EventFlags
import logging

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

def main():
    # Get the default NetworkTables instance
    inst = NetworkTableInstance.getDefault()
    
    # Start a server
    inst.startServer()
    print("Started NetworkTables server")
    
    # Get the SmartDashboard table
    sd = inst.getTable("SmartDashboard")
    
    # Test putting values
    print("\nPutting test values...")
    sd.putNumber("test_number", 42)
    sd.putString("test_string", "Hello NetworkTables!")
    sd.putBoolean("test_boolean", True)
    
    # Test getting values
    print("\nGetting test values...")
    print(f"Number: {sd.getNumber('test_number', 0)}")
    print(f"String: {sd.getString('test_string', '')}")
    print(f"Boolean: {sd.getBoolean('test_boolean', False)}")
    
    # Test value change listener
    def value_changed(table, key, event):
        print(f"Value changed: {key} = {event.value}")
    
    # Add listener for all value changes
    sd.addListener(EventFlags.kValueAll, value_changed)
    
    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        inst.stopServer()

if __name__ == "__main__":
    main() 