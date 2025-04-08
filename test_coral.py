#!/usr/bin/env python3

"""
A simple script to test that the Coral TPU device is properly connected
and accessible from the container.
"""

import os
import sys

try:
    from pycoral.utils import edgetpu
    print("Successfully imported PyCoral libraries!")
    
    # List available TPU devices
    devices = edgetpu.list_edge_tpus()
    
    if devices:
        print(f"Found {len(devices)} Coral TPU device(s):")
        for i, device in enumerate(devices):
            print(f"  Device {i+1}: {device}")
        print("\nCoral TPU is ready to use!")
        sys.exit(0)
    else:
        print("No Coral TPU devices found. Please check your connection.")
        sys.exit(1)
        
except ImportError:
    print("Failed to import PyCoral libraries. Please check your installation.")
    sys.exit(1)
except Exception as e:
    print(f"Error while accessing Coral TPU: {e}")
    sys.exit(1) 