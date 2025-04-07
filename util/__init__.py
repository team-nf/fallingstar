"""
Utility Package for Vision Processing

This package provides utilities for vision processing, including:
- NetworkTables integration for communicating with robotics systems
- Target selection algorithms for choosing which object to track
"""

from .networktables import (
    NetworkTablesInterface,
    get_default_configuration,
    HAS_NETWORKTABLES
)

from .selection import (
    select_target,
    target_info_to_dict,
    filter_objects,
    SELECTION_FUNCTIONS
)

__all__ = [
    # NetworkTables
    'NetworkTablesInterface',
    'get_default_configuration',
    'HAS_NETWORKTABLES',
    
    # Selection
    'select_target',
    'target_info_to_dict',
    'filter_objects',
    'SELECTION_FUNCTIONS'
] 