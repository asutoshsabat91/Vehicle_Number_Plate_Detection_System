#!/usr/bin/env python3
"""
Main entry point for the ANPR system.
"""

import sys
import os
from gui.main_window import main

if __name__ == '__main__':
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    main() 