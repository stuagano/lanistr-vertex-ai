#!/usr/bin/env python3
"""
LANISTR Training Entry Point
This script properly sets up the Python path and runs the training.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the main function
from lanistr.main import main

if __name__ == "__main__":
    main() 