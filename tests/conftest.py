"""
Pytest configuration file
"""
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so that we can import tinysearch
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary packages here to ensure they are available for all tests
import pytest
import yaml
import json
import numpy as np 