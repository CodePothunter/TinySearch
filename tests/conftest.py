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

# Define helper methods for the FlowController in tests
def mock_add_watch_path(self, path, recursive=None):
    """Add a path to watch for changes"""
    if self._hot_update_manager:
        self._hot_update_manager.add_watch_path(path, recursive)

def mock_remove_watch_path(self, path):
    """Remove a path from being watched"""
    if self._hot_update_manager:
        self._hot_update_manager.remove_watch_path(path)

# Apply mocks before any tests run
@pytest.fixture(autouse=True)
def setup_tests(monkeypatch):
    """Setup test environment"""
    # Import controller after we've added the stub HotUpdateManager
    from tinysearch.flow.controller import FlowController
    
    # Patch FlowController methods to work with our stub
    monkeypatch.setattr(FlowController, "add_watch_path", mock_add_watch_path)
    monkeypatch.setattr(FlowController, "remove_watch_path", mock_remove_watch_path) 