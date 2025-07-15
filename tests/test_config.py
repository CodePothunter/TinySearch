"""
Tests for configuration management
"""
import os
import tempfile
import yaml
import json
import sys
from pathlib import Path
import pytest

# Add the parent directory to sys.path so that we can import tinysearch
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinysearch.config import Config


def test_config_default():
    """Test default configuration"""
    config = Config()
    
    # Check default values
    assert config.get("adapter.type") == "text"
    assert config.get("splitter.chunk_size") == 300
    assert config.get("splitter.chunk_overlap") == 50
    assert config.get("embedder.model") == "Qwen/Qwen-Embedding"
    assert config.get("indexer.index_path") == "index.faiss"


def test_config_load_yaml():
    """Test loading configuration from YAML file"""
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump({
            "adapter": {
                "type": "pdf"
            },
            "splitter": {
                "chunk_size": 500
            },
            "embedder": {
                "model": "custom-model"
            }
        }, f)
        filepath = f.name
    
    try:
        # Load configuration
        config = Config(filepath)
        
        # Check loaded values
        assert config.get("adapter.type") == "pdf"
        assert config.get("splitter.chunk_size") == 500
        assert config.get("embedder.model") == "custom-model"
        
        # Default values should still be present for unspecified settings
        assert config.get("splitter.chunk_overlap") == 50
        assert config.get("indexer.index_path") == "index.faiss"
    
    finally:
        # Clean up
        os.unlink(filepath)


def test_config_load_json():
    """Test loading configuration from JSON file"""
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({
            "adapter": {
                "type": "markdown"
            },
            "indexer": {
                "metric": "l2"
            }
        }, f)
        filepath = f.name
    
    try:
        # Load configuration
        config = Config(filepath)
        
        # Check loaded values
        assert config.get("adapter.type") == "markdown"
        assert config.get("indexer.metric") == "l2"
        
        # Default values should still be present for unspecified settings
        assert config.get("splitter.chunk_size") == 300
        assert config.get("embedder.model") == "Qwen/Qwen-Embedding"
    
    finally:
        # Clean up
        os.unlink(filepath)


def test_config_get_set():
    """Test getting and setting configuration values"""
    config = Config()
    
    # Get existing values
    assert config.get("adapter.type") == "text"
    assert config.get("splitter.chunk_size") == 300
    
    # Get with default value for non-existent keys
    assert config.get("non_existent_key", "default") == "default"
    
    # Set values
    config.set("adapter.type", "pdf")
    config.set("custom.nested.key", "value")
    
    # Get updated values
    assert config.get("adapter.type") == "pdf"
    assert config.get("custom.nested.key") == "value"
    
    # Dictionary-style access
    assert config["adapter.type"] == "pdf"
    config["splitter.chunk_size"] = 400
    assert config["splitter.chunk_size"] == 400
    
    # Check contains
    assert "adapter.type" in config
    assert "non_existent_key" not in config


def test_config_save():
    """Test saving configuration to a file"""
    config = Config()
    
    # Modify configuration
    config.set("adapter.type", "markdown")
    config.set("splitter.chunk_size", 600)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save as YAML
        yaml_path = os.path.join(temp_dir, "config.yaml")
        config.save(yaml_path)
        
        # Save as JSON
        json_path = os.path.join(temp_dir, "config.json")
        config.save(json_path)
        
        # Load the saved YAML file
        yaml_config = Config(yaml_path)
        assert yaml_config.get("adapter.type") == "markdown"
        assert yaml_config.get("splitter.chunk_size") == 600
        
        # Load the saved JSON file
        json_config = Config(json_path)
        assert json_config.get("adapter.type") == "markdown"
        assert json_config.get("splitter.chunk_size") == 600


def test_config_update():
    """Test updating configuration with a dictionary"""
    config = Config()
    
    # Update with a dictionary
    config.update({
        "adapter": {
            "type": "json"
        },
        "embedder": {
            "model": "new-model",
            "device": "cpu"
        }
    })
    
    # Check updated values
    assert config.get("adapter.type") == "json"
    assert config.get("embedder.model") == "new-model"
    assert config.get("embedder.device") == "cpu"
    
    # Other values should remain unchanged
    assert config.get("splitter.chunk_size") == 300


def test_config_error_cases():
    """Test error cases for configuration"""
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        Config("/path/to/non_existent_file.yaml")
    
    # Unsupported file format
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("This is not a YAML or JSON file")
        filepath = f.name
    
    try:
        with pytest.raises(ValueError):
            Config(filepath)
    
    finally:
        # Clean up
        os.unlink(filepath)
    
    # Save without path
    config = Config()
    with pytest.raises(ValueError):
        config.save()
    
    # Get non-existent key without default
    with pytest.raises(KeyError):
        config["non_existent_key"] 