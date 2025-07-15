"""
Configuration management for TinySearch
"""
import os
import yaml
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path


class Config:
    """
    Configuration manager for TinySearch
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        """
        self.config = {}
        self.config_path = config_path
        
        # Default configuration
        self.default_config = {
            "adapter": {
                "type": "text",
                "params": {}
            },
            "splitter": {
                "chunk_size": 300,
                "chunk_overlap": 50
            },
            "embedder": {
                "model": "Qwen/Qwen-Embedding",
                "device": "cuda" if self._is_cuda_available() else "cpu",
                "params": {}
            },
            "indexer": {
                "index_path": "index.faiss",
                "metric": "cosine"
            },
            "query_engine": {
                "method": "template",
                "template": "请帮我查找：{query}",
                "top_k": 5
            },
            "flow": {
                "use_cache": True,
                "cache_dir": ".cache"
            },
            "api": {
                "auth_enabled": False,
                "rate_limit_enabled": False,
                "rate_limit": 60,  # requests
                "rate_limit_window": 60,  # seconds
                "default_key": "",  # default API key (if empty, will be generated)
                "master_key": ""  # master key for creating new API keys
            }
        }
        
        if config_path:
            self.load(config_path)
        else:
            self.config = self.default_config.copy()
    
    def load(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from file
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        
        Raises:
            ValueError: If the file format is not supported
        """
        config_path = Path(config_path)
        self.config_path = config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                loaded_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                loaded_config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )
        
        # Merge with default configuration
        self.config = self._merge_configs(self.default_config, loaded_config)
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file
        
        Args:
            config_path: Path to save the configuration file (YAML or JSON)
                         If None, use the path from initialization or previous load
        
        Raises:
            ValueError: If the file format is not supported or no path is specified
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ValueError("No configuration path specified")
        
        config_path = Path(config_path)
        
        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                yaml.dump(self.config, f, default_flow_style=False)
            elif config_path.suffix.lower() == ".json":
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if the key is not found
        
        Returns:
            Configuration value or default
        """
        if "." in key:
            parts = key.split(".")
            value = self.config
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    return default
                value = value[part]
            return value
        
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        if "." in key:
            parts = key.split(".")
            config = self.config
            for i, part in enumerate(parts[:-1]):
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self.config[key] = value
    
    def update(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            config: Configuration dictionary to merge
        """
        self.config = self._merge_configs(self.config, config)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries
        
        Args:
            base: Base configuration
            override: Override configuration
        
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _is_cuda_available(self) -> bool:
        """
        Check if CUDA is available
        
        Returns:
            True if CUDA is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value by key
        
        Args:
            key: Configuration key
        
        Returns:
            Configuration value
        
        Raises:
            KeyError: If the key is not found
        """
        if "." in key:
            parts = key.split(".")
            value = self.config
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    raise KeyError(key)
                value = value[part]
            return value
        
        if key not in self.config:
            raise KeyError(key)
        
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a key is in the configuration
        
        Args:
            key: Configuration key
        
        Returns:
            True if the key is in the configuration, False otherwise
        """
        if "." in key:
            parts = key.split(".")
            value = self.config
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    return False
                value = value[part]
            return True
        
        return key in self.config 