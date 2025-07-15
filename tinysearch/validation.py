"""
Data validation utilities for TinySearch
"""
from typing import Any, Dict, List, Optional, Union, Callable
import re
import os
from pathlib import Path


class ValidationError(Exception):
    """Exception raised for validation errors"""
    pass


class DataValidator:
    """
    Utility for validating data at various stages of processing
    """
    
    @staticmethod
    def validate_file_exists(path: Union[str, Path]) -> Path:
        """
        Validate that a file exists
        
        Args:
            path: Path to the file
            
        Returns:
            Path object for the validated file
            
        Raises:
            ValidationError: If the file does not exist
        """
        file_path = Path(path)
        if not file_path.exists():
            raise ValidationError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValidationError(f"Not a file: {path}")
        return file_path
    
    @staticmethod
    def validate_directory_exists(path: Union[str, Path]) -> Path:
        """
        Validate that a directory exists
        
        Args:
            path: Path to the directory
            
        Returns:
            Path object for the validated directory
            
        Raises:
            ValidationError: If the directory does not exist
        """
        dir_path = Path(path)
        if not dir_path.exists():
            raise ValidationError(f"Directory not found: {path}")
        if not dir_path.is_dir():
            raise ValidationError(f"Not a directory: {path}")
        return dir_path
    
    @staticmethod
    def validate_file_extension(path: Union[str, Path], valid_extensions: List[str]) -> Path:
        """
        Validate that a file has a valid extension
        
        Args:
            path: Path to the file
            valid_extensions: List of valid extensions (e.g., [".txt", ".pdf"])
            
        Returns:
            Path object for the validated file
            
        Raises:
            ValidationError: If the file does not have a valid extension
        """
        file_path = DataValidator.validate_file_exists(path)
        extension = file_path.suffix.lower()
        if extension not in valid_extensions:
            raise ValidationError(
                f"Invalid file extension: {extension}. Expected one of: {', '.join(valid_extensions)}"
            )
        return file_path
    
    @staticmethod
    def validate_embeddings(embeddings: List[List[float]], expected_dim: Optional[int] = None) -> bool:
        """
        Validate embeddings format and dimensions
        
        Args:
            embeddings: List of embedding vectors
            expected_dim: Expected dimensionality of embeddings
            
        Returns:
            True if embeddings are valid
            
        Raises:
            ValidationError: If embeddings are invalid
        """
        if not embeddings:
            raise ValidationError("Empty embeddings list")
        
        # Check that all elements are lists
        if not all(isinstance(emb, list) for emb in embeddings):
            raise ValidationError("All embeddings must be lists")
        
        # Check that all elements have the same dimension
        dims = [len(emb) for emb in embeddings]
        if len(set(dims)) > 1:
            raise ValidationError(f"Inconsistent embedding dimensions: {set(dims)}")
        
        # Check against expected dimension if provided
        if expected_dim is not None and dims[0] != expected_dim:
            raise ValidationError(f"Expected embedding dimension {expected_dim}, but got {dims[0]}")
            
        # Check that all values are floats
        if not all(isinstance(val, (int, float)) for emb in embeddings for val in emb):
            raise ValidationError("All embedding values must be numeric")
        
        return True
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that a configuration dictionary has required keys
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            
        Returns:
            True if the configuration is valid
            
        Raises:
            ValidationError: If any required key is missing
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationError(f"Missing required configuration keys: {', '.join(missing_keys)}")
        return True
    
    @staticmethod
    def validate_with_schema(data: Any, schema: Dict[str, Any]) -> bool:
        """
        Validate data against a schema
        
        Args:
            data: Data to validate
            schema: Schema definition
            
        Returns:
            True if the data is valid
            
        Raises:
            ValidationError: If the data does not match the schema
        """
        # Implement schema validation logic
        # This could use jsonschema or a similar library
        # For now, just provide a simple implementation
        
        if "type" in schema:
            if schema["type"] == "object" and not isinstance(data, dict):
                raise ValidationError(f"Expected object, got {type(data).__name__}")
            elif schema["type"] == "array" and not isinstance(data, list):
                raise ValidationError(f"Expected array, got {type(data).__name__}")
            elif schema["type"] == "string" and not isinstance(data, str):
                raise ValidationError(f"Expected string, got {type(data).__name__}")
            elif schema["type"] == "number" and not isinstance(data, (int, float)):
                raise ValidationError(f"Expected number, got {type(data).__name__}")
            elif schema["type"] == "boolean" and not isinstance(data, bool):
                raise ValidationError(f"Expected boolean, got {type(data).__name__}")
        
        if "properties" in schema and isinstance(data, dict):
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in data:
                    DataValidator.validate_with_schema(data[prop_name], prop_schema)
        
        if "required" in schema and isinstance(schema["required"], list) and isinstance(data, dict):
            for required_prop in schema["required"]:
                if required_prop not in data:
                    raise ValidationError(f"Missing required property: {required_prop}")
        
        return True
    
    @staticmethod
    def validate_text_non_empty(text: str) -> bool:
        """
        Validate that a text string is not empty
        
        Args:
            text: Text string to validate
            
        Returns:
            True if the text is not empty
            
        Raises:
            ValidationError: If the text is empty
        """
        if not text or text.isspace():
            raise ValidationError("Empty or whitespace-only text")
        return True
    
    @staticmethod
    def validate_list_non_empty(items: List[Any]) -> bool:
        """
        Validate that a list is not empty
        
        Args:
            items: List to validate
            
        Returns:
            True if the list is not empty
            
        Raises:
            ValidationError: If the list is empty
        """
        if not items:
            raise ValidationError("Empty list")
        return True
    
    @staticmethod
    def validate_custom(value: Any, validator_fn: Callable[[Any], bool], error_msg: str) -> bool:
        """
        Validate a value using a custom validation function
        
        Args:
            value: Value to validate
            validator_fn: Custom validation function that returns True if valid
            error_msg: Error message to raise if validation fails
            
        Returns:
            True if the value is valid
            
        Raises:
            ValidationError: If the validation fails
        """
        if not validator_fn(value):
            raise ValidationError(error_msg)
        return True 