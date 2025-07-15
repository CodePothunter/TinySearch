"""
Adapter for JSON files
"""
import json
from typing import List, Union, Dict, Any, Optional
from pathlib import Path

from tinysearch.base import DataAdapter


class JSONAdapter(DataAdapter):
    """
    Adapter for extracting text from JSON files
    """
    
    def __init__(
        self, 
        encoding: str = "utf-8",
        fields: Optional[List[str]] = None,
        include_keys: bool = True,
        flatten: bool = True,
        max_depth: int = 5
    ):
        """
        Initialize the JSON adapter
        
        Args:
            encoding: Text encoding to use when reading files
            fields: Specific field paths to extract (dot notation for nested fields, e.g., ["user.name", "description"])
                   If None, extract all fields
            include_keys: Whether to include the field names in the extracted text
            flatten: Whether to flatten nested JSON structures into a flat list
            max_depth: Maximum depth to traverse for nested objects (to prevent infinite recursion)
        """
        self.encoding = encoding
        self.fields = fields
        self.include_keys = include_keys
        self.flatten = flatten
        self.max_depth = max_depth
    
    def extract(self, filepath: Union[str, Path]) -> List[str]:
        """
        Extract text content from the given JSON file
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of text strings extracted from the file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.is_dir():
            # If a directory is provided, process all JSON files in it
            json_files = list(filepath.glob("**/*.json"))
            
            result = []
            for file in json_files:
                try:
                    result.extend(self._extract_from_json(file))
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            return result
        else:
            # Process a single file
            return self._extract_from_json(filepath)
    
    def _extract_from_json(self, filepath: Path) -> List[str]:
        """
        Extract text from a single JSON file
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of text strings extracted from the file
        """
        result = []
        
        try:
            with open(filepath, "r", encoding=self.encoding) as f:
                data = json.load(f)
            
            if self.fields:
                # Extract specific fields
                for field_path in self.fields:
                    value = self._get_nested_field(data, field_path)
                    if value is not None:
                        text = self._format_field(field_path, value)
                        if text:
                            result.append(text)
            else:
                # Extract all fields
                if isinstance(data, list):
                    # JSON array
                    for i, item in enumerate(data):
                        texts = self._extract_value(f"item_{i}", item)
                        result.extend(texts)
                else:
                    # JSON object
                    texts = self._extract_value("", data, depth=0)
                    result.extend(texts)
        
        except Exception as e:
            print(f"Error extracting text from {filepath}: {e}")
        
        return result
    
    def _get_nested_field(self, data: Union[Dict[str, Any], List[Any]], field_path: str) -> Any:
        """
        Get a nested field value from a JSON object using dot notation
        
        Args:
            data: JSON object or array
            field_path: Field path in dot notation (e.g., "user.name")
            
        Returns:
            Field value or None if not found
        """
        parts = field_path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                return None
        
        return current
    
    def _format_field(self, key: str, value: Any) -> str:
        """
        Format a field as text
        
        Args:
            key: Field name
            value: Field value
            
        Returns:
            Formatted text string
        """
        if isinstance(value, (str, int, float, bool)):
            return f"{key}: {value}" if self.include_keys else str(value)
        elif isinstance(value, (list, dict)):
            texts = self._extract_value(key, value)
            return "\n".join(texts)
        else:
            return ""
    
    def _extract_value(self, prefix: str, value: Any, depth: int = 0) -> List[str]:
        """
        Extract text from a JSON value
        
        Args:
            prefix: Prefix for field names
            value: JSON value
            depth: Current recursion depth
            
        Returns:
            List of text strings extracted from the value
        """
        result = []
        
        # Guard against excessive recursion
        if depth >= self.max_depth:
            return [f"{prefix}: [Maximum depth reached]"] if self.include_keys and prefix else ["[Maximum depth reached]"]
        
        if isinstance(value, dict):
            # Process dictionary
            if self.flatten:
                # Flatten dictionary
                for key, val in value.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    result.extend(self._extract_value(new_prefix, val, depth + 1))
            else:
                # Keep as single entry
                dict_text = json.dumps(value, ensure_ascii=False)
                if self.include_keys and prefix:
                    result.append(f"{prefix}: {dict_text}")
                else:
                    result.append(dict_text)
        
        elif isinstance(value, list):
            # Process list
            if self.flatten:
                # Flatten list
                for i, item in enumerate(value):
                    new_prefix = f"{prefix}[{i}]" if prefix else f"item_{i}"
                    result.extend(self._extract_value(new_prefix, item, depth + 1))
            else:
                # Keep as single entry
                list_text = json.dumps(value, ensure_ascii=False)
                if self.include_keys and prefix:
                    result.append(f"{prefix}: {list_text}")
                else:
                    result.append(list_text)
        
        else:
            # Process primitive values
            if self.include_keys and prefix:
                result.append(f"{prefix}: {value}")
            else:
                result.append(str(value))
        
        return result 