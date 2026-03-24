"""
Adapter for plain text files
"""
from typing import List, Union
from pathlib import Path

from tinysearch.base import DataAdapter


class TextAdapter(DataAdapter):
    """
    Adapter for extracting text from plain text files
    """
    
    def __init__(self, encoding: str = "utf-8", errors: str = "strict"):
        """
        Initialize the text adapter
        
        Args:
            encoding: Text encoding to use when reading files
            errors: How encoding errors are handled ('strict', 'replace', 'ignore')
        """
        self.encoding = encoding
        self.errors = errors
    
    def extract(self, filepath: Union[str, Path]) -> List[str]:
        """
        Extract text content from the given text file
        
        Args:
            filepath: Path to the text file
            
        Returns:
            List containing the text content of the file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.is_dir():
            raise ValueError(
                f"TextAdapter.extract() does not accept directories. "
                "Use iter_input_files() to iterate files, then call extract() on each."
            )

        # Process a single file
        with open(filepath, "r", encoding=self.encoding, errors=self.errors) as f:
            content = f.read()

        return [content] 