"""
Adapter for CSV files
"""
import csv
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from tinysearch.base import DataAdapter


class CSVAdapter(DataAdapter):
    """
    Adapter for extracting text from CSV files
    """
    
    def __init__(
        self, 
        encoding: str = "utf-8",
        delimiter: str = ",",
        include_header: bool = True,
        columns: Optional[List[str]] = None,
        row_format: Optional[str] = None
    ):
        """
        Initialize the CSV adapter
        
        Args:
            encoding: Text encoding to use when reading CSV files
            delimiter: CSV delimiter character
            include_header: Whether to include the header row in the extracted text
            columns: Specific columns to extract (by name). If None, extract all columns
            row_format: Format string for formatting rows. 
                        Use {column_name} to reference columns.
                        If None, rows will be formatted as "column1: value1, column2: value2, ..."
        """
        self.encoding = encoding
        self.delimiter = delimiter
        self.include_header = include_header
        self.columns = columns
        self.row_format = row_format
    
    def extract(self, filepath: Union[str, Path]) -> List[str]:
        """
        Extract text content from the given CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            List of text strings extracted from the CSV file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.is_dir():
            # If a directory is provided, process all CSV files in it
            csv_files = list(filepath.glob("**/*.csv"))
            
            result = []
            for file in csv_files:
                try:
                    result.extend(self._extract_from_csv(file))
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            return result
        else:
            # Process a single file
            return self._extract_from_csv(filepath)
    
    def _extract_from_csv(self, filepath: Path) -> List[str]:
        """
        Extract text from a single CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            List of text strings extracted from the CSV file
        """
        result = []
        
        try:
            with open(filepath, "r", encoding=self.encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                if reader.fieldnames is None:
                    raise ValueError("No columns found in the CSV file")
                
                # If specific columns are requested, validate they exist in the header
                if self.columns:
                    for col in self.columns:
                        if col not in reader.fieldnames:
                            print(f"Warning: Column '{col}' not found in {filepath}")
                
                # Include header as the first row if requested
                if self.include_header:
                    columns = self.columns or reader.fieldnames
                    header_text = ", ".join(columns)
                    result.append(header_text)
                
                # Process each row
                for row in reader:
                    row_text = self._format_row(row)
                    if row_text.strip():
                        result.append(row_text)
        except Exception as e:
            print(f"Error extracting text from {filepath}: {e}")
        
        return result
    
    def _format_row(self, row: Dict[str, str]) -> str:
        """
        Format a CSV row as text
        
        Args:
            row: Dictionary representing a CSV row
            
        Returns:
            Formatted text string
        """
        # Filter columns if specific columns are requested
        if self.columns:
            filtered_row = {col: row.get(col, "") for col in self.columns if col in row}
        else:
            filtered_row = row
        
        # Use custom format if provided
        if self.row_format:
            try:
                return self.row_format.format(**filtered_row)
            except KeyError as e:
                print(f"Warning: Format string contains invalid column: {e}")
                # Fall back to default format if custom format fails
        
        # Default format: "column1: value1, column2: value2, ..."
        return ", ".join([f"{key}: {value}" for key, value in filtered_row.items() if key]) 