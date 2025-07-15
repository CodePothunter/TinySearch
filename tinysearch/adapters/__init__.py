"""
Data adapters for extracting text from various file formats
"""

# Make all adapter classes available from the root module
from .text import TextAdapter
from .pdf import PDFAdapter
from .csv import CSVAdapter
from .markdown import MarkdownAdapter
from .json_adapter import JSONAdapter

__all__ = [
    "TextAdapter",
    "PDFAdapter", 
    "CSVAdapter", 
    "MarkdownAdapter",
    "JSONAdapter"
] 