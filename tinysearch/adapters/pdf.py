"""
Adapter for PDF files
"""
from typing import List, Optional, Union
from pathlib import Path
import os

from tinysearch.base import DataAdapter


class PDFAdapter(DataAdapter):
    """
    Adapter for extracting text from PDF files
    """
    
    def __init__(self, pages: Optional[Union[int, List[int], slice]] = None):
        """
        Initialize the PDF adapter
        
        Args:
            pages: Specific pages to extract. Can be:
                  - None to extract all pages
                  - An integer for a specific page (1-indexed)
                  - A list of page numbers (1-indexed)
                  - A slice object for a range of pages (0-indexed)
        """
        self.pages = pages
        
        try:
            import fitz
            self._fitz_available = True
        except ImportError:
            try:
                import PyPDF2
                self._fitz_available = False
            except ImportError:
                raise ImportError(
                    "Neither PyMuPDF (fitz) nor PyPDF2 is installed. "
                    "Please install at least one of them: "
                    "pip install pymupdf or pip install pypdf2"
                )
    
    def extract(self, filepath: Union[str, Path]) -> List[str]:
        """
        Extract text content from the given PDF file
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            List of text strings extracted from the PDF, one per page
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.is_dir():
            # If a directory is provided, process all PDF files in it
            pdf_files = list(filepath.glob("**/*.pdf"))
            
            result = []
            for file in pdf_files:
                try:
                    result.extend(self._extract_from_pdf(file))
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            return result
        else:
            # Process a single file
            return self._extract_from_pdf(filepath)
    
    def _extract_from_pdf(self, filepath: Path) -> List[str]:
        """
        Extract text from a single PDF file
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            List of text strings extracted from the PDF, one per page
        """
        if self._fitz_available:
            return self._extract_with_fitz(filepath)
        else:
            return self._extract_with_pypdf2(filepath)
    
    def _extract_with_fitz(self, filepath: Path) -> List[str]:
        """
        Extract text using PyMuPDF (fitz)
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            List of text strings extracted from the PDF, one per page
        """
        import fitz
        
        result = []
        
        try:
            with fitz.open(filepath) as doc: # type: ignore
                if self.pages is None:
                    # Extract all pages
                    for page_num in range(doc.page_count):
                        page = doc[page_num]
                        text = page.get_text()
                        if text.strip():
                            result.append(text)
                
                elif isinstance(self.pages, int):
                    # Extract a specific page
                    page_idx = self.pages - 1  # Convert 1-indexed to 0-indexed
                    if 0 <= page_idx < doc.page_count:
                        page = doc[page_idx]
                        text = page.get_text()
                        if text.strip():
                            result.append(text)
                
                elif isinstance(self.pages, list):
                    # Extract specific pages
                    for page_num in self.pages:
                        page_idx = page_num - 1  # Convert 1-indexed to 0-indexed
                        if 0 <= page_idx < doc.page_count:
                            page = doc[page_idx]
                            text = page.get_text()
                            if text.strip():
                                result.append(text)
                
                elif isinstance(self.pages, slice):
                    # Extract a range of pages
                    start = self.pages.start or 0
                    stop = self.pages.stop or doc.page_count
                    step = self.pages.step or 1
                    
                    for page_idx in range(start, min(stop, doc.page_count), step):
                        page = doc[page_idx]
                        text = page.get_text()
                        if text.strip():
                            result.append(text)
        except Exception as e:
            print(f"Error extracting text from {filepath} using fitz: {e}")
        
        return result
    
    def _extract_with_pypdf2(self, filepath: Path) -> List[str]:
        """
        Extract text using PyPDF2
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            List of text strings extracted from the PDF, one per page
        """
        import PyPDF2
        
        result = []
        
        try:
            with open(filepath, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                if self.pages is None:
                    # Extract all pages
                    for page_num in range(num_pages):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text.strip():
                            result.append(text)
                
                elif isinstance(self.pages, int):
                    # Extract a specific page
                    page_idx = self.pages - 1  # Convert 1-indexed to 0-indexed
                    if 0 <= page_idx < num_pages:
                        page = reader.pages[page_idx]
                        text = page.extract_text()
                        if text.strip():
                            result.append(text)
                
                elif isinstance(self.pages, list):
                    # Extract specific pages
                    for page_num in self.pages:
                        page_idx = page_num - 1  # Convert 1-indexed to 0-indexed
                        if 0 <= page_idx < num_pages:
                            page = reader.pages[page_idx]
                            text = page.extract_text()
                            if text.strip():
                                result.append(text)
                
                elif isinstance(self.pages, slice):
                    # Extract a range of pages
                    start = self.pages.start or 0
                    stop = self.pages.stop or num_pages
                    step = self.pages.step or 1
                    
                    for page_idx in range(start, min(stop, num_pages), step):
                        page = reader.pages[page_idx]
                        text = page.extract_text()
                        if text.strip():
                            result.append(text)
        except Exception as e:
            print(f"Error extracting text from {filepath} using PyPDF2: {e}")
        
        return result 