"""
Adapter for Markdown files
"""
from typing import List, Union, Dict, Any, Optional
from pathlib import Path
import re

from tinysearch.base import DataAdapter


class MarkdownAdapter(DataAdapter):
    """
    Adapter for extracting text from Markdown files
    """
    
    def __init__(
        self, 
        encoding: str = "utf-8", 
        strip_html: bool = True,
        extract_links: bool = True,
        extract_front_matter: bool = True
    ):
        """
        Initialize the Markdown adapter
        
        Args:
            encoding: Text encoding to use when reading files
            strip_html: Whether to remove HTML tags from the markdown content
            extract_links: Whether to extract link text and URLs
            extract_front_matter: Whether to extract YAML front matter as text
        """
        self.encoding = encoding
        self.strip_html = strip_html
        self.extract_links = extract_links
        self.extract_front_matter = extract_front_matter
        
        # Regex patterns
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.front_matter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    def extract(self, filepath: Union[str, Path]) -> List[str]:
        """
        Extract text content from the given Markdown file
        
        Args:
            filepath: Path to the Markdown file
            
        Returns:
            List of text strings extracted from the file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.is_dir():
            # If a directory is provided, process all Markdown files in it
            md_files = []
            for ext in [".md", ".markdown", ".mdown", ".mkdn"]:
                md_files.extend(filepath.glob(f"**/*{ext}"))
            
            result = []
            for file in md_files:
                try:
                    result.extend(self._extract_from_markdown(file))
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            return result
        else:
            # Process a single file
            return self._extract_from_markdown(filepath)
    
    def _extract_from_markdown(self, filepath: Path) -> List[str]:
        """
        Extract text from a single Markdown file
        
        Args:
            filepath: Path to the Markdown file
            
        Returns:
            List of text strings extracted from the file
        """
        result = []
        
        try:
            with open(filepath, "r", encoding=self.encoding) as f:
                content = f.read()
            
            # Extract YAML front matter if enabled
            if self.extract_front_matter:
                front_matter_match = self.front_matter_pattern.search(content)
                if front_matter_match:
                    front_matter = front_matter_match.group(1)
                    result.append(f"Front Matter: {front_matter}")
                    # Remove front matter from content to avoid duplication
                    content = self.front_matter_pattern.sub('', content)
            
            # Extract link text and URLs if enabled
            if self.extract_links:
                for match in self.link_pattern.finditer(content):
                    text, url = match.groups()
                    result.append(f"Link: {text} ({url})")
            
            # Remove HTML tags if enabled
            if self.strip_html:
                content = self.html_tag_pattern.sub('', content)
            
            # Split content by sections (headers)
            sections = re.split(r'(#+\s+.*)\n', content)
            current_section = "Main Content"
            
            for i, section in enumerate(sections):
                if i % 2 == 0:  # Content
                    if section.strip():
                        result.append(f"{current_section}: {section.strip()}")
                else:  # Header
                    current_section = section.strip()
        
        except Exception as e:
            print(f"Error extracting text from {filepath}: {e}")
        
        return result 