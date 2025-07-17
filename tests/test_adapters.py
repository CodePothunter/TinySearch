"""
Tests for data adapters
"""
import os
import pytest
from pathlib import Path
import tempfile
import sys

# Add the parent directory to sys.path so that we can import tinysearch
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinysearch.adapters.text import TextAdapter
from tinysearch.base import DataAdapter


def test_text_adapter_file():
    """Test TextAdapter with a single file"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test file.\nIt has multiple lines.\nThird line.")
        filepath = f.name
    
    try:
        # Create an adapter
        adapter = TextAdapter()
        
        # Extract text
        texts = adapter.extract(filepath)
        
        # Check results
        assert len(texts) == 1
        assert texts[0] == "This is a test file.\nIt has multiple lines.\nThird line."
    
    finally:
        # Clean up
        os.unlink(filepath)


def test_text_adapter_directory():
    """Test TextAdapter with a directory"""
    # Create a temporary directory with multiple files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        file1 = Path(temp_dir) / "file1.txt"
        file2 = Path(temp_dir) / "file2.txt"
        file3 = Path(temp_dir) / "subdir" / "file3.txt"
        
        # Create subdirectory
        os.makedirs(os.path.dirname(file3), exist_ok=True)
        
        # Write content to files
        with open(file1, "w") as f:
            f.write("Content of file 1")
        
        with open(file2, "w") as f:
            f.write("Content of file 2")
        
        with open(file3, "w") as f:
            f.write("Content of file 3")
        
        # Create an adapter
        adapter = TextAdapter()
        
        # Extract text
        texts = adapter.extract(temp_dir)
        
        # Check results
        assert len(texts) == 3
        assert "Content of file 1" in texts
        assert "Content of file 2" in texts
        assert "Content of file 3" in texts


def test_text_adapter_encoding():
    """Test TextAdapter with different encodings"""
    # Create a temporary file with non-ASCII content
    content = "This file contains non-ASCII characters: üñîçøδε"
    
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
        f.write(content.encode("utf-8"))
        filepath = f.name
    
    try:
        # Create an adapter with UTF-8 encoding
        adapter_utf8 = TextAdapter(encoding="utf-8")
        
        # Extract text
        texts_utf8 = adapter_utf8.extract(filepath)
        
        # Check results
        assert len(texts_utf8) == 1
        assert texts_utf8[0] == content
        
        # Create a TextAdapter with errors='replace' to handle non-ASCII characters
        adapter_ascii = TextAdapter(encoding="ascii", errors="replace")
        
        # This should not raise an error, but should replace non-ASCII characters
        texts_ascii = adapter_ascii.extract(filepath)
        assert len(texts_ascii) == 1
        assert texts_ascii[0] != content  # The content should be different due to replacements
    
    finally:
        # Clean up
        os.unlink(filepath)


def test_text_adapter_nonexistent_file():
    """Test TextAdapter with a nonexistent file"""
    adapter = TextAdapter()
    
    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        adapter.extract("/path/to/nonexistent/file.txt")


def test_text_adapter_encoding_errors():
    """Test TextAdapter handling of encoding errors"""
    # Create a file with non-UTF8 content
    content_bytes = b'\x80\x81\x82\x83\xff'  # Invalid UTF-8 bytes
    
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
        f.write(content_bytes)
        filepath = f.name
    
    try:
        # Adapter with strict encoding should fail
        adapter_strict = TextAdapter(encoding="utf-8", errors="strict")
        with pytest.raises(UnicodeDecodeError):
            adapter_strict.extract(filepath)
        
        # Adapter with replace error handling should succeed
        adapter_replace = TextAdapter(encoding="utf-8", errors="replace")
        texts_replace = adapter_replace.extract(filepath)
        assert len(texts_replace) == 1
        assert texts_replace[0] != ""  # Should contain replacement chars
        
        # Adapter with ignore error handling should succeed
        adapter_ignore = TextAdapter(encoding="utf-8", errors="ignore")
        texts_ignore = adapter_ignore.extract(filepath)
        assert len(texts_ignore) == 1
        
    finally:
        # Clean up
        os.unlink(filepath)


def test_text_adapter_empty_file():
    """Test TextAdapter with an empty file"""
    # Create an empty file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = f.name
    
    try:
        # Empty file should return empty text
        adapter = TextAdapter()
        texts = adapter.extract(filepath)
        
        assert len(texts) == 1
        assert texts[0] == ""
    
    finally:
        # Clean up
        os.unlink(filepath)


def test_text_adapter_large_file_handling():
    """Test TextAdapter with a large file to ensure memory efficiency"""
    # Create a temporary large-ish file (not too large for testing)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Write about 1MB of data
        line = "This is a test line with some content. " * 100  # ~3KB per line
        for _ in range(350):  # ~1MB total
            f.write(line + "\n")
        filepath = f.name
    
    try:
        # Should handle large files without memory issues
        adapter = TextAdapter()
        texts = adapter.extract(filepath)
        
        assert len(texts) == 1
        assert len(texts[0]) > 1000000  # Should contain all the text (>1MB)
    
    finally:
        # Clean up
        os.unlink(filepath)


def test_text_adapter_directory_empty():
    """Test TextAdapter with an empty directory"""
    # Create a temporary empty directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Directory with no files should return empty list
        adapter = TextAdapter()
        texts = adapter.extract(temp_dir)
        
        assert len(texts) == 0


def test_text_adapter_directory_non_text_files():
    """Test TextAdapter with non-text files in directory"""
    # Create a temporary directory with a binary file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a binary file
        bin_file = Path(temp_dir) / "binary.bin"
        with open(bin_file, "wb") as f:
            f.write(b'\x00\x01\x02\x03\x04')
        
        # Create a text file
        txt_file = Path(temp_dir) / "text.txt"
        with open(txt_file, "w") as f:
            f.write("This is a text file")
        
        # Extract - binary file should be skipped by default (errors='ignore')
        adapter = TextAdapter()
        texts = adapter.extract(temp_dir)
        
        # Should only get text from the text file
        assert len(texts) == 1
        assert texts[0] == "This is a text file"


def test_adapter_interface():
    """Test DataAdapter interface"""
    # Ensure TextAdapter implements DataAdapter interface
    assert issubclass(TextAdapter, DataAdapter)
    
    # Create an instance
    adapter = TextAdapter()
    
    # Check that it has the required methods
    assert hasattr(adapter, "extract")
    assert callable(adapter.extract)