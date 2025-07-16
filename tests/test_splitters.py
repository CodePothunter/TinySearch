"""
Tests for text splitters
"""
import sys
from pathlib import Path
import pytest

# Add the parent directory to sys.path so that we can import tinysearch
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.base import TextSplitter, TextChunk


def test_character_text_splitter_basic():
    """Test basic functionality of CharacterTextSplitter"""
    # Create a splitter
    splitter = CharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=2,
        separator=" "
    )
    
    # Test text
    text = "This is a test of the text splitter functionality"
    
    # Split the text
    chunks = splitter.split([text])
    
    # Check the results
    assert len(chunks) > 1
    assert all(isinstance(chunk_item, TextChunk) for chunk_item in chunks)
    assert all(len(chunk_item.text) <= 12 for chunk_item in chunks)  # chunk_size + longest word


def test_character_text_splitter_overlap():
    """Test overlap in CharacterTextSplitter"""
    # Create a splitter with overlap
    splitter = CharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=5,
        separator=" "
    )
    
    # Test text
    text = "one two three four five six seven"
    
    # Split the text
    chunks = splitter.split([text])
    
    # Print chunks for debugging
    print("Chunks for overlap test:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: '{chunk.text}'")
    
    # Check that we have overlapping content between chunks
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1].text
        curr_chunk = chunks[i].text
        
        # Find some overlapping content - at least one word should overlap
        overlap_found = False
        for word in prev_chunk.split():
            if word in curr_chunk:
                overlap_found = True
                break
        
        assert overlap_found, f"No overlap found between '{prev_chunk}' and '{curr_chunk}'"


def test_character_text_splitter_no_separator():
    """Test CharacterTextSplitter with no separator"""
    # Create a splitter with no separator
    splitter = CharacterTextSplitter(
        chunk_size=5,
        chunk_overlap=2,
        separator=""
    )
    
    # Test text
    text = "abcdefghijklmnopqrstuvwxyz"
    
    # Split the text
    chunks = splitter.split([text])
    
    # Print chunks for debugging
    print("Chunks for no separator test:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: '{chunk.text}'")
    
    # Check that chunks have the right size and overlap correctly
    # Since we have overlap=2 and chunk_size=5, each chunk after the first should start
    # with the last 2 characters of the previous chunk
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1].text
        curr_chunk = chunks[i].text
        
        # The current chunk should start with the last 2 chars of previous chunk
        assert curr_chunk.startswith(prev_chunk[-2:]), f"Chunk {i} doesn't start with overlap from previous chunk"
        
        # Each chunk except maybe the last should have length = chunk_size
        if i < len(chunks) - 1:
            assert len(curr_chunk) == 5, f"Chunk {i} has incorrect length: {len(curr_chunk)}"


def test_character_text_splitter_metadata():
    """Test metadata handling in CharacterTextSplitter"""
    # Create a splitter
    splitter = CharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0
    )
    
    # Test texts with metadata
    texts = ["Text one", "Text two"]
    metadata = [{"source": "doc1"}, {"source": "doc2"}]
    
    # Split the texts
    chunks = splitter.split(texts, metadata)
    
    # Check the results
    assert len(chunks) == 2
    assert chunks[0].metadata["source"] == "doc1"
    assert chunks[1].metadata["source"] == "doc2"
    assert "chunk_index" in chunks[0].metadata
    assert "total_chunks" in chunks[0].metadata


def test_character_text_splitter_empty_text():
    """Test CharacterTextSplitter with empty text"""
    # Create a splitter
    splitter = CharacterTextSplitter()
    
    # Test with empty text
    chunks = splitter.split([""])
    
    # Should produce one empty chunk
    assert len(chunks) == 1
    assert chunks[0].text == ""


def test_character_text_splitter_error_cases():
    """Test error cases for CharacterTextSplitter"""
    # Invalid chunk_size and chunk_overlap relationship
    with pytest.raises(ValueError):
        CharacterTextSplitter(chunk_size=10, chunk_overlap=10)
    
    with pytest.raises(ValueError):
        CharacterTextSplitter(chunk_size=10, chunk_overlap=15)
    
    # Metadata length mismatch
    splitter = CharacterTextSplitter()
    with pytest.raises(ValueError):
        splitter.split(["text1", "text2"], [{"source": "doc1"}])


def test_splitter_interface():
    """Test TextSplitter interface"""
    # Ensure CharacterTextSplitter implements TextSplitter interface
    assert issubclass(CharacterTextSplitter, TextSplitter)
    
    # Create an instance
    splitter = CharacterTextSplitter()
    
    # Check that it has the required methods
    assert hasattr(splitter, "split")
    assert callable(splitter.split) 