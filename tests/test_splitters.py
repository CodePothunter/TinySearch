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


def test_character_text_splitter_complex_separators():
    """Test CharacterTextSplitter with complex separators and edge cases"""
    # Create a splitter with newline separator and small chunk size
    splitter = CharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=5,
        separator="\n",
        keep_separator=True
    )
    
    # Text with multiple newlines and edge cases
    text = "Line1\nLine2\nLine3VeryLongWordThatExceedsChunkSize\nLine4\nLine5"
    
    # Split the text
    chunks = splitter.split([text])
    
    # Print chunks for debugging
    print("\nComplex separators test chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: '{chunk.text}'")
    
    # Verify behavior:
    # 1. All chunks should have length <= chunk_size or be a single word that exceeds chunk_size
    # 2. Long words should be handled correctly
    # 3. Separators should be kept as specified
    # 4. Overlap should be maintained between chunks
    
    # The long word may be split across multiple chunks
    long_word_parts_found = False
    for chunk in chunks:
        if "VeryLongWordTh" in chunk.text or "ordThatExceedsChunkSize" in chunk.text:
            long_word_parts_found = True
            break
    
    assert long_word_parts_found, "Parts of the long word should be present in the chunks"
    
    # Each chunk should either be <= chunk_size or contain part of an unsplittable long word
    for chunk in chunks:
        if len(chunk.text) > 20:  # chunk_size
            # If a chunk is longer than chunk_size, it should contain part of the long word
            assert ("VeryLongWord" in chunk.text or 
                   "ThatExceedsChunkSize" in chunk.text or 
                   "ordThatExceeds" in chunk.text), \
                f"Chunk exceeds size but doesn't contain part of the long word: '{chunk.text}'"


def test_character_text_splitter_boundary_handling():
    """Test CharacterTextSplitter's handling of boundary cases"""
    # Create a splitter with space separator
    splitter = CharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
        separator=" "
    )
    
    # Text with exactly chunk_size characters
    text_exact = "1234567890"
    chunks_exact = splitter.split([text_exact])
    assert len(chunks_exact) == 1, "Text exactly equal to chunk_size should be one chunk"
    assert chunks_exact[0].text == "1234567890"
    
    # Text with exactly chunk_size+1 characters
    text_plus_one = "12345678901"
    chunks_plus_one = splitter.split([text_plus_one])
    assert len(chunks_plus_one) == 2, "Text one char longer than chunk_size should be two chunks"
    
    # Text with spaces at boundaries
    text_spaces = "12345 67890 12345"
    chunks_spaces = splitter.split([text_spaces])
    print("\nBoundary test chunks:")
    for i, chunk in enumerate(chunks_spaces):
        print(f"Chunk {i}: '{chunk.text}'")
    
    # Check that chunks are split at word boundaries
    for chunk in chunks_spaces:
        # Chunk should not start or end with space when using space separator
        assert not chunk.text.startswith(" "), f"Chunk should not start with space: '{chunk.text}'"
        # In case strip_whitespace is True (default), check if spaces are properly stripped
        if splitter.strip_whitespace:
            assert not chunk.text.endswith(" "), f"Chunk should not end with space: '{chunk.text}'"


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