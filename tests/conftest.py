"""
Pytest configuration file
"""
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so that we can import tinysearch
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary packages here to ensure they are available for all tests
import pytest
import yaml
import json
import numpy as np
import tempfile
import random
from typing import List, Dict, Any, Optional

from tinysearch.base import Embedder, VectorIndexer, TextChunk

# Define helper methods for the FlowController in tests
def mock_add_watch_path(self, path, recursive=None):
    """Add a path to watch for changes"""
    if self._hot_update_manager:
        self._hot_update_manager.add_watch_path(path, recursive)

def mock_remove_watch_path(self, path):
    """Remove a path from being watched"""
    if self._hot_update_manager:
        self._hot_update_manager.remove_watch_path(path)


class MockEmbedder(Embedder):
    """Improved mock embedder for testing"""
    
    def __init__(self, vector_size: int = 5, deterministic: bool = True, similarity_pattern: bool = False):
        """
        Initialize the mock embedder
        
        Args:
            vector_size: Size of the embedding vectors
            deterministic: Whether to generate deterministic embeddings based on text content
            similarity_pattern: If True, similar texts get similar vectors
        """
        self.vector_size = vector_size
        self.deterministic = deterministic
        self.similarity_pattern = similarity_pattern
        self.call_count = 0
        self.embedded_texts = []
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to mock embedding vectors
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as float lists
        """
        self.call_count += 1
        self.embedded_texts.extend(texts)
        
        results = []
        for text in texts:
            if self.deterministic:
                # Generate a deterministic vector based on the text
                vector = self._text_to_deterministic_vector(text)
            else:
                # Generate a random vector
                vector = [random.random() for _ in range(self.vector_size)]
                
            results.append(vector)
        
        return results
    
    def _text_to_deterministic_vector(self, text: str) -> List[float]:
        """Convert text to a deterministic vector based on content"""
        if not text:
            return [0.0] * self.vector_size
            
        # Simple hashing strategy for deterministic vectors
        hash_val = hash(text)
        random.seed(hash_val)
        
        if self.similarity_pattern:
            # For texts containing similar words, create similar vectors
            words = text.lower().split()
            # Base vector on word presence
            vector = [0.0] * self.vector_size
            for i, word in enumerate(words):
                word_hash = hash(word)
                random.seed(word_hash)
                word_vec = [random.random() for _ in range(self.vector_size)]
                # Add word vector to total
                for j in range(self.vector_size):
                    vector[j] += word_vec[j]
            
            # Normalize
            magnitude = sum(v * v for v in vector) ** 0.5
            if magnitude > 0:
                vector = [v / magnitude for v in vector]
            
            return vector
        else:
            # Simple random but deterministic vector
            return [random.random() for _ in range(self.vector_size)]


class MockIndexer(VectorIndexer):
    """Improved mock indexer for testing"""
    
    def __init__(self, track_operations: bool = True):
        """
        Initialize the mock indexer
        
        Args:
            track_operations: Whether to track operations for testing
        """
        self.vectors = []
        self.texts = []
        self.saved_path = None
        self.loaded_path = None
        self.track_operations = track_operations
        self.build_count = 0
        self.search_count = 0
        self.search_history = []
    
    def build(self, vectors: List[List[float]], texts: List[TextChunk]) -> None:
        """Build the index from vectors and texts"""
        self.vectors.extend(vectors)
        self.texts.extend(texts)
        if self.track_operations:
            self.build_count += 1
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the index for vectors similar to the query vector"""
        if self.track_operations:
            self.search_count += 1
            self.search_history.append({"query_vector": query_vector, "top_k": top_k})
        
        # Simple similarity calculation - not efficient but good for testing
        results = []
        for i, (vector, text) in enumerate(zip(self.vectors[:top_k], self.texts[:top_k])):
            # Calculate cosine similarity (simplified)
            score = 1.0 - (i * 0.1)  # Simple scoring for tests
            results.append({
                "chunk": text,
                "score": score,
                "text": text.text
            })
        
        return results
    
    def save(self, path: Path) -> None:
        """Save the index to disk"""
        self.saved_path = path
    
    def load(self, path: Path) -> None:
        """Load the index from disk"""
        self.loaded_path = path


@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for testing"""
    return [
        "This is the first test document.",
        "This is the second test document with more words.",
        "This is the third test document with even more words for testing.",
        "This document has different content than the others."
    ]


@pytest.fixture
def sample_files():
    """Fixture providing temporary sample files for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample files with content
        files = []
        for i in range(3):
            file_path = Path(temp_dir) / f"sample_{i}.txt"
            with open(file_path, "w") as f:
                f.write(f"This is sample file {i} with test content.\n")
                f.write(f"It has multiple lines and serves as test data.\n")
            files.append(file_path)
        
        # Create a subdirectory with a file
        subdir = Path(temp_dir) / "subdir"
        os.makedirs(subdir, exist_ok=True)
        
        subdir_file = subdir / "subdir_file.txt"
        with open(subdir_file, "w") as f:
            f.write("This is a file in a subdirectory.\n")
        files.append(subdir_file)
        
        # Return the directory path and list of files
        yield {"dir": temp_dir, "files": files}


# Apply mocks before any tests run
@pytest.fixture(autouse=True)
def setup_tests(monkeypatch):
    """Setup test environment"""
    # Import controller after we've added the stub HotUpdateManager
    from tinysearch.flow.controller import FlowController
    
    # Patch FlowController methods to work with our stub
    monkeypatch.setattr(FlowController, "add_watch_path", mock_add_watch_path)
    monkeypatch.setattr(FlowController, "remove_watch_path", mock_remove_watch_path) 