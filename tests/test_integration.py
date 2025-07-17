"""
Integration tests for TinySearch components
These tests verify the interaction between multiple components
"""
import os
import pytest
from pathlib import Path
import tempfile
import numpy as np
from typing import List, Dict, Any

from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.flow.controller import FlowController
from tinysearch.query.template import TemplateQueryEngine
from tinysearch.base import TextChunk

# Import from local conftest file
from tests.conftest import MockEmbedder, MockIndexer


class TestComponentInteractions:
    """Test interactions between components"""
    
    @pytest.fixture
    def test_components(self):
        """Create test components for integration tests"""
        adapter = TextAdapter()
        splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        embedder = MockEmbedder(vector_size=10, similarity_pattern=True)
        indexer = MockIndexer()
        query_engine = TemplateQueryEngine(indexer=indexer, embedder=embedder)
        
        return {
            "adapter": adapter,
            "splitter": splitter,
            "embedder": embedder,
            "indexer": indexer,
            "query_engine": query_engine
        }
    
    def test_adapter_to_splitter_integration(self, test_components, sample_files):
        """Test integration of DataAdapter with TextSplitter"""
        adapter = test_components["adapter"]
        splitter = test_components["splitter"]
        
        # Process a sample file
        file_path = sample_files["files"][0]
        
        # Extract text using adapter
        texts = adapter.extract(file_path)
        assert len(texts) == 1
        
        # Split text using splitter
        chunks = splitter.split(texts, [{"source": str(file_path)}])
        
        # Verify integration results
        assert len(chunks) > 0
        # Each chunk should be <= chunk_size (plus any overflow for long words)
        for chunk in chunks:
            if len(chunk.text) > splitter.chunk_size:
                # If chunk exceeds size, it should contain a long unsplittable word
                words = chunk.text.split()
                assert any(len(word) > splitter.chunk_size - 20 for word in words)
        
        # Source metadata should be preserved
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == str(file_path)
    
    def test_splitter_to_embedder_integration(self, test_components, sample_texts):
        """Test integration of TextSplitter with Embedder"""
        splitter = test_components["splitter"]
        embedder = test_components["embedder"]
        
        # Split text
        chunks = splitter.split(sample_texts)
        assert len(chunks) > 0
        
        # Get text from chunks
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        vectors = embedder.embed(chunk_texts)
        
        # Verify integration results
        assert len(vectors) == len(chunks)
        assert len(vectors[0]) == embedder.vector_size
        
        # Verify similar texts get similar vectors when similarity_pattern=True
        if embedder.similarity_pattern:
            # Find chunks with similar content
            similar_pairs = []
            for i in range(len(chunks)):
                for j in range(i+1, len(chunks)):
                    # Simple similarity check based on shared words
                    words_i = set(chunks[i].text.lower().split())
                    words_j = set(chunks[j].text.lower().split())
                    shared_words = len(words_i.intersection(words_j))
                    if shared_words >= 2:  # Consider similar if sharing at least 2 words
                        similar_pairs.append((i, j))
            
            # There should be at least one similar pair in sample_texts
            if similar_pairs:
                # Calculate vector similarity (cosine) for similar pairs
                for i, j in similar_pairs:
                    similarity = self._cosine_similarity(vectors[i], vectors[j])
                    # Similar texts should have embeddings with similarity > 0.5
                    assert similarity > 0.5, f"Similar texts should have similar vectors: {chunks[i].text} and {chunks[j].text}"
    
    def test_embedder_to_indexer_integration(self, test_components, sample_texts):
        """Test integration of Embedder with VectorIndexer"""
        embedder = test_components["embedder"]
        indexer = test_components["indexer"]
        
        # Create text chunks
        chunks = [TextChunk(text) for text in sample_texts]
        
        # Generate embeddings
        vectors = embedder.embed([chunk.text for chunk in chunks])
        
        # Build index
        indexer.build(vectors, chunks)
        
        # Verify integration results
        assert len(indexer.vectors) == len(vectors)
        assert len(indexer.texts) == len(chunks)
    
    def test_full_pipeline_integration(self, test_components, sample_files):
        """Test full pipeline integration from file to query results"""
        # Create a flow controller
        config = {
            "flow": {
                "use_cache": False
            },
            "query_engine": {
                "top_k": 3
            }
        }
        
        controller = FlowController(
            data_adapter=test_components["adapter"],
            text_splitter=test_components["splitter"],
            embedder=test_components["embedder"],
            indexer=test_components["indexer"],
            query_engine=test_components["query_engine"],
            config=config
        )
        
        # Build index from files
        controller.build_index(sample_files["dir"])
        
        # Perform a query that should match sample content
        results = controller.query("sample file test content")
        
        # Verify query results
        assert len(results) > 0
        assert all("score" in result for result in results)
        assert all("text" in result for result in results)
        
        # Scores should be in descending order
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Convert to numpy arrays for easier calculation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)


class TestRealWorldScenarios:
    """Test realistic scenarios with minimal mocking"""
    
    def test_incremental_index_building(self, sample_files):
        """Test incrementally building an index with new documents"""
        # Create components with minimal mocking
        adapter = TextAdapter()
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        embedder = MockEmbedder(vector_size=10)  # Using mock for embedding only
        indexer = MockIndexer()
        query_engine = TemplateQueryEngine(indexer=indexer, embedder=embedder)
        
        config = {"flow": {"use_cache": True, "cache_dir": ".test_incremental_cache"}}
        
        controller = FlowController(
            data_adapter=adapter,
            text_splitter=splitter,
            embedder=embedder,
            indexer=indexer,
            query_engine=query_engine,
            config=config
        )
        
        try:
            # Initially process only one file
            controller.process_file(sample_files["files"][0])
            initial_chunk_count = len(indexer.texts)
            assert initial_chunk_count > 0
            
            # Process a second file incrementally
            controller.process_file(sample_files["files"][1])
            second_chunk_count = len(indexer.texts)
            assert second_chunk_count > initial_chunk_count
            
            # Query should use all documents
            results = controller.query("test content")
            assert len(results) > 0
            
        finally:
            # Clean up cache
            cache_dir = Path(".test_incremental_cache")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
    
    def test_data_integrity_through_pipeline(self, sample_files):
        """Test that data integrity is maintained throughout the pipeline"""
        # Create components with minimal mocking
        adapter = TextAdapter()
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)  # No overlap for easier testing
        embedder = MockEmbedder(vector_size=5, deterministic=True)
        indexer = MockIndexer()
        
        # Extract a single file for tracking
        file_path = sample_files["files"][0]
        texts = adapter.extract(file_path)
        original_content = texts[0]
        
        # Split content
        chunks = splitter.split(texts, [{"source": str(file_path)}])
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Verify we can reconstruct the original text (approximately)
        reconstructed = "".join(chunk_texts)
        # Remove whitespace for comparison since splitter might strip whitespace
        original_normalized = "".join(original_content.split())
        reconstructed_normalized = "".join(reconstructed.split())
        
        # Check if the reconstructed text contains most of the original content
        assert len(reconstructed_normalized) >= 0.9 * len(original_normalized)
        
        # Generate embeddings and build index
        vectors = embedder.embed(chunk_texts)
        indexer.build(vectors, chunks)
        
        # Search for content from the original text
        # Extract a distinctive phrase from the original content
        search_phrase = original_content.split(".")[0]  # First sentence
        
        # Use query engine directly for more control
        query_engine = TemplateQueryEngine(indexer=indexer, embedder=embedder)
        query_results = query_engine.retrieve(search_phrase)
        
        # The top result should contain parts of our search phrase
        assert len(query_results) > 0
        top_result_text = query_results[0]["text"]
        
        # Check if at least some words from search phrase appear in the result
        search_words = set(search_phrase.lower().split())
        result_words = set(top_result_text.lower().split())
        common_words = search_words.intersection(result_words)
        
        assert len(common_words) > 0, f"Expected some words from '{search_phrase}' in result '{top_result_text}'" 