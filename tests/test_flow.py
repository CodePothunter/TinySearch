"""
Tests for the FlowController component
"""
import os
import pytest
import tempfile
from pathlib import Path

from tinysearch.base import Embedder, VectorIndexer, TextChunk
from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.query.template import TemplateQueryEngine
from tinysearch.flow.controller import FlowController
from tinysearch.flow.hot_update import HotUpdateManager


class MockEmbedder(Embedder):
    """Mock embedder for testing"""
    
    def embed(self, texts):
        # Return fixed-size vectors (2D) for simplicity
        return [[1.0, 0.0] for _ in texts]


class MockIndexer(VectorIndexer):
    """Mock indexer for testing"""
    
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.saved_path = None
        self.loaded_path = None
    
    def build(self, vectors, texts):
        self.vectors.extend(vectors)
        self.texts.extend(texts)
    
    def search(self, query_vector, top_k=5):
        results = []
        for i, text in enumerate(self.texts[:top_k]):
            results.append({
                "chunk": text,
                "score": 1.0 - (i * 0.1),
                "text": text.text
            })
        return results
    
    def save(self, path):
        self.saved_path = path
    
    def load(self, path):
        self.loaded_path = path


class TestFlowController:
    """Test suite for FlowController"""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Create a temporary directory with sample files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sample text file
            sample_file = Path(temp_dir) / "sample.txt"
            with open(sample_file, "w") as f:
                f.write("This is a test document.\n")
                f.write("It contains multiple lines.\n")
                f.write("We will use it to test the FlowController.\n")
            
            # Create a subdirectory with another file
            subdir = Path(temp_dir) / "subdir"
            os.makedirs(subdir, exist_ok=True)
            
            subdir_file = subdir / "another.txt"
            with open(subdir_file, "w") as f:
                f.write("This is another test document.\n")
                f.write("It is in a subdirectory.\n")
            
            yield temp_dir
    
    @pytest.fixture
    def flow_controller(self):
        """Create a flow controller with mock components for testing"""
        adapter = TextAdapter()
        splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        embedder = MockEmbedder()
        indexer = MockIndexer()
        query_engine = TemplateQueryEngine(indexer=indexer, embedder=embedder)
        
        config = {
            "flow": {
                "use_cache": True,
                "cache_dir": ".test_cache"
            },
            "query_engine": {
                "top_k": 3
            }
        }
        
        controller = FlowController(
            data_adapter=adapter,
            text_splitter=splitter,
            embedder=embedder,
            indexer=indexer,
            query_engine=query_engine,
            config=config
        )
        
        yield controller
        
        # Clean up cache directory after tests
        if Path(".test_cache").exists():
            import shutil
            shutil.rmtree(".test_cache")
    
    def test_build_index_single_file(self, flow_controller, sample_data_dir):
        """Test building an index from a single file"""
        sample_file = Path(sample_data_dir) / "sample.txt"
        flow_controller.build_index(sample_file)
        
        # Check if chunks were created
        assert len(flow_controller.indexer.texts) > 0
        
        # Each chunk should have metadata with the source
        for chunk in flow_controller.indexer.texts:
            assert "source" in chunk.metadata
            assert str(sample_file) == chunk.metadata["source"]
    
    def test_build_index_directory(self, flow_controller, sample_data_dir):
        """Test building an index from a directory"""
        flow_controller.build_index(sample_data_dir)
        
        # Check if chunks were created from multiple files
        assert len(flow_controller.indexer.texts) > 0
        
        # Verify chunks come from different files
        sources = {chunk.metadata["source"] for chunk in flow_controller.indexer.texts}
        assert len(sources) >= 2
    
    def test_save_and_load_index(self, flow_controller, sample_data_dir):
        """Test saving and loading an index"""
        # Build the index
        flow_controller.build_index(sample_data_dir)
        
        # Save the index
        index_path = Path(".test_cache") / "test_index.faiss"
        flow_controller.save_index(index_path)
        
        # Verify index was saved
        assert flow_controller.indexer.saved_path == index_path
        
        # Load the index
        flow_controller.load_index(index_path)
        
        # Verify index was loaded
        assert flow_controller.indexer.loaded_path == index_path
    
    def test_query(self, flow_controller, sample_data_dir):
        """Test querying the index"""
        # Build the index
        flow_controller.build_index(sample_data_dir)
        
        # Perform a query
        results = flow_controller.query("test query")
        
        # Check if results were returned
        assert len(results) > 0
        assert "score" in results[0]
        assert "chunk" in results[0]
    
    def test_caching(self, flow_controller, sample_data_dir):
        """Test that caching works"""
        sample_file = Path(sample_data_dir) / "sample.txt"
        
        # Process the file and it should be cached
        flow_controller.process_file(sample_file)
        
        # Check if the file is in processed files
        assert str(sample_file) in flow_controller.processed_files
        
        # Get initial number of chunks
        initial_chunk_count = len(flow_controller.indexer.texts)
        
        # Clear the indexer texts for testing
        flow_controller.indexer.texts = []
        
        # Process again, should load from cache and not re-process
        flow_controller.process_file(sample_file)
        
        # Check if chunks were loaded correctly
        assert len(flow_controller.indexer.texts) > 0
        
        # Force reprocess
        flow_controller.indexer.texts = []
        flow_controller.process_file(sample_file, force_reprocess=True)
        
        # Check if chunks were regenerated
        assert len(flow_controller.indexer.texts) == initial_chunk_count
    
    def test_clear_cache(self, flow_controller, sample_data_dir):
        """Test clearing the cache"""
        # Process some files to create cache
        flow_controller.build_index(sample_data_dir)
        
        # Verify cache exists
        assert len(flow_controller.processed_files) > 0
        
        # Clear cache
        flow_controller.clear_cache()
        
        # Verify cache is cleared
        assert len(flow_controller.processed_files) == 0
    
    def test_get_stats(self, flow_controller, sample_data_dir):
        """Test getting statistics"""
        # Process some files
        flow_controller.build_index(sample_data_dir)
        
        # Get stats
        stats = flow_controller.get_stats()
        
        # Check stats
        assert "processed_files_count" in stats
        assert "cache_enabled" in stats
        assert stats["cache_enabled"] is True
        assert stats["processed_files_count"] > 0 
        
    def test_hot_update_functionality(self, flow_controller, sample_data_dir):
        """Test hot update functionality using the stub implementation"""
        # Start hot update with the sample directory
        flow_controller.start_hot_update(
            watch_paths=[sample_data_dir],
            file_extensions=[".txt"],
            recursive=True
        )
        
        # Verify hot update is active
        assert flow_controller.is_hot_update_active()
        
        # Verify that the hot update manager is our stub implementation
        assert isinstance(flow_controller._hot_update_manager, HotUpdateManager)
        
        # Stop hot update
        flow_controller.stop_hot_update()
        
        # Verify hot update is inactive
        assert not flow_controller.is_hot_update_active()
        
    def test_hot_update_manager_interactions(self, flow_controller, sample_data_dir):
        """Test interactions with the hot update manager"""
        # Start hot update with specific parameters
        watch_paths = [sample_data_dir]
        file_extensions = [".txt"]
        
        flow_controller.start_hot_update(
            watch_paths=watch_paths,
            file_extensions=file_extensions,
            recursive=True
        )
        
        # Test adding a new watch path
        new_path = Path(sample_data_dir) / "new_path"
        os.makedirs(new_path, exist_ok=True)
        
        flow_controller.add_watch_path(new_path)
        
        # Verify the path was added
        assert str(new_path) in flow_controller._hot_update_manager.watch_paths
        
        # Test removing a watch path
        flow_controller.remove_watch_path(new_path)
        
        # Verify the path was removed
        assert str(new_path) not in flow_controller._hot_update_manager.watch_paths
        
    def test_add_watch_path_recursive_parameter(self, flow_controller, sample_data_dir):
        """Test add_watch_path with recursive parameter"""
        # Start hot update
        flow_controller.start_hot_update(
            watch_paths=[sample_data_dir],
            recursive=False  # Default is not recursive
        )
        
        # Test adding with recursive parameter
        new_path = Path(sample_data_dir) / "another_path"
        os.makedirs(new_path, exist_ok=True)
        
        # Add with recursive=True
        flow_controller.add_watch_path(new_path, recursive=True)
        
        # Stop the hot update manager to avoid cleanup issues
        flow_controller.stop_hot_update() 