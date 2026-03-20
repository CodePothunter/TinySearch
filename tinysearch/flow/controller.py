"""
Flow controller implementation for TinySearch
"""
import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple, cast, Callable

from tinysearch.base import DataAdapter, TextSplitter, Embedder, VectorIndexer, QueryEngine, Retriever
from tinysearch.base import TextChunk, FlowController as FlowControllerBase
from tinysearch.flow.hot_update import HotUpdateManager
from tinysearch.query.hybrid import HybridQueryEngine
from tinysearch.retrievers.vector_retriever import VectorRetriever
from tinysearch.utils.file_discovery import iter_input_files


class FlowController(FlowControllerBase):
    """
    Orchestrates the flow of data through the pipeline.
    Manages the entire data processing from ingestion to query handling.
    """
    
    def __init__(
        self,
        data_adapter: DataAdapter,
        text_splitter: TextSplitter,
        embedder: Embedder,
        indexer: VectorIndexer,
        query_engine: QueryEngine,
        config: Dict[str, Any]
    ):
        """
        Initialize the flow controller with all required components
        
        Args:
            data_adapter: Component for data extraction
            text_splitter: Component for text chunking
            embedder: Component for generating embeddings
            indexer: Component for vector indexing and search
            query_engine: Component for query processing
            config: Configuration dictionary
        """
        self.data_adapter = data_adapter
        self.text_splitter = text_splitter
        self.embedder = embedder
        self.indexer = indexer
        self.query_engine = query_engine
        self.config = config
        
        # Setup cache if enabled
        self.use_cache = self.config.get("flow", {}).get("use_cache", False)
        self.cache_dir = Path(self.config.get("flow", {}).get("cache_dir", ".cache"))
        
        if self.use_cache and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processed files tracking
        self.processed_files: Set[str] = set()
        self._load_processed_files()
        
        # Initialize hot update manager
        self._hot_update_manager = None
    
    def _load_processed_files(self) -> None:
        """Load the list of already processed files from cache"""
        if self.use_cache:
            cache_file = self.cache_dir / "processed_files.json"
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.processed_files = set(json.load(f))
    
    def _save_processed_files(self) -> None:
        """Save the list of processed files to cache"""
        if self.use_cache:
            cache_file = self.cache_dir / "processed_files.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(list(self.processed_files), f)
    
    def _get_cache_path(self, file_path: Union[str, Path]) -> Path:
        """Generate a cache file path for a given input file"""
        file_path_str = str(file_path)
        file_hash = str(hash(file_path_str))
        return self.cache_dir / f"{file_hash}.pkl"
    
    def _cache_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if cache exists for a given file"""
        if not self.use_cache:
            return False
        
        cache_path = self._get_cache_path(file_path)
        return cache_path.exists()
    
    def _load_from_cache(self, file_path: Union[str, Path]) -> Tuple[List[TextChunk], List[List[float]]]:
        """Load cached chunks and vectors for a file"""
        cache_path = self._get_cache_path(file_path)
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        
        return cached_data["chunks"], cached_data["vectors"]
    
    def _save_to_cache(self, file_path: Union[str, Path], chunks: List[TextChunk], vectors: List[List[float]]) -> None:
        """Save chunks and vectors to cache"""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(file_path)
        with open(cache_path, "wb") as f:
            pickle.dump({"chunks": chunks, "vectors": vectors}, f)
        
        # Add to processed files
        self.processed_files.add(str(file_path))
        self._save_processed_files()
    
    def process_file(self, file_path: Union[str, Path], force_reprocess: bool = False) -> None:
        """
        Process a single file and add it to the index
        
        Args:
            file_path: Path to the file to process
            force_reprocess: If True, reprocess even if file is in cache
        """
        file_path = Path(file_path)
        
        # Check cache first
        if not force_reprocess and self._cache_exists(file_path):
            chunks, vectors = self._load_from_cache(file_path)
        else:
            # Extract text
            texts = self.data_adapter.extract(file_path)
            
            # Create metadata for each text
            metadata = [{"source": str(file_path)} for _ in range(len(texts))]
            
            # Split text into chunks
            chunks = self.text_splitter.split(texts, metadata)
            
            # Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            vectors = self.embedder.embed(chunk_texts)
            
            # Save to cache
            self._save_to_cache(file_path, chunks, vectors)
        
        # Add to index
        self.indexer.build(vectors, chunks)

        # Build retriever indexes for HybridQueryEngine
        self._build_retriever_indexes(chunks)
    
    def process_directory(self, dir_path: Union[str, Path], extensions: Optional[List[str]] = None, 
                         recursive: bool = True, force_reprocess: bool = False) -> None:
        """
        Process all files in a directory
        
        Args:
            dir_path: Path to the directory
            extensions: List of file extensions to process (e.g. ['.txt', '.pdf'])
            recursive: Whether to recursively process subdirectories
            force_reprocess: If True, reprocess even if file is in cache
        """
        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a directory")

        adapter_type = self.config.get("adapter", {}).get("type", "text")
        for file_path in iter_input_files(dir_path, adapter_type=adapter_type, extensions=extensions, recursive=recursive):
            if not force_reprocess and str(file_path) in self.processed_files:
                continue
            self.process_file(file_path, force_reprocess)
    
    def build_index(self, data_path: Union[str, Path], **kwargs) -> None:
        """
        Build the search index from a data file or directory
        
        Args:
            data_path: Path to the data file or directory
            **kwargs: Additional arguments for customizing the build process
                force_reprocess (bool): If True, reprocess even if already processed
                extensions (List[str]): List of file extensions to process
                recursive (bool): Whether to recursively process subdirectories
        """
        data_path = Path(data_path)
        
        force_reprocess = kwargs.get("force_reprocess", False)
        extensions = kwargs.get("extensions")
        recursive = kwargs.get("recursive", True)
        
        if data_path.is_dir():
            self.process_directory(
                data_path, 
                extensions=extensions,
                recursive=recursive,
                force_reprocess=force_reprocess
            )
        else:
            self.process_file(data_path, force_reprocess=force_reprocess)
    
    def _get_hybrid_retrievers(self) -> List[Retriever]:
        """Get retrievers from HybridQueryEngine, if applicable"""
        if isinstance(self.query_engine, HybridQueryEngine):
            return self.query_engine.retrievers
        return []

    def _build_retriever_indexes(self, chunks: List[TextChunk]) -> None:
        """Build indexes for non-vector retrievers and metadata index in HybridQueryEngine"""
        for retriever in self._get_hybrid_retrievers():
            # Skip VectorRetriever - it's already handled by self.indexer.build()
            if isinstance(retriever, VectorRetriever):
                continue
            retriever.build(chunks)

        # Build metadata index for pre-filtering
        if isinstance(self.query_engine, HybridQueryEngine) and self.query_engine.metadata_index is not None:
            self.query_engine.metadata_index.build(chunks)

    def save_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the built index to disk.
        Also saves retriever indexes for HybridQueryEngine.

        Args:
            path: Path to save the index to, if None use the config path
        """
        if path is None:
            path = self.config.get("indexer", {}).get("index_path", "index.faiss")

        # Save the main vector index
        self.indexer.save(cast(Union[str, Path], path))

        # Save non-vector retriever indexes
        self._save_retriever_indexes(Path(str(path)))

    def _save_retriever_indexes(self, base_path: Path) -> None:
        """Save indexes for non-vector retrievers and metadata index inside the FAISS index directory"""
        # FAISS saves into base_path.with_suffix('') (e.g. "index.faiss" → "index/")
        index_dir = base_path.with_suffix('') if base_path.suffix else base_path
        for retriever in self._get_hybrid_retrievers():
            if isinstance(retriever, VectorRetriever):
                continue
            # Derive subdirectory name from retriever class
            retriever_name = type(retriever).__name__.lower().replace("retriever", "")
            retriever_path = index_dir / f"{retriever_name}_index"
            retriever.save(retriever_path)

        # Save metadata index
        if isinstance(self.query_engine, HybridQueryEngine) and self.query_engine.metadata_index is not None:
            self.query_engine.metadata_index.save(index_dir / "metadata_index.json")

    def load_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Load an index from disk.
        Also loads retriever indexes for HybridQueryEngine.

        Args:
            path: Path to load the index from, if None use the config path
        """
        if path is None:
            path = self.config.get("indexer", {}).get("index_path", "index.faiss")

        # Load the main vector index
        self.indexer.load(cast(Union[str, Path], path))

        # Load non-vector retriever indexes
        self._load_retriever_indexes(Path(str(path)))

    def _load_retriever_indexes(self, base_path: Path) -> None:
        """Load indexes for non-vector retrievers and metadata index from the FAISS index directory"""
        index_dir = base_path.with_suffix('') if base_path.suffix else base_path
        for retriever in self._get_hybrid_retrievers():
            if isinstance(retriever, VectorRetriever):
                continue
            retriever_name = type(retriever).__name__.lower().replace("retriever", "")
            retriever_path = index_dir / f"{retriever_name}_index"
            if retriever_path.exists():
                retriever.load(retriever_path)

        # Load metadata index
        if isinstance(self.query_engine, HybridQueryEngine) and self.query_engine.metadata_index is not None:
            metadata_path = index_dir / "metadata_index.json"
            if metadata_path.exists():
                self.query_engine.metadata_index.load(metadata_path)
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Process a query and return relevant chunks
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            **kwargs: Additional arguments for customizing the query process
            
        Returns:
            List of dictionaries containing text chunks and similarity scores
        """
        if top_k is None:
            top_k = self.config.get("query_engine", {}).get("top_k", 5)
            
        # Use cast to ensure type safety for optional top_k
        return self.query_engine.retrieve(query_text, cast(int, top_k), **kwargs)
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        if not self.use_cache or not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        processed_file = self.cache_dir / "processed_files.json"
        if processed_file.exists():
            processed_file.unlink()
        
        self.processed_files.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "processed_files_count": len(self.processed_files),
            "cache_enabled": self.use_cache,
            "index": {}
        }
        
        # Get index stats if available
        if hasattr(self.indexer, "get_stats"):
            stats["index"] = getattr(self.indexer, "get_stats")()
            
        return stats
    
    # Hot-update functionality
    def start_hot_update(
        self,
        watch_paths: List[str],
        file_extensions: Optional[List[str]] = None,
        process_delay: float = 1.0,
        recursive: bool = True,
        on_update_callback: Optional[Callable] = None
    ) -> None:
        """
        Start hot update monitoring
        
        Args:
            watch_paths: List of paths to watch for changes
            file_extensions: List of file extensions to monitor
            process_delay: Delay in seconds before processing changes
            recursive: Whether to watch subdirectories recursively
            on_update_callback: Optional callback function to call after processing
        """
        # Stop existing hot update if running
        if self._hot_update_manager is not None and self._hot_update_manager.is_watching():
            self._hot_update_manager.stop()
            
        self._hot_update_manager = HotUpdateManager(
            flow_controller=self,
            watch_paths=watch_paths,
            file_extensions=file_extensions,
            process_delay=process_delay,
            recursive=recursive,
            on_update_callback=on_update_callback
        )
        
        self._hot_update_manager.start()
    
    def stop_hot_update(self) -> None:
        """
        Stop hot update monitoring
        """
        if self._hot_update_manager is not None:
            self._hot_update_manager.stop()
    
    def is_hot_update_active(self) -> bool:
        """
        Check if hot update monitoring is active
        
        Returns:
            True if hot update is active
        """
        return self._hot_update_manager is not None and self._hot_update_manager.is_watching() 
        
    def add_watch_path(self, path: Union[str, Path], recursive: Optional[bool] = None) -> None:
        """
        Add a path to watch for file changes
        
        Args:
            path: Path to watch
            recursive: Whether to watch subdirectories recursively
        """
        if self._hot_update_manager is not None:
            self._hot_update_manager.add_watch_path(path, recursive)
        else:
            raise RuntimeError("Hot update manager is not initialized. Call start_hot_update first.")
    
    def remove_watch_path(self, path: Union[str, Path]) -> None:
        """
        Remove a path from being watched for file changes
        
        Args:
            path: Path to stop watching
        """
        if self._hot_update_manager is not None:
            self._hot_update_manager.remove_watch_path(path)
        else:
            raise RuntimeError("Hot update manager is not initialized. Call start_hot_update first.") 