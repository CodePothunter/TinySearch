"""
FAISS-based vector indexer
"""
from typing import List, Dict, Any, Union, Optional, Tuple
import os
import pickle
import numpy as np
from pathlib import Path
import json

from tinysearch.base import VectorIndexer, TextChunk


class FAISSIndexer(VectorIndexer):
    """
    Vector indexer using Facebook AI Similarity Search (FAISS)
    """
    
    def __init__(
        self,
        index_type: str = "Flat",
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False
    ):
        """
        Initialize the FAISS indexer
        
        Args:
            index_type: FAISS index type ("Flat", "IVF", "HNSW")
            metric: Distance metric ("cosine", "l2", "ip")
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search for IVF index
            use_gpu: Whether to use GPU for search (if available)
        """
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        
        # Will be initialized later
        self.index = None
        self.dimension = None
        self.texts = []
        self.ids_map = {}
    
    def build(self, vectors: List[List[float]], texts: List[TextChunk]) -> None:
        """
        Build the index from vectors and their corresponding text chunks
        
        Args:
            vectors: List of embedding vectors
            texts: List of TextChunk objects corresponding to the vectors
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss. "
                "Please install it with: pip install faiss-cpu or pip install faiss-gpu"
            )
        
        if not vectors:
            raise ValueError("Cannot build index with empty vectors")
        
        if len(vectors) != len(texts):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) does not match "
                f"number of text chunks ({len(texts)})"
            )
        
        # Convert to numpy array
        np_vectors = np.array(vectors).astype('float32')
        self.dimension = np_vectors.shape[1]
        
        # Normalize vectors if using cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(np_vectors)
        
        # Create index
        self.index = self._create_index()
        
        # Add vectors to index
        self.index.add(np_vectors)
        
        # Store text chunks
        self.texts = texts
        
        # Create ID mapping
        self.ids_map = {i: i for i in range(len(texts))}
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for vectors similar to the query vector
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
        Returns:
            List of dictionaries containing text chunks, similarity scores, and embedding vectors
        """
        if self.index is None:
            raise ValueError("Index has not been built yet")
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss. "
                "Please install it with: pip install faiss-cpu or pip install faiss-gpu"
            )
        # Convert query vector to numpy array
        query_np = np.array([query_vector]).astype('float32')
        # Normalize query vector if using cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_np)
        # Search index
        distances, indices = self.index.search(query_np, top_k)
        # Convert to result format
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            # Skip invalid indices (can happen if there are fewer results than top_k)
            if idx < 0 or idx >= len(self.texts):
                continue
            # Map to original text chunk
            text_idx = self.ids_map[idx]
            text_chunk = self.texts[text_idx]
            # Convert distance to similarity score
            if self.metric == "cosine" or self.metric == "ip":
                # For cosine and inner product, higher is better, and maximum is 1
                similarity = float(distance)
                if self.metric == "cosine":
                    # Cosine distance in FAISS is actually 1 - cosine similarity
                    similarity = 1 - similarity
            else:
                # For L2, lower is better, so invert and normalize roughly
                similarity = 1.0 / (1.0 + float(distance))
            # 获取embedding向量
            embedding = self.index.reconstruct(int(idx))
            results.append({
                "text": text_chunk.text,
                "metadata": text_chunk.metadata,
                "score": similarity,
                "embedding": embedding
            })
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the index to disk
        
        Args:
            path: Path to save the index to
        """
        if self.index is None:
            raise ValueError("Index has not been built yet")
        
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss. "
                "Please install it with: pip install faiss-cpu or pip install faiss-gpu"
            )
        
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        # Create directory if not exists
        index_dir = path.with_suffix('')
        os.makedirs(index_dir, exist_ok=True)
        
        # Save index
        index_path = index_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata and text chunks
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "use_gpu": self.use_gpu,
            "ids_map": self.ids_map
        }
        
        with open(index_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Save text chunks
        texts_data = [(chunk.text, chunk.metadata) for chunk in self.texts]
        with open(index_dir / "texts.pkl", "wb") as f:
            pickle.dump(texts_data, f)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the index from disk
        
        Args:
            path: Path to load the index from
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss. "
                "Please install it with: pip install faiss-cpu or pip install faiss-gpu"
            )
        
        path = Path(path)
        
        # Handle both directory and file paths
        index_dir = path if path.is_dir() else path.with_suffix('')
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        
        # Load index
        index_path = index_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = index_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]
        self.metric = metadata["metric"]
        self.nlist = metadata["nlist"]
        self.nprobe = metadata["nprobe"]
        self.use_gpu = metadata["use_gpu"]
        self.ids_map = {int(k): v for k, v in metadata["ids_map"].items()}
        
        # Load text chunks
        texts_path = index_dir / "texts.pkl"
        if not texts_path.exists():
            raise FileNotFoundError(f"Texts file not found: {texts_path}")
        
        with open(texts_path, "rb") as f:
            texts_data = pickle.load(f)
        
        self.texts = [TextChunk(text, metadata) for text, metadata in texts_data]
        
        # Set the number of probes for IVF index
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        
        # Move to GPU if requested and available
        if self.use_gpu:
            try:
                self._move_index_to_gpu()
            except Exception as e:
                print(f"Failed to move index to GPU: {e}")
    
    def _create_index(self) -> Any:
        """
        Create a FAISS index based on the specified parameters
        
        Returns:
            FAISS index object
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss. "
                "Please install it with: pip install faiss-cpu or pip install faiss-gpu"
            )
        
        # Set the metric
        if self.metric == "cosine":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            metric_type = faiss.METRIC_L2
        elif self.metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Create the index
        if self.index_type == "Flat":
            index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" or self.metric == "ip" else faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "IVF":
            # IVF requires a training set, so we create a flat index first
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" or self.metric == "ip" else faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_type)
            # Note: IndexIVFFlat needs to be trained before use, but we'll handle that in build()
        
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.dimension, 32, metric_type)
            # 32 is the number of connections per point
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Check if GPU support is available, if requested
        if self.use_gpu:
            # Try to move to GPU
            gpu_index = self._move_index_to_gpu(index)
            # If the index was moved to GPU, use it; otherwise use CPU index
            if gpu_index is not index:
                return gpu_index
        
        return index
    
    def _move_index_to_gpu(self, index=None) -> Any:
        """
        Move the index to GPU if available and requested
        
        This method attempts to move the FAISS index to GPU if:
        1. GPU usage is requested (use_gpu=True)
        2. FAISS has GPU support
        3. A GPU is available in the system
        
        If any condition fails, it will gracefully fall back to CPU.
        
        Args:
            index: FAISS index object (if None, use self.index)
            
        Returns:
            FAISS index object (on GPU or CPU)
        """
        # If GPU not requested, return immediately
        if not self.use_gpu:
            return index or self.index
        
        # Get the target index
        target_index = index or self.index
        
        try:
            # Import FAISS
            import faiss
            
            # Check if FAISS is built with GPU support by checking for necessary functions
            if not hasattr(faiss, 'get_num_gpus'):
                print("FAISS GPU support not detected (get_num_gpus not found)")
                print("For GPU acceleration, install faiss-gpu package")
                self.use_gpu = False
                return target_index
            
            # Check if GPUs are available
            gpu_count = faiss.get_num_gpus()
            if gpu_count <= 0:
                print(f"No GPUs detected by FAISS (get_num_gpus returned {gpu_count})")
                self.use_gpu = False
                return target_index
            
            # Try to create GPU resources and move index to GPU
            try:
                # Check if GPU resource creation is available
                gpu_resources_fn = getattr(faiss, 'StandardGpuResources', None)
                index_to_gpu_fn = getattr(faiss, 'index_cpu_to_gpu', None)
                
                if gpu_resources_fn is None or index_to_gpu_fn is None:
                    print("FAISS GPU functions not available")
                    self.use_gpu = False
                    return target_index
                
                # Create GPU resources if not already created
                if not hasattr(self, '_gpu_resources'):
                    # only work for faiss-gpu
                    self._gpu_resources = faiss.StandardGpuResources() # type: ignore
                
                # Move index to GPU
                # only work for faiss-gpu
                gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, target_index) # type: ignore
                
                print("Successfully moved index to GPU")
                return gpu_index
                
            except Exception as e:
                print(f"Error moving index to GPU, falling back to CPU: {e}")
                return target_index
            
        except ImportError:
            print("FAISS not installed. To use FAISS, install faiss-cpu or faiss-gpu")
            return target_index
        except Exception as e:
            print(f"Unexpected error with FAISS: {e}")
            return target_index
    
    def add(self, vectors: List[List[float]], texts: List[TextChunk]) -> None:
        """
        Add new vectors to the index
        
        Args:
            vectors: List of embedding vectors to add
            texts: List of TextChunk objects corresponding to the vectors
        """
        if self.index is None:
            # If index doesn't exist yet, build it
            return self.build(vectors, texts)
        
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss. "
                "Please install it with: pip install faiss-cpu or pip install faiss-gpu"
            )
        
        if not vectors:
            return
        
        if len(vectors) != len(texts):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) does not match "
                f"number of text chunks ({len(texts)})"
            )
        
        # Convert to numpy array
        np_vectors = np.array(vectors).astype('float32')
        
        # Verify dimensions match
        if np_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension ({np_vectors.shape[1]}) does not match "
                f"index dimension ({self.dimension})"
            )
        
        # Normalize vectors if using cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(np_vectors)
        
        # Get current index size
        current_size = self.index.ntotal
        
        # Add vectors to index
        self.index.add(np_vectors)
        
        # Update ID mapping
        new_ids = {current_size + i: current_size + i for i in range(len(texts))}
        self.ids_map.update(new_ids)
        
        # Update text chunks
        self.texts.extend(texts) 