"""
Base interfaces for TinySearch modules
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pathlib


class DataAdapter(ABC):
    """
    Interface for adapters that extract text from different data formats.
    Users can implement custom adapters for specific data sources.
    """
    
    @abstractmethod
    def extract(self, filepath: Union[str, pathlib.Path]) -> List[str]:
        """
        Extract text content from the given file
        
        Args:
            filepath: Path to the file to extract text from
            
        Returns:
            List of text strings extracted from the file
        """
        pass


class TextChunk:
    """
    Represents a chunk of text with optional metadata
    """
    
    def __init__(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"TextChunk(text='{self.text[:50]}...', metadata={self.metadata})"


class TextSplitter(ABC):
    """
    Interface for text splitters that chunk text into smaller segments
    """
    
    @abstractmethod
    def split(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[TextChunk]:
        """
        Split texts into chunks
        
        Args:
            texts: List of text strings to split
            metadata: Optional list of metadata dicts corresponding to each text
            
        Returns:
            List of TextChunk objects
        """
        pass


class Embedder(ABC):
    """
    Interface for embedding models that convert text to vectors
    """
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to embedding vectors
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as float lists
        """
        pass


class VectorIndexer(ABC):
    """
    Interface for vector indexers that build and maintain search indices
    """
    
    @abstractmethod
    def build(self, vectors: List[List[float]], texts: List[TextChunk]) -> None:
        """
        Build the index from vectors and their corresponding text chunks
        
        Args:
            vectors: List of embedding vectors
            texts: List of TextChunk objects corresponding to the vectors
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for vectors similar to the query vector
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing text chunks and similarity scores
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, pathlib.Path]) -> None:
        """
        Save the index to disk
        
        Args:
            path: Path to save the index to
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, pathlib.Path]) -> None:
        """
        Load the index from disk
        
        Args:
            path: Path to load the index from
        """
        pass


class QueryEngine(ABC):
    """
    Interface for query engines that process user queries
    """
    
    @abstractmethod
    def format_query(self, query: str) -> str:
        """
        Format the raw query string
        
        Args:
            query: Raw query string
            
        Returns:
            Formatted query string
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing text chunks and similarity scores
        """
        pass


class FlowController(ABC):
    """
    Interface for flow controllers that orchestrate the data pipeline
    """
    
    @abstractmethod
    def build_index(self, data_path: Union[str, pathlib.Path], **kwargs) -> None:
        """
        Build the search index from data files
        
        Args:
            data_path: Path to the data file or directory
            **kwargs: Additional arguments for customizing the build process
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def save_index(self, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        """
        Save the built index to disk
        
        Args:
            path: Path to save the index to, if None use a default path
        """
        pass
    
    @abstractmethod
    def load_index(self, path: Optional[Union[str, pathlib.Path]] = None) -> None:
        """
        Load an index from disk
        
        Args:
            path: Path to load the index from, if None use a default path
        """
        pass 