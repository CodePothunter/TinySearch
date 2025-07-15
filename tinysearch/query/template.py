"""
Template-based query engine
"""
from typing import List, Dict, Any, Optional, Union, Callable
import string

from tinysearch.base import QueryEngine, Embedder, VectorIndexer


class TemplateQueryEngine(QueryEngine):
    """
    Query engine that formats queries using templates
    """
    
    def __init__(
        self,
        embedder: Embedder,
        indexer: VectorIndexer,
        template: str = "请帮我查找：{query}",
        rerank_fn: Optional[Callable[[List[Dict[str, Any]], str], List[Dict[str, Any]]]] = None
    ):
        """
        Initialize the template query engine
        
        Args:
            embedder: Embedder to convert queries to vectors
            indexer: Indexer to search for similar vectors
            template: Template string for formatting queries
            rerank_fn: Optional function to rerank results
        """
        self.embedder = embedder
        self.indexer = indexer
        self.template = template
        self.rerank_fn = rerank_fn
    
    def format_query(self, query: str) -> str:
        """
        Format the raw query string using the template
        
        Args:
            query: Raw query string
            
        Returns:
            Formatted query string
        """
        try:
            # Try using string.format with named parameters
            return self.template.format(query=query)
        except (KeyError, ValueError):
            try:
                # Fall back to positional formatting
                return self.template.format(query)
            except (IndexError, ValueError):
                # If all else fails, just concatenate
                return f"{self.template} {query}"
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing text chunks and similarity scores
        """
        # Format the query
        formatted_query = self.format_query(query)
        
        # Convert query to vector
        query_vectors = self.embedder.embed([formatted_query])
        
        if not query_vectors:
            raise ValueError("Failed to generate embeddings for query")
        
        query_vector = query_vectors[0]
        
        # Search the index
        results = self.indexer.search(query_vector, top_k)
        
        # Apply reranking if provided
        if self.rerank_fn is not None:
            results = self.rerank_fn(results, query)
        
        return results 