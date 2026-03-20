"""
Vector retriever - wraps Embedder + VectorIndexer as a Retriever
"""
from typing import Any, Dict, List, Optional, Union
import pathlib
import numpy as np

from tinysearch.base import Retriever, Embedder, VectorIndexer, TextChunk


class VectorRetriever(Retriever):
    """
    Wraps an Embedder and VectorIndexer into the Retriever interface.

    This is the bridge between the existing vector search pipeline and
    the new Retriever abstraction, ensuring backward compatibility.
    """

    def __init__(
        self,
        embedder: Embedder,
        indexer: VectorIndexer,
        query_template: Optional[str] = None,
    ):
        """
        Args:
            embedder: Embedder to convert text to vectors
            indexer: VectorIndexer for similarity search
            query_template: Optional template for formatting queries (e.g. "请帮我查找：{query}")
        """
        self.embedder = embedder
        self.indexer = indexer
        self.query_template = query_template

    def build(self, chunks: List[TextChunk]) -> None:
        """Embed chunks and build the vector index"""
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.embed(texts)
        self.indexer.build(vectors, chunks)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Embed query and search the vector index.

        Args:
            query: Query string
            top_k: Number of results to return
            **kwargs:
                candidate_ids: Optional Set[int] of chunk indices to restrict search to
        """
        candidate_ids = kwargs.get("candidate_ids")

        # Apply query template if configured
        formatted_query = query
        if self.query_template:
            try:
                formatted_query = self.query_template.format(query=query)
            except (KeyError, ValueError):
                formatted_query = query

        # Embed the query
        query_vectors = self.embedder.embed([formatted_query])
        if not query_vectors:
            return []
        query_vector = query_vectors[0]

        # Search (forward candidate_ids to indexer)
        raw_results = self.indexer.search(query_vector, top_k, candidate_ids=candidate_ids)

        # Normalize scores to [0, 1] and add retrieval_method
        results = []
        for r in raw_results:
            result = {
                "text": r["text"],
                "metadata": r.get("metadata", {}),
                "score": float(r.get("score", 0.0)),
                "retrieval_method": "vector",
            }
            # Preserve embedding if present
            if "embedding" in r:
                result["embedding"] = r["embedding"]
            results.append(result)

        return results

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """Save the vector index to disk"""
        self.indexer.save(path)

    def load(self, path: Union[str, pathlib.Path]) -> None:
        """Load the vector index from disk"""
        self.indexer.load(path)
