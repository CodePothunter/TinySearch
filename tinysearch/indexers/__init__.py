"""
Vector indexers for building and maintaining search indices
"""

# Make the FAISS indexer available from the root module
from .faiss_indexer import FAISSIndexer

__all__ = [
    "FAISSIndexer"
] 