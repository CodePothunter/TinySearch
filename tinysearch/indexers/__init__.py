"""
Vector indexers for building and maintaining search indices
"""
from pathlib import Path
from typing import Union

# Make the FAISS indexer available from the root module
from .faiss_indexer import FAISSIndexer

__all__ = [
    "FAISSIndexer"
] 

def index_exists(path: Union[str, Path]) -> bool:
    """
    Check if the index exists
    """
    path = Path(path)
    return path.with_suffix('').exists()