"""
Embedding models for converting text to vectors
"""

# Make all embedding models available from the root module
from .huggingface import HuggingFaceEmbedder

__all__ = [
    "HuggingFaceEmbedder"
] 