"""
Text splitters for chunking text into smaller segments
"""

# Make all splitter classes available from the root module
from .character import CharacterTextSplitter

__all__ = [
    "CharacterTextSplitter"
] 