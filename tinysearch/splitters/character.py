"""
Character-based text splitter
"""
from typing import List, Optional, Dict, Any
import re

from tinysearch.base import TextSplitter, TextChunk


class CharacterTextSplitter(TextSplitter):
    """
    Split text into chunks based on character count
    """
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        separator: str = "\n\n",
        keep_separator: bool = False,
        strip_whitespace: bool = True
    ):
        """
        Initialize the character text splitter
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: String to use as separator between chunks
            keep_separator: Whether to include the separator in chunks
            strip_whitespace: Whether to strip whitespace from chunk ends
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) must be smaller than chunk size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
    
    def split(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[TextChunk]:
        """
        Split texts into chunks
        
        Args:
            texts: List of text strings to split
            metadata: Optional list of metadata dicts corresponding to each text
            
        Returns:
            List of TextChunk objects
        """
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError(
                f"Number of metadata items ({len(metadata)}) does not match "
                f"number of texts ({len(texts)})"
            )
        
        result = []
        
        for i, text in enumerate(texts):
            # Get metadata for this text
            meta = metadata[i] if metadata is not None else {}
            
            # Split the text into chunks
            chunks = self._split_text(text)
            
            # Create TextChunk objects
            for j, chunk in enumerate(chunks):
                # Add chunk metadata
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "chunk_index": j,
                    "total_chunks": len(chunks)
                })
                
                result.append(TextChunk(chunk, chunk_meta))
        
        return result
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks
        
        Args:
            text: Text string to split
            
        Returns:
            List of text chunks
        """
        # First split the text by separator
        if self.separator:
            splits = self._split_by_separator(text)
        else:
            splits = [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            if len(split) > self.chunk_size:
                # If a single split is too large, we need to break it down further
                if current_chunk:
                    # First add the current chunk we've accumulated
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Then break down the large split
                subsplits = self._split_long_text(split)
                chunks.extend(subsplits)
            elif current_length + len(split) <= self.chunk_size:
                # The split fits in the current chunk
                current_chunk.append(split)
                current_length += len(split)
            else:
                # The split doesn't fit, start a new chunk
                chunks.append("".join(current_chunk))
                current_chunk = [split]
                current_length = len(split)
        
        # Don't forget to add the last chunk
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        # Process the chunks for overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        # Strip whitespace if needed
        if self.strip_whitespace:
            chunks = [chunk.strip() for chunk in chunks]
        
        return chunks
    
    def _split_by_separator(self, text: str) -> List[str]:
        """
        Split text by separator
        
        Args:
            text: Text string to split
            
        Returns:
            List of text segments
        """
        # If we want to keep the separator with the text, we use a regex with lookahead
        if self.keep_separator:
            pattern = f"(?<={re.escape(self.separator)})"
            splits = re.split(pattern, text)
            # Make sure each split (except the first one) starts with the separator
            for i in range(1, len(splits)):
                if not splits[i].startswith(self.separator):
                    splits[i] = self.separator + splits[i]
        else:
            # Simple split by separator
            splits = text.split(self.separator)
            # Reintroduce the separator if we're not at the final split
            splits = [
                (split + self.separator) if i < len(splits) - 1 else split
                for i, split in enumerate(splits)
            ]
        
        return splits
    
    def _split_long_text(self, text: str) -> List[str]:
        """
        Split a text that's too long for a single chunk
        
        Args:
            text: Text string to split
            
        Returns:
            List of text chunks
        """
        result = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            # If we're not at the beginning of the text, include overlap
            if start > 0:
                start = max(0, start - self.chunk_overlap)
            
            # Don't go past the end of the text
            if end >= len(text):
                result.append(text[start:])
                break
            
            # Try to end at a natural boundary (whitespace)
            if self.separator:
                # Look for the separator near the end of the chunk
                last_sep = text.rfind(self.separator, start, end)
                if last_sep > start:
                    end = last_sep + len(self.separator)
            
            result.append(text[start:end])
            start = end
        
        return result
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of text chunks with overlap
        """
        result = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk doesn't need to add text from previous chunk
                result.append(chunk)
            else:
                # Take the last chunk_overlap characters from the previous chunk
                prev_chunk = chunks[i - 1]
                overlap = prev_chunk[-self.chunk_overlap:]
                
                # Only add the overlap if it's not already at the beginning of the current chunk
                # (which could happen due to the way we split)
                if not chunk.startswith(overlap):
                    chunk = overlap + chunk
                
                result.append(chunk)
        
        return result 