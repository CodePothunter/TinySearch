"""
Context window management for TinySearch
"""
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class ContextWindow:
    """
    Represents a context window with a maximum token limit
    """
    text: str
    metadata: Dict[str, Any]
    token_count: int
    source_id: Optional[str] = None


class ContextWindowManager:
    """
    Manages context windows for LLM input/output
    """
    
    def __init__(
        self, 
        max_tokens: int = 4096,
        token_counting_func=None,
        reserved_tokens: int = 1000,
        overlap_strategy: str = "smart"
    ):
        """
        Initialize the context window manager
        
        Args:
            max_tokens: Maximum tokens in the context window
            token_counting_func: Function to count tokens (takes str, returns int)
            reserved_tokens: Tokens reserved for the prompt template and response
            overlap_strategy: How to handle overlapping windows ("none", "fixed", "smart")
        """
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = max_tokens - reserved_tokens
        self.overlap_strategy = overlap_strategy
        
        # Use default token counter if none provided
        if token_counting_func is None:
            self.count_tokens = self._default_token_counter
        else:
            self.count_tokens = token_counting_func
    
    def _default_token_counter(self, text: str) -> int:
        """
        Simple token counter based on word count (fallback method)
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Estimated token count
        """
        # This is a very simple approximation
        # In practice, you'd want to use a tokenizer from the LLM you're using
        words = text.split()
        return len(words) + len(text) // 20  # Add some overhead for punctuation
    
    def fit_text_to_window(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> List[ContextWindow]:
        """
        Fit text chunks into context windows
        
        Args:
            texts: List of text chunks
            metadata_list: List of metadata for each text chunk
            
        Returns:
            List of ContextWindow objects
        """
        if not texts:
            return []
        
        if len(texts) != len(metadata_list):
            raise ValueError("texts and metadata_list must have the same length")
        
        # Count tokens for each text
        token_counts = [self.count_tokens(text) for text in texts]
        
        windows = []
        current_window_text = []
        current_window_metadata = []
        current_token_count = 0
        
        for i, (text, token_count, metadata) in enumerate(zip(texts, token_counts, metadata_list)):
            # If adding this text would exceed the available tokens, create a new window
            if current_token_count + token_count > self.available_tokens and current_window_text:
                # Create a window from the accumulated text
                combined_text = "\n\n".join(current_window_text)
                # Combine metadata into a list
                combined_metadata = {"sources": current_window_metadata}
                
                windows.append(ContextWindow(
                    text=combined_text,
                    metadata=combined_metadata,
                    token_count=current_token_count
                ))
                
                # Reset for the next window
                current_window_text = []
                current_window_metadata = []
                current_token_count = 0
            
            # If a single text is too large, we need to truncate it
            if token_count > self.available_tokens:
                truncated_text = self._truncate_text(text, self.available_tokens)
                truncated_token_count = self.count_tokens(truncated_text)
                
                windows.append(ContextWindow(
                    text=truncated_text,
                    metadata={"sources": [metadata]},
                    token_count=truncated_token_count
                ))
                
                # Continue to the next text
                continue
            
            # Add the text to the current window
            current_window_text.append(text)
            current_window_metadata.append(metadata)
            current_token_count += token_count
        
        # Don't forget the last window if there's anything left
        if current_window_text:
            combined_text = "\n\n".join(current_window_text)
            combined_metadata = {"sources": current_window_metadata}
            
            windows.append(ContextWindow(
                text=combined_text,
                metadata=combined_metadata,
                token_count=current_token_count
            ))
        
        return windows
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Text to truncate
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated text
        """
        # Simple truncation strategy - this could be much more sophisticated
        words = text.split()
        
        # Approximate the number of words per token
        avg_words_per_token = len(words) / self.count_tokens(text)
        target_word_count = int(max_tokens * avg_words_per_token * 0.95)  # 5% safety margin
        
        if target_word_count >= len(words):
            return text
            
        truncated_text = " ".join(words[:target_word_count])
        truncated_text += "... (truncated)"
        
        return truncated_text
    
    def merge_context_windows(self, windows: List[ContextWindow], strategy: Optional[str] = None) -> List[ContextWindow]:
        """
        Merge multiple context windows with optional overlap
        
        Args:
            windows: List of context windows
            strategy: Overlap strategy (overrides instance setting if provided)
            
        Returns:
            List of merged context windows
        """
        if not windows or len(windows) == 1:
            return windows
            
        strategy = strategy or self.overlap_strategy
        
        if strategy == "none":
            return windows
        
        merged_windows = []
        
        for i in range(len(windows)):
            if i == 0:
                merged_windows.append(windows[i])
                continue
                
            prev_window = merged_windows[-1]
            curr_window = windows[i]
            
            if strategy == "fixed":
                # Take 20% from the end of previous window
                prev_text = prev_window.text
                prev_words = prev_text.split()
                overlap_count = max(1, len(prev_words) // 5)
                overlap_text = " ".join(prev_words[-overlap_count:])
                
                # Prepend to current window
                new_text = f"{overlap_text}...\n\n{curr_window.text}"
                new_token_count = self.count_tokens(new_text)
                
                # Combine metadata
                new_metadata = {"sources": prev_window.metadata.get("sources", []) + curr_window.metadata.get("sources", [])}
                
                merged_windows.append(ContextWindow(
                    text=new_text,
                    metadata=new_metadata,
                    token_count=new_token_count
                ))
                
            elif strategy == "smart":
                # Find potential overlap points (e.g., paragraph breaks, sentences)
                # This is a simplified version - could be made more sophisticated
                last_para = self._get_last_paragraph(prev_window.text)
                if last_para and len(last_para) > 20:  # Only if paragraph is substantial
                    new_text = f"{last_para}...\n\n{curr_window.text}"
                    new_token_count = self.count_tokens(new_text)
                    
                    # If adding overlap exceeds token limit, fall back to no overlap
                    if new_token_count <= self.available_tokens:
                        new_metadata = {"sources": prev_window.metadata.get("sources", []) + curr_window.metadata.get("sources", [])}
                        merged_windows.append(ContextWindow(
                            text=new_text,
                            metadata=new_metadata,
                            token_count=new_token_count
                        ))
                    else:
                        merged_windows.append(curr_window)
                else:
                    merged_windows.append(curr_window)
        
        return merged_windows
    
    def _get_last_paragraph(self, text: str) -> str:
        """
        Get the last paragraph of text
        
        Args:
            text: Input text
            
        Returns:
            Last paragraph
        """
        paras = text.strip().split("\n\n")
        if paras:
            return paras[-1]
        return ""
    
    def generate_context_for_query(self, query: str, texts: List[str], metadata_list: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an optimal context window for a specific query
        
        Args:
            query: The query text
            texts: List of text chunks
            metadata_list: List of metadata for each text chunk
            
        Returns:
            Tuple of (context text, metadata)
        """
        query_tokens = self.count_tokens(query)
        available_tokens = self.available_tokens - query_tokens
        
        if available_tokens <= 0:
            raise ValueError("Query is too long to fit in context window with reserved tokens")
        
        # Create initial context windows
        windows = self.fit_text_to_window(texts, metadata_list)
        
        # If we have multiple windows, we need to select or merge them
        if len(windows) > 1:
            # For now, just take the first window that fits
            # This could be improved with more sophisticated selection logic
            selected_window = windows[0]
        elif windows:
            selected_window = windows[0]
        else:
            # No windows were created
            return "", {"sources": []}
        
        return selected_window.text, selected_window.metadata 