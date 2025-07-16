"""
HuggingFace-based embedding model
"""
import traceback
from typing import List, Dict, Any, Optional, Union, Callable
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from tinysearch.base import Embedder


class HuggingFaceEmbedder(Embedder):
    """
    Embedding model using HuggingFace transformers
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        max_length: int = 8192,
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        pooling: str = "last_token"
    ):
        """
        Initialize the HuggingFace embedder
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use for inference (e.g., "cpu", "cuda", "cuda:0")
                   If None, will use CUDA if available, otherwise CPU
            max_length: Maximum sequence length for the model
            batch_size: Batch size for inference
            normalize_embeddings: Whether to normalize the embeddings to unit length
            cache_dir: Directory to cache downloaded models
            progress_callback: Optional callback function to report progress
                              Args: (current_batch, total_batches)
            pooling: Pooling method to use ("last_token", "mean", "cls")
                     For Qwen3-Embedding models, "last_token" is recommended
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.cache_dir = cache_dir
        self.progress_callback = progress_callback
        self.pooling = pooling
        
        # Set device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
                print("PyTorch not installed. Using CPU for embeddings.")
        
        self.device = device
        
        # Lazy initialization of model and tokenizer
        self._model = None
        self._tokenizer = None
        self._initialized = False
    
    def _initialize_model(self):
        """
        Initialize the model and tokenizer
        """
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                
                # Set padding side to 'left' for Qwen3 models to improve performance
                padding_side = "left" if "Qwen3" in self.model_name else "right"
                
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    padding_side=padding_side
                )
                
                # Load model - use flash attention for Qwen3 models if available
                model_kwargs = {}
                if "Qwen3" in self.model_name:
                    try:
                        import torch
                        model_kwargs = {
                            "attn_implementation": "flash_attention_2", 
                            "torch_dtype": torch.float16
                        }
                    except (ImportError, AttributeError):
                        pass
                try:
                    self._model = AutoModel.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        **model_kwargs
                    )
                except Exception as e:
                    print(f"Failed to load model with flash attention, trying normal attention")
                    model_kwargs = {}
                
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    **model_kwargs
                )
                print(f"Model loaded successfully")
                
                # Move model to device
                self._model.to(self.device)
                print(f"Model moved to device {self.device}")
                
                # Set to evaluation mode
                self._model.eval()
                
                self._initialized = True
            except ImportError:
                print(traceback.format_exc())
                raise ImportError(
                    "Could not import transformers. "
                    "Please install it with: pip install transformers"
                )
            except Exception as e:
                raise RuntimeError(f"Error loading model: {e}")

    def get_embedding_dim(self) -> int:
        """
        Returns the dimensionality of the embedding vectors
        
        Returns:
            Integer representing embedding dimensions
        """
        # Initialize model if not already done
        self._initialize_model()
        
        # Different models store their config differently
        try:
            assert self._initialized, "Model not initialized"
            import torch
            with torch.no_grad():
                # Create a simple input to get output shape
                inputs = self._tokenizer(
                    ["Test text"],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ) # type: ignore
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._model(**inputs) # type: ignore
                
                # Try different ways to get dimensions depending on model architecture
                try:
                    return outputs.pooler_output.shape[1]
                except AttributeError:
                    return outputs.last_hidden_state.shape[2]
        except Exception as e:
            # Fallback dimensions for common embedding models
            model_dimensions = {
                "Qwen/Qwen-Embedding": 1536,
                "Qwen/Qwen3-Embedding-0.6B": 1024,
                "Qwen/Qwen3-Embedding-4B": 2560,
                "Qwen/Qwen3-Embedding-8B": 4096,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "sentence-transformers/all-MiniLM-L6-v2": 384
            }
            
            for model_prefix, dim in model_dimensions.items():
                if model_prefix in self.model_name:
                    return dim
            
            # Default fallback
            return 768
    
    def _last_token_pool(self, last_hidden_states, attention_mask):
        """
        Extract embeddings from the last token (recommended for Qwen3 models)
        
        Args:
            last_hidden_states: Hidden states from the model
            attention_mask: Attention mask
            
        Returns:
            Tensor containing the embeddings
        """
        import torch
        # Check if left padding (common in Qwen3 models)
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            # For right padding, find the last non-padding token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to embedding vectors
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as float lists
        """
        # Initialize model if not already done
        self._initialize_model()
        
        # Split texts into batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        total_batches = len(batches)
        
        all_embeddings = []
        
        try:
            assert self._initialized, "Model not initialized"
            import torch
            
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(batches), total=total_batches, desc="Embedding texts"):
                    # Report progress
                    if self.progress_callback:
                        self.progress_callback(batch_idx + 1, total_batches)
                    
                    # Tokenize
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ) # type: ignore
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get model output
                    outputs = self._model(**inputs) # type: ignore
                    
                    # Get embeddings based on pooling method
                    if self.pooling == "last_token" or "Qwen3" in self.model_name:
                        embeddings = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                    elif self.pooling == "cls":
                        # Use [CLS] token (first token)
                        embeddings = outputs.last_hidden_state[:, 0]
                    elif self.pooling == "mean" or True:  # Default to mean pooling as fallback
                        # Use mean pooling over token embeddings
                        token_embeddings = outputs.last_hidden_state
                        attention_mask = inputs["attention_mask"]
                        
                        # Compute mean of token embeddings (excluding padding tokens)
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        embeddings = sum_embeddings / sum_mask
                    
                    # Normalize if requested
                    if self.normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Convert to list of lists
                    batch_embeddings = embeddings.cpu().numpy().tolist()
                    all_embeddings.extend(batch_embeddings)
        
        except Exception as e:
            raise RuntimeError(f"Error during embedding generation: {e}")
        
        return all_embeddings 