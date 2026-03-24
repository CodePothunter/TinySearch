"""
Cross-encoder reranker using FlagEmbedding BGE Reranker
"""
from typing import Any, Dict, List, Optional

from tinysearch.base import Reranker
from tinysearch.logger import get_logger

logger = get_logger("CrossEncoderReranker")

try:
    from FlagEmbedding import FlagReranker
    FLAGEMBEDDING_AVAILABLE = True
except ImportError:
    FLAGEMBEDDING_AVAILABLE = False


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3 (or compatible model).

    Uses FlagEmbedding for GPU-accelerated cross-encoder inference.
    The model is lazily loaded on first use.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
        use_fp16: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name or local path
            device: Device to use ("cuda", "cpu", or None for auto)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            use_fp16: Whether to use FP16 for faster inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self._model = None

    def _ensure_model(self) -> None:
        """Lazily load the reranker model"""
        if self._model is not None:
            return

        if not FLAGEMBEDDING_AVAILABLE:
            raise ImportError(
                "FlagEmbedding is required for CrossEncoderReranker. "
                "Install with: pip install FlagEmbedding"
            )

        # Normalize device for FlagReranker
        device = self.device
        if device is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        elif device == "cuda":
            device = 0
        elif device.startswith("cuda:") and device[5:].isdigit():
            device = int(device[5:])

        logger.info(f"Loading reranker model: {self.model_name} on device={device}")

        self._model = FlagReranker(
            self.model_name,
            devices=device,
            batch_size=self.batch_size,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
        )

        # Trigger full weight loading with dummy inference
        try:
            self._model.compute_score([["warmup", "warmup"]])
        except Exception:
            pass

        logger.info("Reranker model loaded successfully")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-rank candidates using cross-encoder scoring"""
        if not candidates:
            return []

        self._ensure_model()

        # Build query-document pairs
        pairs = []
        for item in candidates:
            doc_text = item.get("text", "")
            pairs.append([query, doc_text])

        # Compute scores
        scores = self._model.compute_score(pairs)

        # Handle single result (compute_score returns a float instead of list)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Attach rerank scores
        reranked = []
        for item, score in zip(candidates, scores):
            reranked.append({
                **item,
                "rerank_score": float(score),
            })

        # Sort by rerank score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    @classmethod
    def is_available(cls) -> bool:
        """Check if FlagEmbedding is installed"""
        return FLAGEMBEDDING_AVAILABLE
