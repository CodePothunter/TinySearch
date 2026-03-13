"""
BM25 Retriever - Keyword-based search using bm25s

Fast BM25 implementation with optional jieba Chinese tokenization.
"""
import json
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tinysearch.base import Retriever, TextChunk
from tinysearch.logger import get_logger

logger = get_logger("BM25Retriever")

try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


def _default_tokenizer(text: str) -> List[str]:
    """Default tokenizer: jieba if available, else whitespace split"""
    text_lower = text.lower()
    if JIEBA_AVAILABLE:
        return list(jieba.cut(text_lower))
    return re.findall(r'[\w]+', text_lower)


class BM25Retriever(Retriever):
    """
    BM25 keyword-based retriever using the bm25s library.

    Features:
    - Fast indexing and retrieval via bm25s
    - Configurable tokenizer (default: jieba for Chinese, fallback to whitespace)
    - Persistent index storage
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Args:
            tokenizer: Custom tokenizer function. Takes a string, returns list of tokens.
                       Defaults to jieba (if available) or whitespace splitting.
        """
        if not BM25S_AVAILABLE:
            logger.warning(
                "bm25s not installed. BM25 retrieval will not work. "
                "Install with: pip install bm25s"
            )
        self.tokenizer = tokenizer or _default_tokenizer

        # Internal state
        self._index: Optional[Any] = None  # bm25s.BM25
        self._chunks: List[TextChunk] = []
        self._corpus_tokens: List[List[str]] = []

    def build(self, chunks: List[TextChunk]) -> None:
        """Build BM25 index from text chunks"""
        if not BM25S_AVAILABLE:
            raise ImportError(
                "bm25s is required for BM25Retriever. Install with: pip install bm25s"
            )
        if not chunks:
            return

        self._chunks = list(chunks)

        # Tokenize all documents
        self._corpus_tokens = [self.tokenizer(chunk.text) for chunk in chunks]

        # Build bm25s index
        self._index = bm25s.BM25()
        self._index.index(self._corpus_tokens)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25 keyword matching"""
        if self._index is None:
            return []

        # Tokenize query
        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        # Clamp top_k to number of indexed documents
        effective_k = min(top_k, len(self._chunks))
        if effective_k == 0:
            return []

        # Retrieve (no corpus → returns indices instead of documents)
        result = self._index.retrieve(
            [query_tokens], k=effective_k, return_as="tuple"
        )
        indices = result.documents[0]
        scores = result.scores[0]

        # Build result list
        results = []
        for idx, score in zip(indices, scores):
            idx = int(idx)
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            results.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": float(score),
                "retrieval_method": "bm25",
            })

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save BM25 index to disk"""
        if self._index is None:
            raise ValueError("No index to save. Call build() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save bm25s index
        self._index.save(str(path), corpus=self._corpus_tokens)

        # Save chunks metadata
        chunks_data = [(chunk.text, chunk.metadata) for chunk in self._chunks]
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(chunks_data, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load BM25 index from disk"""
        if not BM25S_AVAILABLE:
            raise ImportError(
                "bm25s is required for BM25Retriever. Install with: pip install bm25s"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index directory not found: {path}")

        # Load bm25s index (don't load corpus - we manage chunks separately)
        self._index = bm25s.BM25.load(str(path), load_corpus=False)

        # Load chunks metadata
        chunks_file = path / "chunks.pkl"
        if chunks_file.exists():
            with open(chunks_file, "rb") as f:
                chunks_data = pickle.load(f)
            self._chunks = [TextChunk(text, metadata) for text, metadata in chunks_data]
        else:
            self._chunks = []
