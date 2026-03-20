"""
Substring Retriever - Ctrl+F style exact match search

Fast regex-based substring search. No external dependencies.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import pickle

from tinysearch.base import Retriever, TextChunk


class SubstringRetriever(Retriever):
    """
    Regex/substring retriever for exact match queries.

    Operates on raw text with regex or plain substring matching.
    Useful for finding exact phrases, codes, or patterns.
    """

    def __init__(self, is_regex: bool = False):
        """
        Args:
            is_regex: If True, treat query as regex pattern.
                      If False, escape query for literal substring matching.
        """
        self.is_regex = is_regex
        self._chunks: List[TextChunk] = []

    def build(self, chunks: List[TextChunk]) -> None:
        """Store chunks in memory for substring search"""
        self._chunks = list(chunks)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search chunks using regex/substring matching.

        Args:
            query: Query string or regex pattern
            top_k: Number of results to return
            **kwargs:
                candidate_ids: Optional Set[int] of chunk indices to restrict search to
        """
        if not self._chunks or not query:
            return []

        candidate_ids = kwargs.get("candidate_ids")
        results = []

        try:
            if self.is_regex:
                pattern = re.compile(query, re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(query), re.IGNORECASE)

            for i, chunk in enumerate(self._chunks):
                if candidate_ids is not None and i not in candidate_ids:
                    continue
                match = pattern.search(chunk.text)
                if match:
                    score = self._calculate_match_score(match, chunk.text)
                    results.append({
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "score": score,
                        "retrieval_method": "substring",
                        "match_text": match.group(0),
                    })

                # Collect extra then sort
                if len(results) >= top_k * 2:
                    break

        except re.error:
            # Fallback to plain substring if regex fails
            results = self._plain_substring_search(query, top_k * 2)

        # Sort by score descending and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _calculate_match_score(self, match: re.Match, text: str) -> float:
        """Calculate match quality score"""
        score = 1.0

        # Bonus for match at start
        if match.start() == 0:
            score += 2.0

        # Bonus for longer matches (max +3)
        match_len = len(match.group(0))
        score += min(match_len / 10.0, 3.0)

        # Small penalty for matching deep in long text
        if len(text) > 500 and match.start() > 100:
            score -= 0.5

        return score

    def _plain_substring_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback plain substring search when regex fails"""
        results = []
        query_lower = query.lower()

        for chunk in self._chunks:
            text_lower = chunk.text.lower()
            if query_lower in text_lower:
                pos = text_lower.find(query_lower)
                score = 1.0
                if pos == 0:
                    score += 2.0
                score += min(len(query) / 10.0, 3.0)

                results.append({
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": score,
                    "retrieval_method": "substring",
                    "match_text": query[:50],
                })

                if len(results) >= limit:
                    break

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save chunks to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        chunks_data = [(chunk.text, chunk.metadata) for chunk in self._chunks]
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(chunks_data, f)

        # Save config
        config = {"is_regex": self.is_regex}
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load chunks from disk"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Substring index directory not found: {path}")

        chunks_file = path / "chunks.pkl"
        if chunks_file.exists():
            with open(chunks_file, "rb") as f:
                chunks_data = pickle.load(f)
            self._chunks = [TextChunk(text, metadata) for text, metadata in chunks_data]

        config_file = path / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
            self.is_regex = config.get("is_regex", False)
