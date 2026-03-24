"""
Shared utilities for fusion strategies.
"""
from typing import Any, Dict


def make_doc_key(result: Dict[str, Any]) -> str:
    """
    Create a dedup key from a retrieval result.

    Uses text + metadata (source, chunk_index) to distinguish genuinely
    different chunks that happen to share the same text content, while still
    merging the same chunk found by different retrievers.

    Disambiguation levels:
      - source: separates same text from different files
      - chunk_index: separates same text in different positions within one file
    """
    meta = result.get("metadata", {})
    source = meta.get("source", "")
    chunk_index = meta.get("chunk_index", "")
    return f"{result['text']}\x00{source}\x00{chunk_index}"
