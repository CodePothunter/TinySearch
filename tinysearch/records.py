"""
Utilities for building TextChunks from structured records via RecordAdapter.
"""
import logging
from typing import Any, Dict, List, Optional

from tinysearch.base import RecordAdapter, TextChunk, TextSplitter

logger = logging.getLogger(__name__)


def build_chunks_from_records(
    records: Dict[str, Dict[str, Any]],
    adapter: RecordAdapter,
    splitter: Optional[TextSplitter] = None,
) -> List[TextChunk]:
    """
    Convert a dict of records to TextChunks ready for indexing.

    Args:
        records: Mapping of record_id -> record_data
        adapter: RecordAdapter that converts each record to a TextChunk
        splitter: Optional TextSplitter. If None, each record becomes exactly
                  one TextChunk. If provided, each record's text is further
                  split, with the original metadata inherited by all sub-chunks.

    Returns:
        Ordered list of TextChunks (order follows dict iteration order)
    """
    chunks: List[TextChunk] = []

    for record_id, record_data in records.items():
        chunk = adapter.to_chunk(record_id, record_data)

        # Ensure record_id is always in metadata
        if "record_id" not in chunk.metadata:
            chunk.metadata["record_id"] = record_id

        if splitter is None:
            chunks.append(chunk)
        else:
            sub_chunks = splitter.split([chunk.text], [chunk.metadata])
            chunks.extend(sub_chunks)

    logger.info(
        "Built %d chunks from %d records (splitter=%s)",
        len(chunks),
        len(records),
        type(splitter).__name__ if splitter else "None",
    )
    return chunks
