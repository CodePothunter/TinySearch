"""
Inverted index over TextChunk metadata for fast candidate set lookup.

Enables O(1) pre-filtering by metadata fields (e.g., grade, type, tags)
instead of linear post-filtering over all retrieval results.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from tinysearch.base import TextChunk

logger = logging.getLogger(__name__)

# Filter value types matching HybridQueryEngine._match_filters
FilterValue = Union[str, int, float, bool, List, Callable]

# Scalar types that can be directly indexed
_INDEXABLE_TYPES = (str, int, float, bool)


class MetadataIndex:
    """
    Inverted index: metadata field -> value -> set of chunk IDs.

    Supports:
    - Scalar values (str, int, float, bool): indexed directly
    - List[str/int/...] values (e.g., tags): each element indexed separately
    """

    def __init__(self) -> None:
        self._index: Dict[str, Dict[Any, Set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._total_chunks: int = 0

    def build(self, chunks: List[TextChunk]) -> None:
        """
        Build inverted indices from chunk metadata.

        Args:
            chunks: Ordered list of TextChunks. The positional index (0-based)
                    becomes the chunk ID used in candidate sets.
        """
        self._index = defaultdict(lambda: defaultdict(set))
        self._total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            if not chunk.metadata:
                continue
            for key, value in chunk.metadata.items():
                if isinstance(value, _INDEXABLE_TYPES):
                    self._index[key][value].add(i)
                elif isinstance(value, list):
                    for element in value:
                        if isinstance(element, _INDEXABLE_TYPES):
                            self._index[key][element].add(i)

        field_stats = {k: len(v) for k, v in self._index.items()}
        logger.info(
            "MetadataIndex built: %d chunks, fields=%s",
            self._total_chunks,
            field_stats,
        )

    def add_chunks(self, chunks: List[TextChunk], start_id: int) -> None:
        """
        Incrementally add new chunks to the inverted index.

        Args:
            chunks: New TextChunks to index
            start_id: First chunk ID to assign (typically current total_chunks)
        """
        for offset, chunk in enumerate(chunks):
            chunk_id = start_id + offset
            if not chunk.metadata:
                continue
            for key, value in chunk.metadata.items():
                if isinstance(value, _INDEXABLE_TYPES):
                    self._index[key][value].add(chunk_id)
                elif isinstance(value, list):
                    for element in value:
                        if isinstance(element, _INDEXABLE_TYPES):
                            self._index[key][element].add(chunk_id)

        self._total_chunks += len(chunks)
        logger.info(
            "MetadataIndex: added %d chunks (total now %d)",
            len(chunks),
            self._total_chunks,
        )

    def lookup(self, filters: Dict[str, FilterValue]) -> Optional[Set[int]]:
        """
        Resolve filters to a candidate ID set via inverted index.

        Filter semantics (matches HybridQueryEngine._match_filters):
        - scalar (str/int/float/bool): exact match -> direct set lookup
        - list: OR over values -> union of sets
        - multiple filter keys: AND -> set intersection
        - callable: cannot be resolved -> returns None

        Returns:
            Set[int] of matching chunk IDs, or None if any filter is callable.
        """
        if not filters:
            return None

        result: Optional[Set[int]] = None

        for key, condition in filters.items():
            if callable(condition):
                return None

            matched = self._lookup_single(key, condition)

            if result is None:
                result = matched
            else:
                result = result & matched

            # Short-circuit on empty intersection
            if result is not None and len(result) == 0:
                return set()

        return result if result is not None else set()

    def _lookup_single(self, key: str, condition: FilterValue) -> Set[int]:
        """Lookup a single filter key-value pair."""
        field_index = self._index.get(key)
        if field_index is None:
            return set()

        if isinstance(condition, list):
            # OR: union of all matching values
            matched = set()
            for val in condition:
                matched |= field_index.get(val, set())
            return matched
        else:
            # Exact match
            return set(field_index.get(condition, set()))

    def classify_filters(
        self, filters: Dict[str, FilterValue]
    ) -> Tuple[Dict[str, FilterValue], Dict[str, FilterValue]]:
        """
        Split filters into indexable and non-indexable (callable) parts.

        Returns:
            (indexable_filters, callable_filters)
        """
        indexable = {}
        callables = {}
        for key, condition in filters.items():
            if callable(condition):
                callables[key] = condition
            else:
                indexable[key] = condition
        return indexable, callables

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    @property
    def fields(self) -> List[str]:
        return list(self._index.keys())

    def save(self, path: Union[str, Path]) -> None:
        """
        Save to JSON. Sets become sorted lists; value types are preserved
        via a type tag for proper reconstruction on load.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            "version": 1,
            "total_chunks": self._total_chunks,
            "fields": {},
        }

        for field_name, value_map in self._index.items():
            entries = {}
            for value, ids in value_map.items():
                # Use repr-style key that preserves type info
                type_tag = type(value).__name__
                str_key = json.dumps({"v": value, "t": type_tag}, ensure_ascii=False)
                entries[str_key] = sorted(ids)
            serializable["fields"][field_name] = entries

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)

        logger.info("MetadataIndex saved to %s", path)

    def load(self, path: Union[str, Path]) -> None:
        """Load from JSON, reconstructing sets and value types."""
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._total_chunks = data["total_chunks"]
        self._index = defaultdict(lambda: defaultdict(set))

        type_constructors = {"str": str, "int": int, "float": float, "bool": bool}

        for field_name, entries in data["fields"].items():
            for str_key, id_list in entries.items():
                key_data = json.loads(str_key)
                value = key_data["v"]
                type_tag = key_data["t"]
                # Reconstruct typed value
                constructor = type_constructors.get(type_tag)
                if constructor:
                    value = constructor(value)
                self._index[field_name][value] = set(id_list)

        logger.info(
            "MetadataIndex loaded from %s: %d chunks, %d fields",
            path,
            self._total_chunks,
            len(self._index),
        )
