"""
Content hash tracking for incremental indexing change detection.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from tinysearch.base import TextChunk

logger = logging.getLogger(__name__)


class ChangeSet:
    """Result of change detection between current and tracked records."""

    __slots__ = ("new", "modified", "deleted", "unchanged")

    def __init__(
        self,
        new: List[str],
        modified: List[str],
        deleted: Set[str],
        unchanged: Set[str],
    ):
        self.new = new
        self.modified = modified
        self.deleted = deleted
        self.unchanged = unchanged

    @property
    def has_changes(self) -> bool:
        return bool(self.new or self.modified or self.deleted)

    def __repr__(self) -> str:
        return (
            f"ChangeSet(new={len(self.new)}, modified={len(self.modified)}, "
            f"deleted={len(self.deleted)}, unchanged={len(self.unchanged)})"
        )


class ContentHashTracker:
    """
    Track content hashes for record-level change detection.

    Each record is identified by its record_id (string). The hash is
    computed from the text and a configurable subset of metadata keys.
    """

    _INTERNAL_KEYS = frozenset({"chunk_index", "total_chunks"})

    def __init__(self, hash_metadata_keys: Optional[List[str]] = None):
        """
        Args:
            hash_metadata_keys: Metadata keys to include in hash.
                If None, all keys except internal ones are included.
        """
        self._hashes: Dict[str, str] = {}
        self._hash_metadata_keys = hash_metadata_keys

    def compute_hash(self, text: str, metadata: Dict[str, Any]) -> str:
        """Compute MD5 hash of text + sorted metadata."""
        hasher = hashlib.md5()
        hasher.update(text.encode("utf-8"))

        if self._hash_metadata_keys is not None:
            keys = self._hash_metadata_keys
        else:
            keys = sorted(k for k in metadata if k not in self._INTERNAL_KEYS)

        for key in sorted(keys):
            if key in metadata:
                val = metadata[key]
                # Normalize lists for deterministic hashing
                if isinstance(val, list):
                    val = json.dumps(sorted(str(v) for v in val), ensure_ascii=False)
                hasher.update(f"{key}={val}".encode("utf-8"))

        return hasher.hexdigest()

    def detect_changes(self, current_records: Dict[str, TextChunk]) -> ChangeSet:
        """
        Compare current records against tracked hashes.

        Args:
            current_records: Mapping of record_id -> TextChunk

        Returns:
            ChangeSet with new, modified, deleted, unchanged
        """
        tracked_ids = set(self._hashes.keys())

        new_ids: List[str] = []
        modified_ids: List[str] = []
        unchanged_ids: Set[str] = set()

        for rid, chunk in current_records.items():
            h = self.compute_hash(chunk.text, chunk.metadata)
            if rid not in tracked_ids:
                new_ids.append(rid)
            elif self._hashes[rid] != h:
                modified_ids.append(rid)
            else:
                unchanged_ids.add(rid)

        deleted_ids = tracked_ids - set(current_records.keys())

        return ChangeSet(
            new=new_ids,
            modified=modified_ids,
            deleted=deleted_ids,
            unchanged=unchanged_ids,
        )

    def update(self, records: Dict[str, TextChunk]) -> None:
        """Update tracked hashes after a successful build."""
        for rid, chunk in records.items():
            self._hashes[rid] = self.compute_hash(chunk.text, chunk.metadata)

    def remove(self, record_ids: Set[str]) -> None:
        """Remove record_ids from tracking."""
        for rid in record_ids:
            self._hashes.pop(rid, None)

    @property
    def tracked_count(self) -> int:
        return len(self._hashes)

    def save(self, path: Union[str, Path]) -> None:
        """Save hash state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "hash_metadata_keys": self._hash_metadata_keys,
                    "hashes": self._hashes,
                },
                f,
                ensure_ascii=False,
            )
        logger.info("ContentHashTracker saved to %s (%d records)", path, len(self._hashes))

    def load(self, path: Union[str, Path]) -> None:
        """Load hash state from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._hashes = data["hashes"]
        self._hash_metadata_keys = data.get("hash_metadata_keys")
        logger.info("ContentHashTracker loaded from %s (%d records)", path, len(self._hashes))
