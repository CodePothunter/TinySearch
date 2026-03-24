"""
Tests for incremental indexing: ContentHashTracker, MetadataIndex.add_chunks,
soft delete, and FlowController.build_incremental.
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict

from tinysearch.base import RecordAdapter, TextChunk
from tinysearch.indexers.hash_tracker import ContentHashTracker, ChangeSet
from tinysearch.indexers.metadata_index import MetadataIndex
from tinysearch.query.hybrid import HybridQueryEngine
from tinysearch.retrievers.bm25_retriever import BM25Retriever
from tinysearch.retrievers.substring_retriever import SubstringRetriever
from tinysearch.fusion.weighted import WeightedFusion


class SimpleAdapter(RecordAdapter):
    def to_chunk(self, record_id: str, record: Dict[str, Any]) -> TextChunk:
        return TextChunk(
            text=record.get("text", ""),
            metadata={"record_id": record_id, "grade": record.get("grade", "")},
        )


# ── ContentHashTracker ───────────────────────────

class TestContentHashTracker:
    def test_compute_hash_deterministic(self):
        tracker = ContentHashTracker()
        h1 = tracker.compute_hash("hello", {"k": "v"})
        h2 = tracker.compute_hash("hello", {"k": "v"})
        assert h1 == h2

    def test_compute_hash_changes_on_text(self):
        tracker = ContentHashTracker()
        h1 = tracker.compute_hash("hello", {"k": "v"})
        h2 = tracker.compute_hash("world", {"k": "v"})
        assert h1 != h2

    def test_compute_hash_changes_on_metadata(self):
        tracker = ContentHashTracker()
        h1 = tracker.compute_hash("hello", {"grade": "六年级"})
        h2 = tracker.compute_hash("hello", {"grade": "七年级"})
        assert h1 != h2

    def test_compute_hash_ignores_internal_keys(self):
        tracker = ContentHashTracker()
        h1 = tracker.compute_hash("hello", {"grade": "六年级", "chunk_index": 0})
        h2 = tracker.compute_hash("hello", {"grade": "六年级", "chunk_index": 5})
        assert h1 == h2

    def test_compute_hash_custom_metadata_keys(self):
        tracker = ContentHashTracker(hash_metadata_keys=["grade"])
        h1 = tracker.compute_hash("hello", {"grade": "六年级", "type": "A"})
        h2 = tracker.compute_hash("hello", {"grade": "六年级", "type": "B"})
        assert h1 == h2  # "type" is not in hash_metadata_keys

    def test_detect_all_new(self):
        tracker = ContentHashTracker()
        current = {
            "q1": TextChunk("hello", {"record_id": "q1"}),
            "q2": TextChunk("world", {"record_id": "q2"}),
        }
        changes = tracker.detect_changes(current)
        assert len(changes.new) == 2
        assert len(changes.modified) == 0
        assert len(changes.deleted) == 0

    def test_detect_no_changes(self):
        tracker = ContentHashTracker()
        current = {"q1": TextChunk("hello", {"record_id": "q1"})}
        tracker.update(current)
        changes = tracker.detect_changes(current)
        assert not changes.has_changes
        assert len(changes.unchanged) == 1

    def test_detect_modified(self):
        tracker = ContentHashTracker()
        original = {"q1": TextChunk("hello", {"record_id": "q1"})}
        tracker.update(original)
        modified = {"q1": TextChunk("changed", {"record_id": "q1"})}
        changes = tracker.detect_changes(modified)
        assert len(changes.modified) == 1
        assert changes.modified[0] == "q1"

    def test_detect_deleted(self):
        tracker = ContentHashTracker()
        original = {"q1": TextChunk("a", {}), "q2": TextChunk("b", {})}
        tracker.update(original)
        current = {"q1": TextChunk("a", {})}
        changes = tracker.detect_changes(current)
        assert changes.deleted == {"q2"}

    def test_detect_mixed(self):
        tracker = ContentHashTracker()
        original = {
            "q1": TextChunk("unchanged", {}),
            "q2": TextChunk("will_modify", {}),
            "q3": TextChunk("will_delete", {}),
        }
        tracker.update(original)
        current = {
            "q1": TextChunk("unchanged", {}),
            "q2": TextChunk("modified_text", {}),
            "q4": TextChunk("brand_new", {}),
        }
        changes = tracker.detect_changes(current)
        assert set(changes.new) == {"q4"}
        assert set(changes.modified) == {"q2"}
        assert changes.deleted == {"q3"}
        assert changes.unchanged == {"q1"}

    def test_update_and_remove(self):
        tracker = ContentHashTracker()
        records = {"q1": TextChunk("hello", {})}
        tracker.update(records)
        assert tracker.tracked_count == 1
        tracker.remove({"q1"})
        assert tracker.tracked_count == 0

    def test_save_load_roundtrip(self, tmp_path):
        tracker = ContentHashTracker(hash_metadata_keys=["grade"])
        records = {
            "q1": TextChunk("hello", {"grade": "六年级"}),
            "q2": TextChunk("world", {"grade": "七年级"}),
        }
        tracker.update(records)

        path = tmp_path / "hashes.json"
        tracker.save(path)

        loaded = ContentHashTracker()
        loaded.load(path)
        assert loaded.tracked_count == 2
        # Should detect no changes
        changes = loaded.detect_changes(records)
        assert not changes.has_changes


class TestChangeSet:
    def test_has_changes_true(self):
        cs = ChangeSet(new=["a"], modified=[], deleted=set(), unchanged=set())
        assert cs.has_changes

    def test_has_changes_false(self):
        cs = ChangeSet(new=[], modified=[], deleted=set(), unchanged={"a"})
        assert not cs.has_changes

    def test_repr(self):
        cs = ChangeSet(new=["a"], modified=["b"], deleted={"c"}, unchanged={"d"})
        assert "new=1" in repr(cs)
        assert "modified=1" in repr(cs)


# ── MetadataIndex.add_chunks ─────────────────────

class TestMetadataIndexAddChunks:
    def test_add_increases_total(self):
        idx = MetadataIndex()
        idx.build([TextChunk("a", {"grade": "六年级"})])
        assert idx.total_chunks == 1
        idx.add_chunks([TextChunk("b", {"grade": "七年级"})], start_id=1)
        assert idx.total_chunks == 2

    def test_add_chunks_findable(self):
        idx = MetadataIndex()
        idx.build([TextChunk("a", {"grade": "六年级"})])
        idx.add_chunks([TextChunk("b", {"grade": "七年级"})], start_id=1)
        assert idx.lookup({"grade": "七年级"}) == {1}
        assert idx.lookup({"grade": "六年级"}) == {0}

    def test_add_preserves_existing(self):
        idx = MetadataIndex()
        idx.build([TextChunk("a", {"grade": "六年级"})])
        original = idx.lookup({"grade": "六年级"})
        idx.add_chunks([TextChunk("b", {"grade": "七年级"})], start_id=1)
        assert idx.lookup({"grade": "六年级"}) == original

    def test_add_chunks_list_metadata(self):
        idx = MetadataIndex()
        idx.build([])
        idx.add_chunks([TextChunk("a", {"tags": ["动词", "时态"]})], start_id=0)
        assert idx.lookup({"tags": "动词"}) == {0}
        assert idx.lookup({"tags": "时态"}) == {0}


# ── Soft Delete ──────────────────────────────────

class TestSoftDelete:
    def _make_engine(self, chunks, soft_deleted_ids=None):
        bm25 = BM25Retriever()
        bm25.build(chunks)
        return HybridQueryEngine(
            [bm25],
            WeightedFusion(),
            soft_deleted_ids=soft_deleted_ids,
        )

    def test_soft_delete_filters_results(self):
        chunks = [
            TextChunk("Python编程", {"record_id": "q1"}),
            TextChunk("Java编程", {"record_id": "q2"}),
        ]
        engine = self._make_engine(chunks, soft_deleted_ids={"q1"})
        results = engine.retrieve("编程", top_k=5)
        record_ids = {r["metadata"]["record_id"] for r in results}
        assert "q1" not in record_ids
        assert "q2" in record_ids

    def test_add_and_clear_soft_deletes(self):
        chunks = [TextChunk("test", {"record_id": "q1"})]
        engine = self._make_engine(chunks)
        assert engine.soft_delete_count == 0
        engine.add_soft_deletes({"q1"})
        assert engine.soft_delete_count == 1
        engine.clear_soft_deletes()
        assert engine.soft_delete_count == 0

    def test_backward_compat_no_soft_deletes(self):
        """Default None doesn't break existing behavior."""
        chunks = [TextChunk("Python编程", {"record_id": "q1"})]
        engine = self._make_engine(chunks)
        results = engine.retrieve("编程", top_k=5)
        assert len(results) > 0


# ── build_incremental ────────────────────────────

class TestBuildIncremental:
    def _make_fc(self):
        """Create a FlowController with mocked embedder/indexer for testing."""
        from tinysearch.flow.controller import FlowController

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 10]

        mock_indexer = MagicMock()

        bm25 = BM25Retriever()
        substr = SubstringRetriever()
        metadata_index = MetadataIndex()

        engine = HybridQueryEngine(
            [bm25, substr],
            WeightedFusion([0.6, 0.4]),
            metadata_index=metadata_index,
        )

        fc = FlowController(
            data_adapter=None,
            text_splitter=MagicMock(),
            embedder=mock_embedder,
            indexer=mock_indexer,
            query_engine=engine,
            config={},
        )
        return fc

    def test_no_changes_returns_early(self):
        fc = self._make_fc()
        adapter = SimpleAdapter()
        tracker = ContentHashTracker()

        records = {"q1": {"text": "hello", "grade": "六年级"}}
        # First build
        tracker.update({rid: adapter.to_chunk(rid, r) for rid, r in records.items()})

        # Second call — no changes
        stats = fc.build_incremental(records, adapter, tracker)
        assert not stats["full_rebuild"]
        assert stats["new"] == 0
        assert stats["modified"] == 0

    def test_new_records_detected(self):
        fc = self._make_fc()
        adapter = SimpleAdapter()
        tracker = ContentHashTracker()

        records = {"q1": {"text": "hello", "grade": "六年级"}}
        stats = fc.build_incremental(records, adapter, tracker)
        assert stats["new"] == 1
        assert stats["deleted"] == 0

    def test_threshold_triggers_full_rebuild(self):
        fc = self._make_fc()
        adapter = SimpleAdapter()
        tracker = ContentHashTracker()

        # Build initial 150 records
        initial = {f"q{i}": {"text": f"text{i}"} for i in range(150)}
        tracker.update({rid: adapter.to_chunk(rid, r) for rid, r in initial.items()})

        # Delete all of them (150 > threshold 100)
        stats = fc.build_incremental({}, adapter, tracker, delete_rebuild_threshold=100)
        assert stats["full_rebuild"] is True
        assert stats["deleted"] == 150

    def test_stats_correct(self):
        fc = self._make_fc()
        adapter = SimpleAdapter()
        tracker = ContentHashTracker()

        # Initial build
        initial = {
            "q1": {"text": "unchanged"},
            "q2": {"text": "will_modify"},
            "q3": {"text": "will_delete"},
        }
        tracker.update({rid: adapter.to_chunk(rid, r) for rid, r in initial.items()})

        # Incremental update
        current = {
            "q1": {"text": "unchanged"},
            "q2": {"text": "modified_text"},
            "q4": {"text": "brand_new"},
        }
        stats = fc.build_incremental(current, adapter, tracker)
        assert stats["new"] == 1
        assert stats["modified"] == 1
        assert stats["deleted"] == 1
        assert stats["unchanged"] == 1
        assert not stats["full_rebuild"]
