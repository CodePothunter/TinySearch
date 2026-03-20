"""
Tests for MetadataIndex: inverted index over TextChunk metadata.
"""
import pytest
from pathlib import Path

from tinysearch.base import TextChunk
from tinysearch.indexers.metadata_index import MetadataIndex


@pytest.fixture
def sample_chunks():
    return [
        TextChunk("Q1", {"grade": "六年级上", "type": "选择题", "difficulty": 1}),
        TextChunk("Q2", {"grade": "六年级下", "type": "填空题", "difficulty": 2}),
        TextChunk("Q3", {"grade": "六年级上", "type": "填空题", "difficulty": 3}),
        TextChunk("Q4", {"grade": "七年级", "tags": ["动词", "时态"], "difficulty": 1}),
        TextChunk("Q5", {"grade": "七年级", "tags": ["名词", "复数"]}),
    ]


@pytest.fixture
def built_index(sample_chunks):
    idx = MetadataIndex()
    idx.build(sample_chunks)
    return idx


# ── Build & Lookup ─────────────────────────────

class TestBuildAndLookup:
    def test_build_basic(self, built_index):
        assert built_index.total_chunks == 5
        assert "grade" in built_index.fields
        assert "type" in built_index.fields
        assert "tags" in built_index.fields

    def test_lookup_exact_str(self, built_index):
        result = built_index.lookup({"grade": "六年级上"})
        assert result == {0, 2}

    def test_lookup_exact_int(self, built_index):
        result = built_index.lookup({"difficulty": 1})
        assert result == {0, 3}

    def test_lookup_list_filter(self, built_index):
        """List filter = OR over values."""
        result = built_index.lookup({"grade": ["六年级上", "六年级下"]})
        assert result == {0, 1, 2}

    def test_lookup_multiple_filters_intersect(self, built_index):
        """Multiple filters = AND (set intersection)."""
        result = built_index.lookup({"grade": "六年级上", "type": "填空题"})
        assert result == {2}

    def test_lookup_callable_returns_none(self, built_index):
        result = built_index.lookup({"grade": lambda v: True})
        assert result is None

    def test_lookup_nonexistent_field(self, built_index):
        result = built_index.lookup({"nonexistent": "value"})
        assert result == set()

    def test_lookup_nonexistent_value(self, built_index):
        result = built_index.lookup({"grade": "不存在的年级"})
        assert result == set()

    def test_list_metadata_indexed_per_element(self, built_index):
        """List[str] metadata: each element indexed separately."""
        assert built_index.lookup({"tags": "动词"}) == {3}
        assert built_index.lookup({"tags": "名词"}) == {4}
        assert built_index.lookup({"tags": ["动词", "名词"]}) == {3, 4}

    def test_empty_filters_returns_none(self, built_index):
        assert built_index.lookup({}) is None

    def test_empty_intersection_short_circuits(self, built_index):
        """When first filter yields empty, result is empty."""
        result = built_index.lookup({"grade": "不存在", "type": "选择题"})
        assert result == set()


# ── Classify Filters ─────────────────────────────

class TestClassifyFilters:
    def test_all_indexable(self, built_index):
        indexable, callables = built_index.classify_filters({
            "grade": "六年级上",
            "type": ["选择题", "填空题"],
        })
        assert len(indexable) == 2
        assert len(callables) == 0

    def test_all_callable(self, built_index):
        indexable, callables = built_index.classify_filters({
            "grade": lambda v: "六" in v,
        })
        assert len(indexable) == 0
        assert len(callables) == 1

    def test_mixed(self, built_index):
        indexable, callables = built_index.classify_filters({
            "grade": "六年级上",
            "difficulty": lambda v: v > 2,
        })
        assert "grade" in indexable
        assert "difficulty" in callables


# ── Save / Load ────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, built_index, tmp_path):
        save_path = tmp_path / "metadata_index.json"
        built_index.save(save_path)

        loaded = MetadataIndex()
        loaded.load(save_path)

        assert loaded.total_chunks == built_index.total_chunks
        assert loaded.lookup({"grade": "六年级上"}) == {0, 2}
        assert loaded.lookup({"difficulty": 1}) == {0, 3}
        assert loaded.lookup({"tags": "动词"}) == {3}

    def test_save_creates_parent_dirs(self, built_index, tmp_path):
        save_path = tmp_path / "deep" / "nested" / "idx.json"
        built_index.save(save_path)
        assert save_path.exists()


# ── Edge Cases ─────────────────────────────────

class TestEdgeCases:
    def test_empty_chunks(self):
        idx = MetadataIndex()
        idx.build([])
        assert idx.total_chunks == 0
        assert idx.lookup({"key": "val"}) == set()

    def test_chunks_with_no_metadata(self):
        idx = MetadataIndex()
        idx.build([TextChunk("text", None), TextChunk("text2", {})])
        assert idx.total_chunks == 2
        assert idx.lookup({"any": "thing"}) == set()

    def test_bool_metadata(self):
        idx = MetadataIndex()
        idx.build([
            TextChunk("a", {"active": True}),
            TextChunk("b", {"active": False}),
        ])
        assert idx.lookup({"active": True}) == {0}
        assert idx.lookup({"active": False}) == {1}
