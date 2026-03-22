"""
Tests for RecordAdapter and build_chunks_from_records.
"""
import pytest
from typing import Any, Dict

from tinysearch.base import RecordAdapter, TextChunk, TextSplitter
from tinysearch.records import build_chunks_from_records


class SimpleAdapter(RecordAdapter):
    """Test adapter: extracts 'text' field, puts rest in metadata."""

    def to_chunk(self, record_id: str, record: Dict[str, Any]) -> TextChunk:
        return TextChunk(
            text=record.get("text", ""),
            metadata={"record_id": record_id, "grade": record.get("grade", "")},
        )


class TestRecordAdapterABC:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            RecordAdapter()

    def test_concrete_implementation(self):
        adapter = SimpleAdapter()
        chunk = adapter.to_chunk("q1", {"text": "hello", "grade": "六年级"})
        assert isinstance(chunk, TextChunk)
        assert chunk.text == "hello"
        assert chunk.metadata["record_id"] == "q1"
        assert chunk.metadata["grade"] == "六年级"


class TestBuildChunksFromRecords:
    def test_basic_no_splitter(self):
        adapter = SimpleAdapter()
        records = {
            "q1": {"text": "Python编程", "grade": "六年级"},
            "q2": {"text": "Java编程", "grade": "七年级"},
            "q3": {"text": "TinySearch", "grade": "八年级"},
        }
        chunks = build_chunks_from_records(records, adapter)
        assert len(chunks) == 3
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert [c.metadata["record_id"] for c in chunks] == ["q1", "q2", "q3"]

    def test_record_id_guaranteed_in_metadata(self):
        """Even if adapter doesn't set record_id, it's auto-added."""

        class NoIdAdapter(RecordAdapter):
            def to_chunk(self, rid, record):
                return TextChunk(text=record["text"], metadata={"grade": "test"})

        chunks = build_chunks_from_records(
            {"q1": {"text": "hello"}}, NoIdAdapter()
        )
        assert chunks[0].metadata["record_id"] == "q1"

    def test_with_splitter(self):
        """Long text gets split; metadata inherited."""
        from tinysearch.splitters import CharacterTextSplitter

        adapter = SimpleAdapter()
        # Create a record with text long enough to be split
        long_text = "A" * 500 + " " + "B" * 500
        records = {"q1": {"text": long_text, "grade": "六年级"}}

        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        chunks = build_chunks_from_records(records, adapter, splitter=splitter)

        assert len(chunks) > 1
        # All sub-chunks inherit the original metadata
        for c in chunks:
            assert c.metadata["record_id"] == "q1"
            assert c.metadata["grade"] == "六年级"
            assert "chunk_index" in c.metadata

    def test_empty_records(self):
        chunks = build_chunks_from_records({}, SimpleAdapter())
        assert chunks == []

    def test_order_preserved(self):
        adapter = SimpleAdapter()
        records = {"c": {"text": "C"}, "a": {"text": "A"}, "b": {"text": "B"}}
        chunks = build_chunks_from_records(records, adapter)
        assert [c.text for c in chunks] == ["C", "A", "B"]
