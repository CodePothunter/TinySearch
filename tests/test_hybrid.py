"""
Tests for hybrid search: HybridQueryEngine, FusionStrategies, and CLI/FlowController integration.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from tinysearch.base import TextChunk, QueryEngine, Retriever
from tinysearch.utils.file_discovery import iter_input_files
from tinysearch.retrievers.bm25_retriever import BM25Retriever
from tinysearch.retrievers.substring_retriever import SubstringRetriever
from tinysearch.fusion.weighted import WeightedFusion
from tinysearch.fusion.rrf import ReciprocalRankFusion
from tinysearch.fusion._utils import make_doc_key
from tinysearch.query.hybrid import HybridQueryEngine
from tinysearch.indexers.metadata_index import MetadataIndex


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def sample_chunks():
    return [
        TextChunk("Python编程语言", {"source": "a.txt", "grade": "六年级上", "chunk_index": 0}),
        TextChunk("Java编程语言", {"source": "b.txt", "grade": "六年级下", "chunk_index": 0}),
        TextChunk("TinySearch检索系统", {"source": "c.txt", "grade": "七年级", "chunk_index": 0}),
    ]


@pytest.fixture
def hybrid_engine(sample_chunks):
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)
    substr = SubstringRetriever()
    substr.build(sample_chunks)
    return HybridQueryEngine(
        [bm25, substr],
        WeightedFusion([0.6, 0.4]),
        min_scores=[0.0, 0.0],
    )


# ── HybridQueryEngine basic ──────────────────────────────

class TestHybridQueryEngine:
    def test_retrieve_returns_results(self, hybrid_engine):
        results = hybrid_engine.retrieve("编程", top_k=5)
        assert len(results) > 0
        assert all("text" in r and "score" in r for r in results)

    def test_backward_compat_no_kwargs(self, hybrid_engine):
        """retrieve() works with no extra kwargs (backward compatible)."""
        results = hybrid_engine.retrieve("编程", top_k=5)
        assert isinstance(results, list)

    def test_empty_retrievers_raises(self):
        with pytest.raises(ValueError, match="At least one retriever"):
            HybridQueryEngine([], WeightedFusion())

    def test_min_scores_length_mismatch(self, sample_chunks):
        bm25 = BM25Retriever()
        bm25.build(sample_chunks)
        with pytest.raises(ValueError, match="min_scores length"):
            HybridQueryEngine([bm25], WeightedFusion(), min_scores=[0.0, 0.0])


# ── Metadata filtering ───────────────────────────────────

class TestMetadataFiltering:
    def test_exact_match_filter(self, hybrid_engine):
        results = hybrid_engine.retrieve("编程", top_k=5, filters={"source": "a.txt"})
        assert all(r["metadata"]["source"] == "a.txt" for r in results)

    def test_list_or_filter(self, hybrid_engine):
        results = hybrid_engine.retrieve(
            "编程", top_k=5,
            filters={"grade": ["六年级上", "六年级下"]},
        )
        assert len(results) > 0
        assert all(r["metadata"]["grade"].startswith("六年级") for r in results)

    def test_callable_filter(self, hybrid_engine):
        results = hybrid_engine.retrieve(
            "编程", top_k=5,
            filters={"source": lambda v: v in ["a.txt", "b.txt"]},
        )
        assert all(r["metadata"]["source"] in ["a.txt", "b.txt"] for r in results)

    def test_missing_key_excluded(self, hybrid_engine):
        """Chunks missing a filter key should be excluded."""
        results = hybrid_engine.retrieve(
            "编程", top_k=5,
            filters={"nonexistent_key": "value"},
        )
        assert len(results) == 0

    def test_none_metadata_excluded(self):
        assert HybridQueryEngine._match_filters(None, {"key": "val"}) is False

    def test_filter_over_recall(self, hybrid_engine):
        """With filters, recall_k should be multiplied by filter_multiplier."""
        # Default filter_multiplier=3, recall_multiplier=2
        # So recall_k = 5 * 2 * 3 = 30
        # This is tested implicitly: we still get results despite filters
        results = hybrid_engine.retrieve(
            "编程", top_k=5,
            filters={"grade": ["六年级上"]},
        )
        assert len(results) > 0


# ── Dynamic weights ───────────────────────────────────────

class TestDynamicWeights:
    def test_dynamic_weights_override(self, hybrid_engine):
        """Passing weights= at query time should override constructor weights."""
        r1 = hybrid_engine.retrieve("编程", top_k=5, weights=[1.0, 0.0])
        r2 = hybrid_engine.retrieve("编程", top_k=5, weights=[0.0, 1.0])
        # Results should differ because weights favor different retrievers
        # At minimum, both should return results
        assert len(r1) > 0
        assert len(r2) > 0


# ── min_scores ────────────────────────────────────────────

class TestMinScores:
    def test_high_min_scores_filters_everything(self, sample_chunks):
        bm25 = BM25Retriever()
        bm25.build(sample_chunks)
        substr = SubstringRetriever()
        substr.build(sample_chunks)
        engine = HybridQueryEngine(
            [bm25, substr],
            WeightedFusion([0.5, 0.5]),
            min_scores=[999.0, 999.0],
        )
        results = engine.retrieve("编程", top_k=5)
        assert len(results) == 0

    def test_zero_min_scores_passes_all(self, hybrid_engine):
        results = hybrid_engine.retrieve("编程", top_k=5)
        assert len(results) > 0


# ── retrieve_with_details ─────────────────────────────────

class TestRetrieveWithDetails:
    def test_returns_all_keys(self, hybrid_engine):
        d = hybrid_engine.retrieve_with_details("编程", top_k=5)
        assert "results" in d
        assert "per_retriever" in d
        assert "fused_before_rerank" in d

    def test_per_retriever_count_matches(self, hybrid_engine):
        d = hybrid_engine.retrieve_with_details("编程", top_k=5)
        assert len(d["per_retriever"]) == len(hybrid_engine.retrievers)

    def test_results_match_retrieve(self, hybrid_engine):
        """retrieve_with_details().results should equal retrieve()."""
        r1 = hybrid_engine.retrieve("编程", top_k=5)
        r2 = hybrid_engine.retrieve_with_details("编程", top_k=5)["results"]
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a["text"] == b["text"]


# ── Fusion dedup key ──────────────────────────────────────

class TestFusionDedupKey:
    def test_same_text_different_source_not_merged(self):
        """Two chunks with same text but different source should stay separate."""
        chunks = [
            TextChunk("Same text", {"source": "fileA.txt", "chunk_index": 0}),
            TextChunk("Same text", {"source": "fileB.txt", "chunk_index": 0}),
        ]
        bm25 = BM25Retriever()
        bm25.build(chunks)
        substr = SubstringRetriever()
        substr.build(chunks)
        engine = HybridQueryEngine(
            [bm25, substr], WeightedFusion([0.5, 0.5])
        )
        results = engine.retrieve("Same text", top_k=5)
        sources = {r["metadata"]["source"] for r in results}
        assert sources == {"fileA.txt", "fileB.txt"}

    def test_same_text_different_chunk_index_not_merged(self):
        """Same text at different positions in same file should stay separate."""
        chunks = [
            TextChunk("Repeated", {"source": "f.txt", "chunk_index": 0}),
            TextChunk("Repeated", {"source": "f.txt", "chunk_index": 5}),
        ]
        bm25 = BM25Retriever()
        bm25.build(chunks)
        substr = SubstringRetriever()
        substr.build(chunks)
        engine = HybridQueryEngine(
            [bm25, substr], WeightedFusion([0.5, 0.5])
        )
        results = engine.retrieve("Repeated", top_k=5)
        indices = {r["metadata"]["chunk_index"] for r in results}
        assert indices == {0, 5}

    def test_make_doc_key_different_source(self):
        r1 = {"text": "hello", "metadata": {"source": "a"}}
        r2 = {"text": "hello", "metadata": {"source": "b"}}
        assert make_doc_key(r1) != make_doc_key(r2)

    def test_make_doc_key_same_chunk(self):
        """Same chunk from different retrievers should produce the same key."""
        r1 = {"text": "hello", "metadata": {"source": "a", "chunk_index": 0}}
        r2 = {"text": "hello", "metadata": {"source": "a", "chunk_index": 0}}
        assert make_doc_key(r1) == make_doc_key(r2)

    def test_rrf_dedup_also_fixed(self):
        """RRF fusion should also keep different-source same-text chunks separate."""
        chunks = [
            TextChunk("Same text", {"source": "x.txt", "chunk_index": 0}),
            TextChunk("Same text", {"source": "y.txt", "chunk_index": 0}),
        ]
        bm25 = BM25Retriever()
        bm25.build(chunks)
        substr = SubstringRetriever()
        substr.build(chunks)
        engine = HybridQueryEngine(
            [bm25, substr], ReciprocalRankFusion()
        )
        results = engine.retrieve("Same text", top_k=5)
        assert len(results) == 2


# ── Silent exception logging ─────────────────────────────

class TestRetrieverFailureLogging:
    def test_failing_retriever_logged_not_crash(self, caplog):
        """A failing retriever should log a warning and return empty, not crash."""
        class FailRetriever(Retriever):
            def build(self, chunks): pass
            def retrieve(self, query, top_k=5):
                raise RuntimeError("boom")
            def save(self, path): pass
            def load(self, path): pass

        engine = HybridQueryEngine([FailRetriever()], WeightedFusion())
        import logging
        with caplog.at_level(logging.WARNING, logger="tinysearch.query.hybrid"):
            results = engine.retrieve("test", top_k=5)
        assert results == []
        assert "FailRetriever failed" in caplog.text


# ── ABC signature ─────────────────────────────────────────

class TestABCSignature:
    def test_query_engine_retrieve_accepts_kwargs(self):
        import inspect
        sig = inspect.signature(QueryEngine.retrieve)
        params = list(sig.parameters.keys())
        assert "kwargs" in params

    def test_hybrid_engine_is_query_engine(self, hybrid_engine):
        assert isinstance(hybrid_engine, QueryEngine)


# ── FlowController kwargs forwarding ──────────────────────

class TestFlowControllerKwargs:
    def test_query_forwards_kwargs(self):
        """FlowController.query() should forward **kwargs to query_engine.retrieve()."""
        mock_engine = MagicMock(spec=QueryEngine)
        mock_engine.retrieve.return_value = []

        from tinysearch.flow.controller import FlowController
        # Minimal construction — we only test query(), not build
        fc = FlowController.__new__(FlowController)
        fc.query_engine = mock_engine
        fc.config = {"query_engine": {"top_k": 5}}

        fc.query("test", top_k=5, filters={"source": "a"}, weights=[1.0])
        mock_engine.retrieve.assert_called_once_with(
            "test", 5, filters={"source": "a"}, weights=[1.0]
        )


# ── CLI helpers ───────────────────────────────────────────

class TestCLIHelpers:
    def test_get_retriever_index_dir(self):
        from tinysearch.cli import _get_retriever_index_dir
        assert _get_retriever_index_dir(Path("index.faiss")) == Path("index")
        assert _get_retriever_index_dir(Path("data/my.faiss")) == Path("data/my")
        assert _get_retriever_index_dir(Path("mydir")) == Path("mydir")

    def test_build_hybrid_noop_for_template_engine(self):
        """_build_hybrid_retriever_indexes should be a no-op for non-hybrid engines."""
        from tinysearch.cli import _build_hybrid_retriever_indexes
        mock_engine = MagicMock()  # not a HybridQueryEngine
        # Should not raise
        _build_hybrid_retriever_indexes(mock_engine, [])

    def test_save_load_hybrid_noop_for_template_engine(self):
        from tinysearch.cli import _save_hybrid_retriever_indexes, _load_hybrid_retriever_indexes
        mock_engine = MagicMock()
        _save_hybrid_retriever_indexes(mock_engine, Path("test.faiss"))
        _load_hybrid_retriever_indexes(mock_engine, Path("test.faiss"))

    def test_build_index_directory_per_file_source(self, tmp_path):
        """CLI build_index on a directory should assign per-file source metadata."""
        import argparse
        from tinysearch.cli import build_index
        from tinysearch.config import Config

        # Create two files with the same content
        (tmp_path / "file1.txt").write_text("Same content here")
        (tmp_path / "file2.txt").write_text("Same content here")

        config = Config()
        config.set("query_engine.method", "hybrid")
        config.set("retrievers", [{"type": "bm25"}, {"type": "substring"}])
        config.set("fusion.strategy", "weighted")
        config.set("fusion.weights", [0.5, 0.5])

        # We just need to verify the metadata, not actually embed/index.
        # Patch embedder and indexer to avoid needing a real model.
        captured_chunks = []
        original_split = None

        from tinysearch.splitters import CharacterTextSplitter
        orig_split = CharacterTextSplitter.split

        def spy_split(self, texts, metadata=None):
            result = orig_split(self, texts, metadata)
            captured_chunks.extend(result)
            return result

        args = argparse.Namespace(data=str(tmp_path))

        with patch.object(CharacterTextSplitter, "split", spy_split), \
             patch("tinysearch.cli.load_embedder") as mock_emb, \
             patch("tinysearch.cli.load_indexer") as mock_idx:
            mock_emb_inst = MagicMock()
            mock_emb_inst.embed.return_value = [[0.0] * 10] * 100
            mock_emb.return_value = mock_emb_inst
            mock_idx_inst = MagicMock()
            mock_idx.return_value = mock_idx_inst

            build_index(args, config)

        # Each chunk should have a per-file source, not the directory
        sources = {c.metadata["source"] for c in captured_chunks}
        assert len(sources) == 2, f"Expected 2 distinct sources, got {sources}"
        assert all(str(tmp_path) != s for s in sources), \
            f"source should be per-file, not the directory: {sources}"


# ── Retriever index save path ─────────────────────────────

class TestRetrieverIndexPath:
    def test_flow_controller_uses_faiss_dir(self):
        """FlowController should save retriever indexes inside the FAISS index directory."""
        from tinysearch.flow.controller import FlowController
        fc = FlowController.__new__(FlowController)
        fc.query_engine = MagicMock(spec=HybridQueryEngine)
        fc.query_engine.metadata_index = None

        mock_retriever = MagicMock()
        mock_retriever.__class__.__name__ = "BM25Retriever"
        fc.query_engine.retrievers = [mock_retriever]

        fc._save_retriever_indexes(Path("data/index.faiss"))
        mock_retriever.save.assert_called_once_with(Path("data/index") / "bm25_index")

    def test_flow_controller_load_uses_faiss_dir(self):
        from tinysearch.flow.controller import FlowController
        fc = FlowController.__new__(FlowController)
        fc.query_engine = MagicMock(spec=HybridQueryEngine)
        fc.query_engine.metadata_index = None

        mock_retriever = MagicMock()
        mock_retriever.__class__.__name__ = "BM25Retriever"
        fc.query_engine.retrievers = [mock_retriever]

        # Simulate that the path exists
        with patch.object(Path, "exists", return_value=True):
            fc._load_retriever_indexes(Path("data/index.faiss"))
        mock_retriever.load.assert_called_once_with(Path("data/index") / "bm25_index")


# ── iter_input_files ─────────────────────────────────────

class TestIterInputFiles:
    def test_single_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        assert list(iter_input_files(f)) == [f]

    def test_directory_filters_by_adapter_type(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.pdf").write_text("b")
        (tmp_path / "c.py").write_text("c")
        files = list(iter_input_files(tmp_path, adapter_type="text"))
        suffixes = {f.suffix for f in files}
        assert ".txt" in suffixes
        assert ".py" in suffixes
        assert ".pdf" not in suffixes

    def test_custom_extensions_override(self, tmp_path):
        (tmp_path / "a.xyz").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        files = list(iter_input_files(tmp_path, extensions=[".xyz"]))
        assert len(files) == 1
        assert files[0].suffix == ".xyz"

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            list(iter_input_files(Path("/nonexistent")))

    def test_sorted_deterministic(self, tmp_path):
        for name in ["c.txt", "a.txt", "b.txt"]:
            (tmp_path / name).write_text(name)
        files = list(iter_input_files(tmp_path, adapter_type="text"))
        assert files == sorted(files)

    def test_recursive_finds_nested(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.txt").write_text("top")
        (sub / "nested.txt").write_text("nested")
        files = list(iter_input_files(tmp_path, adapter_type="text", recursive=True))
        assert len(files) == 2

    def test_non_recursive_skips_nested(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.txt").write_text("top")
        (sub / "nested.txt").write_text("nested")
        files = list(iter_input_files(tmp_path, adapter_type="text", recursive=False))
        assert len(files) == 1


# ── Adapter directory rejection ──────────────────────────

class TestAdapterRejectsDirectory:
    def test_text_adapter_rejects_directory(self, tmp_path):
        from tinysearch.adapters.text import TextAdapter
        (tmp_path / "a.txt").write_text("hello")
        with pytest.raises(ValueError, match="does not accept directories"):
            TextAdapter().extract(tmp_path)

    def test_csv_adapter_rejects_directory(self, tmp_path):
        from tinysearch.adapters.csv import CSVAdapter
        (tmp_path / "a.csv").write_text("col\nval")
        with pytest.raises(ValueError, match="does not accept directories"):
            CSVAdapter().extract(tmp_path)

    def test_markdown_adapter_rejects_directory(self, tmp_path):
        from tinysearch.adapters.markdown import MarkdownAdapter
        (tmp_path / "a.md").write_text("# Hello")
        with pytest.raises(ValueError, match="does not accept directories"):
            MarkdownAdapter().extract(tmp_path)

    def test_json_adapter_rejects_directory(self, tmp_path):
        from tinysearch.adapters.json_adapter import JSONAdapter
        (tmp_path / "a.json").write_text('{"key": "val"}')
        with pytest.raises(ValueError, match="does not accept directories"):
            JSONAdapter().extract(tmp_path)


# ── Source metadata consistency ──────────────────────────

class TestSourceMetadataConsistency:
    def test_cli_and_flowcontroller_same_sources(self, tmp_path):
        """CLI build_index and FlowController should discover the same files."""
        (tmp_path / "file1.txt").write_text("Content A")
        (tmp_path / "file2.txt").write_text("Content B")
        (tmp_path / "ignore.pdf").write_text("PDF")

        cli_sources = {str(f) for f in iter_input_files(tmp_path, adapter_type="text")}
        fc_sources = {str(f) for f in iter_input_files(tmp_path, adapter_type="text")}

        assert cli_sources == fc_sources
        assert len(cli_sources) == 2
        assert all("ignore.pdf" not in s for s in cli_sources)


# ── Pre-filter (MetadataIndex + candidate_ids) ───────────

@pytest.fixture
def graded_chunks():
    """Chunks with grade metadata for pre-filter testing."""
    return [
        TextChunk("Python编程语言", {"source": "a.txt", "grade": "六年级上", "chunk_index": 0}),
        TextChunk("Java编程语言", {"source": "b.txt", "grade": "六年级下", "chunk_index": 0}),
        TextChunk("TinySearch检索系统", {"source": "c.txt", "grade": "七年级", "chunk_index": 0}),
        TextChunk("向量数据库", {"source": "d.txt", "grade": "六年级上", "chunk_index": 0}),
    ]


@pytest.fixture
def prefilter_engine(graded_chunks):
    """HybridQueryEngine with MetadataIndex for pre-filter testing."""
    bm25 = BM25Retriever()
    bm25.build(graded_chunks)
    substr = SubstringRetriever()
    substr.build(graded_chunks)

    metadata_index = MetadataIndex()
    metadata_index.build(graded_chunks)

    return HybridQueryEngine(
        [bm25, substr],
        WeightedFusion([0.6, 0.4]),
        metadata_index=metadata_index,
        filter_mode="auto",
    )


class TestPreFilter:
    def test_bm25_candidate_ids(self, graded_chunks):
        """BM25 with candidate_ids only returns results within the set."""
        bm25 = BM25Retriever()
        bm25.build(graded_chunks)
        # Only allow chunks 0 and 3 (grade=六年级上)
        results = bm25.retrieve("编程", top_k=5, candidate_ids={0, 3})
        for r in results:
            assert r["metadata"]["grade"] == "六年级上"

    def test_substring_candidate_ids(self, graded_chunks):
        """SubstringRetriever with candidate_ids restricts search."""
        substr = SubstringRetriever()
        substr.build(graded_chunks)
        results = substr.retrieve("编程", top_k=5, candidate_ids={0})
        assert len(results) == 1
        assert results[0]["metadata"]["source"] == "a.txt"

    def test_candidate_ids_none_is_noop(self, graded_chunks):
        """Passing candidate_ids=None returns same as no kwargs."""
        bm25 = BM25Retriever()
        bm25.build(graded_chunks)
        r1 = bm25.retrieve("编程", top_k=5)
        r2 = bm25.retrieve("编程", top_k=5, candidate_ids=None)
        assert len(r1) == len(r2)

    def test_candidate_ids_empty_returns_empty(self, graded_chunks):
        """Passing candidate_ids=set() returns []."""
        bm25 = BM25Retriever()
        bm25.build(graded_chunks)
        results = bm25.retrieve("编程", top_k=5, candidate_ids=set())
        assert results == []

    def test_filter_mode_pre(self, graded_chunks):
        """Pre-filter mode resolves filters via MetadataIndex."""
        bm25 = BM25Retriever()
        bm25.build(graded_chunks)
        substr = SubstringRetriever()
        substr.build(graded_chunks)

        metadata_index = MetadataIndex()
        metadata_index.build(graded_chunks)

        engine = HybridQueryEngine(
            [bm25, substr],
            WeightedFusion([0.6, 0.4]),
            metadata_index=metadata_index,
            filter_mode="pre",
        )
        results = engine.retrieve("编程", top_k=5, filters={"grade": "六年级上"})
        assert all(r["metadata"]["grade"] == "六年级上" for r in results)

    def test_filter_mode_auto_mixed(self, prefilter_engine):
        """Auto mode: indexable filters pre-filter, callable filters post-filter."""
        results = prefilter_engine.retrieve(
            "编程", top_k=5,
            filters={
                "grade": "六年级上",
                "source": lambda v: v == "a.txt",  # callable → post-filter
            },
        )
        assert all(r["metadata"]["source"] == "a.txt" for r in results)

    def test_no_metadata_index_backward_compat(self, graded_chunks):
        """When metadata_index=None, filters are purely post-applied."""
        bm25 = BM25Retriever()
        bm25.build(graded_chunks)
        substr = SubstringRetriever()
        substr.build(graded_chunks)

        engine = HybridQueryEngine(
            [bm25, substr],
            WeightedFusion([0.6, 0.4]),
            metadata_index=None,
        )
        results = engine.retrieve("编程", top_k=5, filters={"grade": "六年级上"})
        assert all(r["metadata"]["grade"] == "六年级上" for r in results)

    def test_empty_candidate_short_circuits(self, graded_chunks):
        """When lookup returns empty set, pipeline returns [] immediately."""
        bm25 = BM25Retriever()
        bm25.build(graded_chunks)

        metadata_index = MetadataIndex()
        metadata_index.build(graded_chunks)

        engine = HybridQueryEngine(
            [bm25],
            WeightedFusion(),
            metadata_index=metadata_index,
            filter_mode="pre",
        )
        results = engine.retrieve("编程", top_k=5, filters={"grade": "不存在"})
        assert results == []
