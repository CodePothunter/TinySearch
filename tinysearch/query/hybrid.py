"""
Hybrid query engine - multi-retriever fusion with optional reranking
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tinysearch.base import QueryEngine, Retriever, FusionStrategy, Reranker

logger = logging.getLogger(__name__)

# Type alias for filter values
FilterValue = Union[str, int, float, bool, List, Callable]


class HybridQueryEngine(QueryEngine):
    """
    Query engine that combines multiple retrievers via a fusion strategy,
    with optional reranking.

    Pipeline:
        1. (If filter_mode is "pre" or "auto") Use MetadataIndex to resolve
           indexable filters into a candidate_ids set; pass to retrievers
        2. Each retriever recalls top_k * recall_multiplier candidates
           (× filter_multiplier only when post-filters are active)
        3. Per-retriever min_score filtering
        4. Metadata post-filtering for callable/non-indexable filters
        5. FusionStrategy merges and deduplicates results
        6. Optional Reranker re-scores the fused candidates
        7. Return top_k final results
    """

    def __init__(
        self,
        retrievers: List[Retriever],
        fusion_strategy: FusionStrategy,
        reranker: Optional[Reranker] = None,
        recall_multiplier: int = 2,
        min_scores: Optional[List[float]] = None,
        filter_multiplier: int = 3,
        metadata_index=None,
        filter_mode: str = "auto",
        soft_deleted_ids: Optional[set] = None,
    ):
        """
        Args:
            retrievers: List of Retriever instances for multi-path retrieval
            fusion_strategy: Strategy to fuse results from multiple retrievers
            reranker: Optional reranker for final re-scoring
            recall_multiplier: Multiply top_k by this for each retriever's recall
            min_scores: Per-retriever minimum score thresholds (length must match retrievers)
            filter_multiplier: Extra recall multiplier when post-filters are active
            metadata_index: Optional MetadataIndex for inverted-index pre-filtering
            filter_mode: "pre" (always pre-filter), "post" (always post-filter),
                         or "auto" (pre-filter indexable parts, post-filter callables)
            soft_deleted_ids: Optional set of record_ids to exclude from results
        """
        if not retrievers:
            raise ValueError("At least one retriever is required")
        if min_scores is not None and len(min_scores) != len(retrievers):
            raise ValueError(
                f"min_scores length ({len(min_scores)}) must match "
                f"retrievers length ({len(retrievers)})"
            )
        if filter_mode not in ("pre", "post", "auto"):
            raise ValueError(f"filter_mode must be 'pre', 'post', or 'auto', got '{filter_mode}'")
        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker
        self.recall_multiplier = recall_multiplier
        self.min_scores = min_scores
        self.filter_multiplier = filter_multiplier
        self.metadata_index = metadata_index
        self.filter_mode = filter_mode
        self.soft_deleted_ids: set = soft_deleted_ids or set()

    def add_soft_deletes(self, record_ids: set) -> None:
        """Mark record_ids as soft-deleted (excluded from results)."""
        self.soft_deleted_ids |= record_ids

    def clear_soft_deletes(self) -> None:
        """Clear all soft deletes (typically after a full rebuild)."""
        self.soft_deleted_ids.clear()

    @property
    def soft_delete_count(self) -> int:
        return len(self.soft_deleted_ids)

    def format_query(self, query: str) -> str:
        """Pass-through: hybrid engine doesn't transform queries"""
        return query

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Multi-path retrieval with fusion and optional reranking.

        Args:
            query: Query string
            top_k: Number of final results to return
            **kwargs:
                filters: Dict of metadata filters (see _match_filters)
                weights: List of floats to override fusion weights dynamically

        Returns:
            Fused (and optionally reranked) list of results
        """
        return self._retrieve_pipeline(query, top_k, **kwargs)["results"]

    def retrieve_with_details(
        self, query: str, top_k: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """
        Like retrieve(), but returns structured details of each pipeline stage.

        Returns:
            Dict with keys:
                results: Final top_k results
                per_retriever: List of per-retriever raw results (after min_score)
                fused_before_rerank: Fused results before reranking
        """
        return self._retrieve_pipeline(query, top_k, **kwargs)

    def _retrieve_pipeline(
        self, query: str, top_k: int, **kwargs
    ) -> Dict[str, Any]:
        """Core pipeline shared by retrieve() and retrieve_with_details()."""
        filters = kwargs.pop("filters", None)
        weights = kwargs.pop("weights", None)

        # Resolve pre-filter vs post-filter strategy
        candidate_ids = None
        post_filters = None

        if filters and self.metadata_index is not None:
            if self.filter_mode == "pre":
                candidate_ids = self.metadata_index.lookup(filters)
                if candidate_ids is None:
                    # Has callable filters, fall back to post-filter
                    post_filters = filters
                elif len(candidate_ids) == 0:
                    return {
                        "results": [],
                        "per_retriever": [[] for _ in self.retrievers],
                        "fused_before_rerank": [],
                    }
            elif self.filter_mode == "post":
                post_filters = filters
            else:  # auto
                indexable, callables = self.metadata_index.classify_filters(filters)
                if indexable:
                    candidate_ids = self.metadata_index.lookup(indexable)
                    if candidate_ids is not None and len(candidate_ids) == 0:
                        return {
                            "results": [],
                            "per_retriever": [[] for _ in self.retrievers],
                            "fused_before_rerank": [],
                        }
                if callables:
                    post_filters = callables
        elif filters:
            # No metadata_index available, always post-filter
            post_filters = filters

        # Compute recall amount — filter_multiplier only when post-filtering
        recall_k = top_k * self.recall_multiplier
        if post_filters:
            recall_k *= self.filter_multiplier

        # Step 1: Recall from each retriever
        per_retriever: List[List[Dict[str, Any]]] = []
        for i, retriever in enumerate(self.retrievers):
            try:
                retriever_kwargs: Dict[str, Any] = {}
                if candidate_ids is not None:
                    retriever_kwargs["candidate_ids"] = candidate_ids
                results = retriever.retrieve(query, top_k=recall_k, **retriever_kwargs)
            except Exception as e:
                retriever_name = type(retriever).__name__
                logger.warning(
                    "Retriever %s failed, skipping: %s", retriever_name, e
                )
                results = []

            # Per-retriever min_score filtering
            if self.min_scores is not None:
                threshold = self.min_scores[i]
                results = [r for r in results if r.get("score", 0) >= threshold]

            per_retriever.append(results)

        # Step 2: Post-filtering (only for callable/non-indexable filters)
        if post_filters:
            per_retriever = [
                self._apply_filters(results, post_filters) for results in per_retriever
            ]

        # Step 3: Fuse results (pass dynamic weights if provided)
        fuse_kwargs: Dict[str, Any] = {}
        if weights is not None:
            fuse_kwargs["weights"] = weights
        fused = self.fusion_strategy.fuse(per_retriever, **fuse_kwargs)

        # Step 3.5: Remove soft-deleted results
        if self.soft_deleted_ids:
            fused = [
                r for r in fused
                if r.get("metadata", {}).get("record_id") not in self.soft_deleted_ids
            ]

        fused_before_rerank = list(fused)

        # Step 4: Optional reranking
        if self.reranker is not None and fused:
            fused = self.reranker.rerank(query, fused, top_k=top_k)

        final = fused[:top_k]

        return {
            "results": final,
            "per_retriever": per_retriever,
            "fused_before_rerank": fused_before_rerank,
        }

    @staticmethod
    def _match_filters(
        metadata: Optional[Dict[str, Any]], filters: Dict[str, FilterValue]
    ) -> bool:
        """
        Check whether a metadata dict matches all filter criteria.

        Filter syntax (all keys are AND-ed):
            - str/int/float/bool → exact match
            - list → match any value in the list (OR)
            - callable → predicate function returning bool

        A result with missing metadata or missing a required key does NOT pass.
        """
        if metadata is None:
            return False
        for key, condition in filters.items():
            if key not in metadata:
                return False
            value = metadata[key]
            if callable(condition):
                if not condition(value):
                    return False
            elif isinstance(condition, list):
                if value not in condition:
                    return False
            else:
                if value != condition:
                    return False
        return True

    @classmethod
    def _apply_filters(
        cls, results: List[Dict[str, Any]], filters: Dict[str, FilterValue]
    ) -> List[Dict[str, Any]]:
        """Filter a list of results by metadata criteria."""
        return [r for r in results if cls._match_filters(r.get("metadata"), filters)]
