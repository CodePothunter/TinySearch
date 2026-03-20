"""
Hybrid query engine - multi-retriever fusion with optional reranking
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from tinysearch.base import QueryEngine, Retriever, FusionStrategy, Reranker

logger = logging.getLogger(__name__)

# Type alias for filter values
FilterValue = Union[str, int, float, bool, List, Callable]


class HybridQueryEngine(QueryEngine):
    """
    Query engine that combines multiple retrievers via a fusion strategy,
    with optional reranking.

    Pipeline:
        1. Each retriever recalls top_k * recall_multiplier candidates
           (× filter_multiplier when metadata filters are active)
        2. Per-retriever min_score filtering
        3. Metadata post-filtering (if filters provided)
        4. FusionStrategy merges and deduplicates results
        5. Optional Reranker re-scores the fused candidates
        6. Return top_k final results
    """

    def __init__(
        self,
        retrievers: List[Retriever],
        fusion_strategy: FusionStrategy,
        reranker: Optional[Reranker] = None,
        recall_multiplier: int = 2,
        min_scores: Optional[List[float]] = None,
        filter_multiplier: int = 3,
    ):
        """
        Args:
            retrievers: List of Retriever instances for multi-path retrieval
            fusion_strategy: Strategy to fuse results from multiple retrievers
            reranker: Optional reranker for final re-scoring
            recall_multiplier: Multiply top_k by this for each retriever's recall
            min_scores: Per-retriever minimum score thresholds (length must match retrievers)
            filter_multiplier: Extra recall multiplier when filters are active
        """
        if not retrievers:
            raise ValueError("At least one retriever is required")
        if min_scores is not None and len(min_scores) != len(retrievers):
            raise ValueError(
                f"min_scores length ({len(min_scores)}) must match "
                f"retrievers length ({len(retrievers)})"
            )
        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker
        self.recall_multiplier = recall_multiplier
        self.min_scores = min_scores
        self.filter_multiplier = filter_multiplier

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

        # Compute recall amount — over-recall when filters are active
        recall_k = top_k * self.recall_multiplier
        if filters:
            recall_k *= self.filter_multiplier

        # Step 1: Recall from each retriever
        per_retriever: List[List[Dict[str, Any]]] = []
        for i, retriever in enumerate(self.retrievers):
            try:
                results = retriever.retrieve(query, top_k=recall_k)
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

        # Step 2: Metadata post-filtering
        if filters:
            per_retriever = [
                self._apply_filters(results, filters) for results in per_retriever
            ]

        # Step 3: Fuse results (pass dynamic weights if provided)
        fuse_kwargs: Dict[str, Any] = {}
        if weights is not None:
            fuse_kwargs["weights"] = weights
        fused = self.fusion_strategy.fuse(per_retriever, **fuse_kwargs)

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
