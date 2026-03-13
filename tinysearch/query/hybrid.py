"""
Hybrid query engine - multi-retriever fusion with optional reranking
"""
from typing import Any, Dict, List, Optional

from tinysearch.base import QueryEngine, Retriever, FusionStrategy, Reranker


class HybridQueryEngine(QueryEngine):
    """
    Query engine that combines multiple retrievers via a fusion strategy,
    with optional reranking.

    Pipeline:
        1. Each retriever recalls top_k * recall_multiplier candidates
        2. FusionStrategy merges and deduplicates results
        3. Optional Reranker re-scores the fused candidates
        4. Return top_k final results
    """

    def __init__(
        self,
        retrievers: List[Retriever],
        fusion_strategy: FusionStrategy,
        reranker: Optional[Reranker] = None,
        recall_multiplier: int = 2,
    ):
        """
        Args:
            retrievers: List of Retriever instances for multi-path retrieval
            fusion_strategy: Strategy to fuse results from multiple retrievers
            reranker: Optional reranker for final re-scoring
            recall_multiplier: Multiply top_k by this for each retriever's recall
        """
        if not retrievers:
            raise ValueError("At least one retriever is required")
        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker
        self.recall_multiplier = recall_multiplier

    def format_query(self, query: str) -> str:
        """Pass-through: hybrid engine doesn't transform queries"""
        return query

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Multi-path retrieval with fusion and optional reranking.

        Args:
            query: Query string
            top_k: Number of final results to return

        Returns:
            Fused (and optionally reranked) list of results
        """
        recall_k = top_k * self.recall_multiplier

        # Step 1: Recall from each retriever
        all_results = []
        for retriever in self.retrievers:
            try:
                results = retriever.retrieve(query, top_k=recall_k)
                all_results.append(results)
            except Exception:
                # If a retriever fails, skip it rather than failing entirely
                all_results.append([])

        # Step 2: Fuse results
        fused = self.fusion_strategy.fuse(all_results)

        # Step 3: Optional reranking
        if self.reranker is not None and fused:
            fused = self.reranker.rerank(query, fused, top_k=top_k)

        return fused[:top_k]
