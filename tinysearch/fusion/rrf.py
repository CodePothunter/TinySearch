"""
Reciprocal Rank Fusion (RRF) strategy
"""
from collections import defaultdict
from typing import Any, Dict, List

from tinysearch.base import FusionStrategy


class ReciprocalRankFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion combines multiple ranked lists using:
        score(doc) = sum( 1 / (rank_i + k) )

    where k is a constant (default 60) that reduces the impact of high-ranked items.
    This is a robust, parameter-free fusion method widely used in information retrieval.
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF constant. Higher values reduce the gap between ranks.
               Default 60 is the standard value from the original RRF paper.
        """
        self.k = k

    def fuse(self, results_list: List[List[Dict[str, Any]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Fuse multiple result lists using RRF.

        Args:
            results_list: List of result lists from different retrievers.
                          Each result must have 'text' key for deduplication.

        Returns:
            Fused list sorted by RRF score descending.
        """
        # Track RRF scores and best result per document (keyed by text)
        rrf_scores: Dict[str, float] = defaultdict(float)
        best_result: Dict[str, Dict[str, Any]] = {}
        per_method_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

        for result_list in results_list:
            for rank, result in enumerate(result_list):
                doc_key = result["text"]
                rrf_score = 1.0 / (rank + self.k)
                rrf_scores[doc_key] += rrf_score

                method = result.get("retrieval_method", "unknown")
                per_method_scores[doc_key][method] = result.get("score", 0.0)

                # Keep the result with the highest original score
                if doc_key not in best_result or result.get("score", 0) > best_result[doc_key].get("score", 0):
                    best_result[doc_key] = result

        # Build fused results
        fused = []
        for doc_key, rrf_score in rrf_scores.items():
            base = best_result[doc_key]
            fused.append({
                "text": base["text"],
                "metadata": base.get("metadata", {}),
                "score": rrf_score,
                "retrieval_method": "hybrid",
                "scores": per_method_scores[doc_key],
            })

        # Sort by fused score descending
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused
