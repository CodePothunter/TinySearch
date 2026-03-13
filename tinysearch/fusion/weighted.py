"""
Weighted score fusion strategy
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tinysearch.base import FusionStrategy


class WeightedFusion(FusionStrategy):
    """
    Weighted fusion normalizes each retriever's scores to [0, 1] via min-max,
    then computes a weighted sum:
        fusion_score = sum( normalized_score_i * weight_i )
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        min_score: float = 0.0,
    ):
        """
        Args:
            weights: Weight for each retriever. If None, equal weights are used.
            min_score: Minimum fusion score threshold. Results below this are dropped.
        """
        self.weights = weights
        self.min_score = min_score

    def fuse(self, results_list: List[List[Dict[str, Any]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Fuse multiple result lists using weighted score fusion.

        Args:
            results_list: List of result lists from different retrievers.
            **kwargs:
                weights: Override weights (takes precedence over self.weights)

        Returns:
            Fused list sorted by fusion score descending.
        """
        if not results_list:
            return []

        n_retrievers = len(results_list)
        weights = kwargs.get("weights", self.weights)
        if weights is None:
            weights = [1.0 / n_retrievers] * n_retrievers

        if len(weights) != n_retrievers:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of result lists ({n_retrievers})"
            )

        # Normalize scores per retriever (min-max to [0, 1])
        normalized_lists = []
        for result_list in results_list:
            if not result_list:
                normalized_lists.append([])
                continue
            scores = [r.get("score", 0.0) for r in result_list]
            min_s = min(scores)
            max_s = max(scores)
            range_s = max_s - min_s if max_s != min_s else 1.0

            normalized = []
            for r in result_list:
                norm_score = (r.get("score", 0.0) - min_s) / range_s
                normalized.append({**r, "_norm_score": norm_score})
            normalized_lists.append(normalized)

        # Aggregate by document text
        doc_scores: Dict[str, float] = defaultdict(float)
        best_result: Dict[str, Dict[str, Any]] = {}
        per_method_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

        for i, (result_list, weight) in enumerate(zip(normalized_lists, weights)):
            for result in result_list:
                doc_key = result["text"]
                doc_scores[doc_key] += result["_norm_score"] * weight

                method = result.get("retrieval_method", "unknown")
                per_method_scores[doc_key][method] = result.get("score", 0.0)

                if doc_key not in best_result:
                    best_result[doc_key] = result

        # Build fused results
        fused = []
        for doc_key, fusion_score in doc_scores.items():
            if fusion_score < self.min_score:
                continue
            base = best_result[doc_key]
            fused.append({
                "text": base["text"],
                "metadata": base.get("metadata", {}),
                "score": fusion_score,
                "retrieval_method": "hybrid",
                "scores": per_method_scores[doc_key],
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused
