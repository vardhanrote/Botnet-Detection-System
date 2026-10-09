
"""
CA: Consensus Attribution.

This module consumes already-computed SHAP and LIME explanations.
It does not fit the classifier or the explainers.

Provide mappings of feature name -> attribution value for the SAME
observation and the SAME predicted class. Absolute attribution magnitude
is used for ranking; signed values can be retained separately for display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import kendalltau


def _rank_attributions(attributions: dict[str, float]) -> dict[str, int]:
    """Rank features by absolute attribution magnitude, rank 1 = highest."""
    cleaned = {
        str(name): abs(float(value))
        for name, value in attributions.items()
        if np.isfinite(float(value))
    }

    ordered = sorted(
        cleaned,
        key=lambda name: (-cleaned[name], name),
    )
    return {name: rank + 1 for rank, name in enumerate(ordered)}


def build_consensus(
    shap_values: dict[str, float],
    lime_values: dict[str, float],
    top_k: int = 10,
) -> dict:
    """
    Combine SHAP and LIME evidence for one observation.

    Returns:
        ranked_evidence: combined feature ranking and original values
        top_k_jaccard: overlap between the two top-k feature sets
        kendall_tau: rank correlation over features shared by both methods
        agreement_score: bounded 0..1 agreement summary
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not shap_values or not lime_values:
        raise ValueError("Both SHAP and LIME explanations are required.")

    shap_rank = _rank_attributions(shap_values)
    lime_rank = _rank_attributions(lime_values)

    if not shap_rank or not lime_rank:
        raise ValueError("Attributions must contain finite numeric values.")

    union_features = set(shap_rank) | set(lime_rank)
    shared_features = set(shap_rank) & set(lime_rank)

    # Missing features receive a rank just below the bottom of that
    # explainer's list, so a feature explained by only one method is
    # penalized rather than silently treated as agreement.
    shap_missing_rank = len(shap_rank) + 1
    lime_missing_rank = len(lime_rank) + 1

    rows = []
    for feature in union_features:
        sr = shap_rank.get(feature, shap_missing_rank)
        lr = lime_rank.get(feature, lime_missing_rank)

        # Borda-style score: lower combined rank is better.
        combined_rank_score = sr + lr

        rows.append(
            {
                "feature": feature,
                "shap_value": shap_values.get(feature),
                "lime_value": lime_values.get(feature),
                "shap_rank": sr,
                "lime_rank": lr,
                "combined_rank_score": combined_rank_score,
                "rank_difference": abs(sr - lr),
            }
        )

    evidence = pd.DataFrame(rows).sort_values(
        by=["combined_rank_score", "feature"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    shap_top = set(
        sorted(shap_rank, key=shap_rank.get)[:top_k]
    )
    lime_top = set(
        sorted(lime_rank, key=lime_rank.get)[:top_k]
    )

    top_union = shap_top | lime_top
    top_intersection = shap_top & lime_top

    top_k_jaccard = (
        len(top_intersection) / len(top_union)
        if top_union else 1.0
    )

    if len(shared_features) >= 2:
        ordered_shared = sorted(shared_features)
        shap_orders = [shap_rank[f] for f in ordered_shared]
        lime_orders = [lime_rank[f] for f in ordered_shared]
        tau_result = kendalltau(shap_orders, lime_orders)
        tau = float(tau_result.statistic)

        if not np.isfinite(tau):
            tau = 0.0
    else:
        # Insufficient common features means rank correlation is unknown,
        # not evidence of perfect agreement.
        tau = 0.0

    # Map Kendall tau from [-1, 1] to [0, 1], then combine with overlap.
    normalized_tau = (tau + 1.0) / 2.0
    agreement_score = float(
        np.clip(0.5 * top_k_jaccard + 0.5 * normalized_tau, 0, 1)
    )

    return {
        "ranked_evidence": evidence.to_dict(orient="records"),
        "top_k": top_k,
        "top_k_jaccard": float(top_k_jaccard),
        "kendall_tau": float(tau),
        "agreement_score": agreement_score,
        "shared_feature_count": len(shared_features),
    }
