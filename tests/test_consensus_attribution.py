
from csep.consensus_attribution import build_consensus


def test_matching_explanations_have_high_overlap():
    shap = {"bytes": 0.9, "duration": 0.7, "packets": 0.3}
    lime = {"bytes": 0.8, "duration": 0.6, "packets": 0.2}

    result = build_consensus(shap, lime, top_k=3)

    assert result["top_k_jaccard"] == 1.0
    assert result["kendall_tau"] > 0.99
    assert result["agreement_score"] > 0.99


def test_conflicting_explanations_reduce_agreement():
    shap = {"bytes": 0.9, "duration": 0.7, "packets": 0.2}
    lime = {"packets": 0.9, "duration": 0.6, "bytes": 0.1}

    result = build_consensus(shap, lime, top_k=1)

    assert result["top_k_jaccard"] == 0.0
    assert result["agreement_score"] < 0.75


def test_consensus_returns_ranked_evidence():
    result = build_consensus(
        {"bytes": 0.9, "duration": 0.2},
        {"bytes": 0.7, "packets": 0.3},
        top_k=2,
    )

    evidence = result["ranked_evidence"]

    assert evidence[0]["feature"] == "bytes"
    assert "shap_rank" in evidence[0]
    assert "lime_rank" in evidence[0]
