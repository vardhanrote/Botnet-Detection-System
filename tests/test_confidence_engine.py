from src.investigation.confidence_engine import (
    InvestigationConfidenceEngine,
)


def test_confidence_engine():
    engine = InvestigationConfidenceEngine()

    binary_detection = {
        "prediction": "Attack",
        "attack_probability": 0.95,
    }

    classification = {
        "random_forest": {
            "predicted_class": "Exploits",
            "confidence": 0.91,
        },
        "two_stream": {
            "multiclass_prediction": "Exploits",
            "multiclass_confidence": 0.88,
        },
    }

    anomaly_detection = {
        "is_anomaly": True,
    }

    model_consistency = {
        "model_disagreement": False,
    }

    evidence_strength = {
        "score": 90,
        "level": "High",
    }

    threat_intelligence = {
        "candidate_mappings": [
            {
                "technique_id": "T1190",
                "technique_name": (
                    "Exploit Public-Facing Application"
                ),
            },
            {
                "technique_id": "T1210",
                "technique_name": (
                    "Exploitation of Remote Services"
                ),
            },
        ]
    }

    result = engine.calculate(
        binary_detection=binary_detection,
        classification=classification,
        anomaly_detection=anomaly_detection,
        model_consistency=model_consistency,
        evidence_strength=evidence_strength,
        threat_intelligence=threat_intelligence,
        threat_class="Exploits",
    )

    print("\nConfidence Engine Test")
    print("----------------------")
    print("Score:", result["score"])
    print("Level:", result["level"])
    print("Interpretation:")
    print(result["interpretation"])

    print("\nComponents:")
    for key, value in result["components"].items():
        print(f"{key}: {value}")

    assert 0 <= result["score"] <= 100
    assert result["level"] in [
        "Very Low",
        "Low",
        "Moderate",
        "High",
    ]
    assert "not an attack probability" in result["note"].lower()