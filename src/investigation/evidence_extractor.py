"""
CyberAgent Investigation Evidence Extractor

This module combines outputs from multiple detection and explainability
components into a single investigation evidence package.

Components:
- Random Forest multiclass classifier
- Weighted Two-Stream neural network
- Isolation Forest anomaly detector
- Global SHAP
- Fresh local SHAP
- Fresh local LIME
- Model consistency analysis
- Evidence strength scoring
- Human-readable analyst summary
"""

from pathlib import Path
import json
import pickle
import joblib
import warnings

import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.models.two_stream_model import TwoStreamModel
from src.investigation.local_shap import LocalSHAPExplainer
from src.investigation.local_lime import LocalLIMEExplainer


warnings.filterwarnings("ignore")

# ============================================================
# FINAL TEST DATA PATHS
# ============================================================

FINAL_NETWORK_TEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_network_test.npy"
)

FINAL_DNS_TEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_dns_test.npy"
)


# ============================================================
# CLASS NAMES
# ============================================================

ATTACK_CLASSES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]


# ============================================================
# MAIN EVIDENCE EXTRACTOR
# ============================================================

class EvidenceExtractor:

    def __init__(self, project_root=PROJECT_ROOT):

        self.project_root = Path(project_root)

        # ----------------------------------------------------
        # Model paths
        # ----------------------------------------------------

        self.rf_model_path = (
            self.project_root
            / "models"
            / "baselines"
            / "random_forest_multiclass.pkl"
        )

        self.two_stream_model_path = (
            self.project_root
            / "models"
            / "two_stream_weighted_best.pth"
        )

        self.isolation_model_path = (
            self.project_root
            / "models"
            / "anomaly"
            / "isolation_forest_normal.pkl"
        )

        # ----------------------------------------------------
        # Explainability paths
        # ----------------------------------------------------

        self.global_shap_csv = (
            self.project_root
            / "results"
            / "explainability"
            / "shap_global_feature_importance.csv"
        )

        self.global_shap_values = (
            self.project_root
            / "results"
            / "explainability"
            / "shap_values.npy"
        )

        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        self.output_path = (
            self.project_root
            / "results"
            / "investigation"
            / "evidence_extraction_results.json"
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Load Random Forest
        # ----------------------------------------------------

        print("Loading Random Forest...")

        self.random_forest = joblib.load(
            self.rf_model_path
        )

        print("Random Forest loaded.")

        # ----------------------------------------------------
        # Load Two-Stream model
        # ----------------------------------------------------

        print("Loading Two-Stream model...")

        self.two_stream_model = TwoStreamModel(
            num_network_features=5,
            num_dns_features=5,
            num_classes=10,
            hidden_size=64,
            num_attention_heads=4,
            dropout=0.2,
        )

        checkpoint = torch.load(
            self.two_stream_model_path,
            map_location="cpu"
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            self.two_stream_model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            self.two_stream_model.load_state_dict(
                checkpoint
            )

        self.two_stream_model.eval()

        print("Two-Stream model loaded.")

        # ----------------------------------------------------
        # Load Isolation Forest
        # ----------------------------------------------------

        print("Loading Isolation Forest...")

        self.isolation_forest = joblib.load(
            self.isolation_model_path
        )


        print("Isolation Forest loaded.")

        # ----------------------------------------------------
        # Load global SHAP information
        # ----------------------------------------------------

        self.global_shap_importance = None  

        if self.global_shap_csv.exists():

            try:

                import pandas as pd

                self.global_shap_importance = pd.read_csv(
                    self.global_shap_csv
                )

            except Exception as e:

                print(
                    f"Warning: Could not load global SHAP CSV: {e}"
                )

        if self.global_shap_values.exists():

            try:

                self.global_shap_values = np.load(
                    self.global_shap_values
                )

            except Exception as e:

                print(
                    f"Warning: Could not load global SHAP values: {e}"
                )

        # ----------------------------------------------------
        # Fresh Local SHAP Explainer
        # ----------------------------------------------------

        print("Initializing Local SHAP explainer...")

        self.local_shap = LocalSHAPExplainer(
            project_root=self.project_root
        )

        print("Local SHAP explainer ready.")

        # ----------------------------------------------------
        # Fresh Local LIME Explainer
        # ----------------------------------------------------

        print("Initializing Local LIME explainer...")

        self.local_lime = LocalLIMEExplainer(
            project_root=self.project_root
        )

        print("Local LIME explainer ready.")

    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    def predict_random_forest(
        self,
        network_features,
        dns_features
    ):

        combined_features = np.concatenate(
            [
                network_features,
                dns_features
            ]
        ).reshape(1, -1)

        probabilities = (
            self.random_forest.predict_proba(
                combined_features
            )[0]
        )

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = ATTACK_CLASSES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        # ----------------------------------------------------
        # Top 3 predictions
        # ----------------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]

        top_predictions = []

        for index in top_indices:

            top_predictions.append(
                {
                    "class": ATTACK_CLASSES[int(index)],
                    "probability": float(
                        probabilities[index]
                    ),
                }
            )

        return {

            "predicted_class": predicted_class,

            "confidence": confidence,

            "top_predictions": top_predictions,

        }

    # ========================================================
    # TWO-STREAM PREDICTION
    # ========================================================

    def predict_two_stream(
        self,
        network_features,
        dns_features
    ):

        network_tensor = torch.tensor(
            network_features,
            dtype=torch.float32
        ).unsqueeze(0)

        dns_tensor = torch.tensor(
            dns_features,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            outputs = self.two_stream_model(
                network_tensor,
                dns_tensor
            )

            # ------------------------------------------------
            # IMPORTANT:
            # TwoStreamModel returns more than two outputs.
            # The first two are binary and multiclass outputs.
            # ------------------------------------------------

            binary_output = outputs[0]

            multiclass_output = outputs[1]

        # ----------------------------------------------------
        # Binary prediction
        # ----------------------------------------------------

        binary_probability = float(
            torch.sigmoid(
                binary_output
            ).item()
        )

        # Weighted model threshold
        binary_threshold = 0.34

        if binary_probability >= binary_threshold:

            binary_prediction = "Attack"

        else:

            binary_prediction = "Normal"

        # ----------------------------------------------------
        # Multiclass prediction
        # ----------------------------------------------------

        multiclass_probabilities = torch.softmax(
            multiclass_output,
            dim=1
        )[0].cpu().numpy()

        multiclass_index = int(
            np.argmax(
                multiclass_probabilities
            )
        )

        multiclass_prediction = (
            ATTACK_CLASSES[
                multiclass_index
            ]
        )

        multiclass_confidence = float(
            multiclass_probabilities[
                multiclass_index
            ]
        )

        # ----------------------------------------------------
        # Top 3 multiclass predictions
        # ----------------------------------------------------

        top_indices = np.argsort(
            multiclass_probabilities
        )[::-1][:3]

        top_predictions = []

        for index in top_indices:

            top_predictions.append(
                {
                    "class": ATTACK_CLASSES[
                        int(index)
                    ],
                    "probability": float(
                        multiclass_probabilities[
                            index
                        ]
                    ),
                }
            )

        return {

            "binary_prediction": binary_prediction,

            "binary_probability": binary_probability,

            "binary_threshold": binary_threshold,

            "multiclass_prediction": (
                multiclass_prediction
            ),

            "multiclass_confidence": (
                multiclass_confidence
            ),

            "top_predictions": top_predictions,

        }

    # ========================================================
    # ISOLATION FOREST ANOMALY DETECTION
    # ========================================================

    def predict_anomaly(
        self,
        network_features,
        dns_features
    ):

        combined_features = np.concatenate(
            [
                network_features,
                dns_features
            ]
        ).reshape(1, -1)

        # Isolation Forest prediction
        prediction = self.isolation_forest.predict(
            combined_features
        )[0]

        # Decision function
        anomaly_score = float(
            self.isolation_forest.decision_function(
                combined_features
            )[0]
        )

        is_anomaly = (
            prediction == -1
        )

        return {

            "is_anomaly": bool(
                is_anomaly
            ),

            "anomaly_score": anomaly_score,

            "interpretation": (
                "Deviation from learned normal "
                "traffic distribution"
                if is_anomaly
                else
                "Within learned normal "
                "traffic distribution"
            ),

        }

    # ========================================================
    # GLOBAL SHAP EVIDENCE
    # ========================================================

    def get_global_shap_evidence(self):

        if self.global_shap_importance is None:

            return {

                "available": False,

                "message": (
                    "Global SHAP importance data "
                    "is not available."
                ),

            }

        top_features = []

        try:

            dataframe = (
                self.global_shap_importance
                .sort_values(
                    by="mean_abs_shap",
                    ascending=False
                )
                .head(10)
            )

            for _, row in dataframe.iterrows():

                top_features.append(
                    {
                        "feature": str(
                            row["feature"]
                        ),
                        "mean_abs_shap": float(
                            row["mean_abs_shap"]
                        ),
                    }
                )

        except Exception as e:

            return {

                "available": False,

                "message": (
                    f"Could not process global SHAP: {e}"
                ),

            }

        return {

            "available": True,

            "top_features": top_features,

            "interpretation": (
                "Global SHAP shows which features "
                "most influenced the Random Forest "
                "multiclass predictions across the "
                "evaluation samples."
            ),

        }

    # ========================================================
    # FRESH LOCAL SHAP
    # ========================================================

    def get_local_shap(
        self,
        combined_features
    ):

        try:

            result = self.local_shap.explain(
                combined_features
            )

            return result

        except Exception as e:

            return {

                "available": False,

                "error": str(e),

            }

    # ========================================================
    # MODEL CONSISTENCY
    # ========================================================

    def calculate_model_consistency(
        self,
        rf_result,
        two_stream_result
    ):

        rf_class = (
            rf_result["predicted_class"]
        )

        two_stream_class = (
            two_stream_result[
                "multiclass_prediction"
            ]
        )

        model_disagreement = (
            rf_class != two_stream_class
        )

        if model_disagreement:

            agreement_status = "Disagreement"

        else:

            agreement_status = "Agreement"

        return {

            "random_forest_class": rf_class,

            "two_stream_class": (
                two_stream_class
            ),

            "model_disagreement": bool(
                model_disagreement
            ),

            "agreement_status": (
                agreement_status
            ),

        }

    # ========================================================
    # EVIDENCE STRENGTH
    # ========================================================

    def calculate_evidence_strength(
        self,
        rf_result,
        two_stream_result,
        anomaly_result,
        consistency_result
    ):

        score = 0

        # ----------------------------------------------------
        # Random Forest confidence
        # ----------------------------------------------------

        rf_confidence = (
            rf_result["confidence"]
        )

        if rf_confidence >= 0.80:

            score += 25

        elif rf_confidence >= 0.60:

            score += 20

        elif rf_confidence >= 0.40:

            score += 15

        else:

            score += 5

        # ----------------------------------------------------
        # Two-Stream confidence
        # ----------------------------------------------------

        two_stream_confidence = (
            two_stream_result[
                "multiclass_confidence"
            ]
        )

        if two_stream_confidence >= 0.80:

            score += 25

        elif two_stream_confidence >= 0.60:

            score += 20

        elif two_stream_confidence >= 0.40:

            score += 15

        else:

            score += 5

        # ----------------------------------------------------
        # Anomaly signal
        # ----------------------------------------------------

        if anomaly_result["is_anomaly"]:

            score += 25

        else:

            score += 10

        # ----------------------------------------------------
        # Model agreement
        # ----------------------------------------------------

        if not consistency_result[
            "model_disagreement"
        ]:

            score += 25

        else:

            score += 10

        # ----------------------------------------------------
        # Limit to 100
        # ----------------------------------------------------

        score = min(
            score,
            100
        )

        # ----------------------------------------------------
        # Evidence level
        # ----------------------------------------------------

        if score >= 80:

            level = "High"

        elif score >= 60:

            level = "Medium"

        else:

            level = "Low"

        return {

            "score": int(score),

            "level": level,

        }

    # ========================================================
    # HUMAN-READABLE ANALYST SUMMARY
    # ========================================================

    def generate_summary(
        self,
        evidence
    ):

        binary = evidence[
            "binary_detection"
        ]

        classification = evidence[
            "attack_classification"
        ]

        rf = classification[
            "random_forest"
        ]

        two_stream = classification[
            "two_stream"
        ]

        anomaly = evidence[
            "anomaly_detection"
        ]

        consistency = evidence[
            "model_consistency"
        ]

        strength = evidence[
            "evidence_strength"
        ]

        summary = []

        # ----------------------------------------------------
        # Binary detection
        # ----------------------------------------------------

        if binary["prediction"] == "Attack":

            summary.append(
                "The binary detector identified "
                "the traffic as an attack."
            )

        else:

            summary.append(
                "The binary detector classified "
                "the traffic as normal."
            )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        summary.append(
            f"Random Forest predicted "
            f"{rf['predicted_class']} "
            f"with {rf['confidence']:.2%} confidence."
        )

        # ----------------------------------------------------
        # Two-Stream
        # ----------------------------------------------------

        summary.append(
            f"Two-Stream predicted "
            f"{two_stream['multiclass_prediction']} "
            f"with "
            f"{two_stream['multiclass_confidence']:.2%} "
            f"confidence."
        )

        # ----------------------------------------------------
        # Isolation Forest
        # ----------------------------------------------------

        if anomaly["is_anomaly"]:

            summary.append(
                "Isolation Forest detected a deviation "
                "from the learned normal traffic "
                "distribution."
            )

        else:

            summary.append(
                "Isolation Forest did not flag the "
                "sample as anomalous."
            )

        # ----------------------------------------------------
        # Model consistency
        # ----------------------------------------------------

        if consistency[
            "model_disagreement"
        ]:

            summary.append(
                "The supervised classifiers disagree "
                "on the specific attack class."
            )

        else:

            summary.append(
                "The supervised classifiers agree "
                "on the predicted attack class."
            )

        # ----------------------------------------------------
        # Evidence strength
        # ----------------------------------------------------

        summary.append(
            f"Overall evidence strength is "
            f"{strength['level']} "
            f"({strength['score']}/100)."
        )

        return summary

    # ========================================================
    # MAIN EVIDENCE EXTRACTION FUNCTION
    # ========================================================

    def extract(
        self,
        network_features,
        dns_features,
        sample_index=None
    ):

        network_features = np.asarray(
            network_features,
            dtype=np.float32
        )

        dns_features = np.asarray(
            dns_features,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Combine features
        # ----------------------------------------------------

        combined_features = np.concatenate(
            [
                network_features,
                dns_features
            ]
        )

        # ----------------------------------------------------
        # 1. Random Forest
        # ----------------------------------------------------

        rf_result = self.predict_random_forest(
            network_features,
            dns_features
        )

        # ----------------------------------------------------
        # 2. Two-Stream model
        # ----------------------------------------------------

        two_stream_result = (
            self.predict_two_stream(
                network_features,
                dns_features
            )
        )

        # ----------------------------------------------------
        # 3. Isolation Forest
        # ----------------------------------------------------

        anomaly_result = (
            self.predict_anomaly(
                network_features,
                dns_features
            )
        )

        # ----------------------------------------------------
        # 4. Fresh Local SHAP
        # ----------------------------------------------------

        shap_local = self.get_local_shap(
            combined_features
        )

        # ----------------------------------------------------
        # 5. Global SHAP
        # ----------------------------------------------------

        shap_global = (
            self.get_global_shap_evidence()
        )

        # ----------------------------------------------------
        # 6. Fresh Local LIME
        # ----------------------------------------------------

        lime_result = (
            self.local_lime.explain(
                combined_features,
                num_features=10,
                num_samples=2000,
            )
        )

        # ----------------------------------------------------
        # 7. Model consistency
        # ----------------------------------------------------

        consistency_result = (
            self.calculate_model_consistency(
                rf_result,
                two_stream_result
            )
        )

        # ----------------------------------------------------
        # 8. Evidence strength
        # ----------------------------------------------------

        evidence_strength = (
            self.calculate_evidence_strength(
                rf_result,
                two_stream_result,
                anomaly_result,
                consistency_result
            )
        )

        # ----------------------------------------------------
        # 9. Binary detection
        # ----------------------------------------------------

        binary_prediction = (
            two_stream_result[
                "binary_prediction"
            ]
        )

        binary_probability = (
            two_stream_result[
                "binary_probability"
            ]
        )

        binary_detection = {

            "prediction": binary_prediction,

            "attack_probability": (
                binary_probability
            ),

            "normal_probability": (
                1.0 - binary_probability
            ),

            "threshold": (
                two_stream_result[
                    "binary_threshold"
                ]
            ),

        }

        # ----------------------------------------------------
        # 10. Build evidence object
        # ----------------------------------------------------

        evidence = {

            "sample": {

                "sample_index": (
                    int(sample_index)
                    if sample_index is not None
                    else None
                ),

                "network_features": (
                    network_features.tolist()
                ),

                "dns_features": (
                    dns_features.tolist()
                ),

            },

            # ------------------------------------------------
            # Binary Detection
            # ------------------------------------------------

            "binary_detection": (
                binary_detection
            ),

            # ------------------------------------------------
            # Attack Classification
            # ------------------------------------------------

            "attack_classification": {

                "random_forest": (
                    rf_result
                ),

                "two_stream": (
                    two_stream_result
                ),

            },

            # ------------------------------------------------
            # Anomaly Detection
            # ------------------------------------------------

            "anomaly_detection": (
                anomaly_result
            ),

            # ------------------------------------------------
            # Explainability
            # ------------------------------------------------

            "explainability": {

                "local_shap": (
                    shap_local
                ),

                "global_shap": (
                    shap_global
                ),

                "lime": (
                    lime_result
                ),

            },

            # ------------------------------------------------
            # Model Consistency
            # ------------------------------------------------

            "model_consistency": (
                consistency_result
            ),

            # ------------------------------------------------
            # Evidence Strength
            # ------------------------------------------------

            "evidence_strength": (
                evidence_strength
            ),

        }

        # ----------------------------------------------------
        # 11. Human-readable analyst summary
        # ----------------------------------------------------

        evidence["analyst_summary"] = (
            self.generate_summary(
                evidence
            )
        )

        return evidence

    # ========================================================
    # SAVE EVIDENCE
    # ========================================================

    def save_evidence(
        self,
        evidence,
        output_path=None
    ):

        if output_path is None:

            output_path = (
                self.output_path
            )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                evidence,
                f,
                indent=4
            )

        print(
            f"\nEvidence saved to:\n{output_path}"
        )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("CYBERAGENT — INVESTIGATION EVIDENCE EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test features...")

    network_test = np.load(
        FINAL_NETWORK_TEST
    )

    dns_test = np.load(
        FINAL_DNS_TEST
    )

    print(
        f"Network test shape: {network_test.shape}"
    )

    print(
        f"DNS test shape: {dns_test.shape}"
    )

    # --------------------------------------------------------
    # Select sample
    # --------------------------------------------------------

    sample_index = 0

    network_sample = (
        network_test[sample_index]
    )

    dns_sample = (
        dns_test[sample_index]
    )

    print(
        f"\nInvestigating test sample "
        f"{sample_index}..."
    )

    # --------------------------------------------------------
    # Create extractor
    # --------------------------------------------------------

    extractor = EvidenceExtractor()

    # --------------------------------------------------------
    # Extract evidence
    # --------------------------------------------------------

    evidence = extractor.extract(
        network_features=network_sample,
        dns_features=dns_sample,
        sample_index=sample_index
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BINARY DETECTION")
    print("=" * 70)

    print(
        "Prediction:",
        evidence[
            "binary_detection"
        ]["prediction"]
    )

    print(
        "Attack probability:",
        f"{evidence['binary_detection']['attack_probability']:.4f}"
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST")
    print("=" * 70)

    rf = evidence[
        "attack_classification"
    ]["random_forest"]

    print(
        "Prediction:",
        rf["predicted_class"]
    )

    print(
        "Confidence:",
        f"{rf['confidence']:.4f}"
    )

    print("\nTop predictions:")

    for item in rf[
        "top_predictions"
    ]:

        print(
            f"  {item['class']}: "
            f"{item['probability']:.4f}"
        )

    print("\n" + "=" * 70)
    print("TWO-STREAM MODEL")
    print("=" * 70)

    two_stream = evidence[
        "attack_classification"
    ]["two_stream"]

    print(
        "Binary:",
        two_stream[
            "binary_prediction"
        ]
    )

    print(
        "Binary probability:",
        f"{two_stream['binary_probability']:.4f}"
    )

    print(
        "Multiclass:",
        two_stream[
            "multiclass_prediction"
        ]
    )

    print(
        "Multiclass confidence:",
        f"{two_stream['multiclass_confidence']:.4f}"
    )

    print("\n" + "=" * 70)
    print("ISOLATION FOREST")
    print("=" * 70)

    anomaly = evidence[
        "anomaly_detection"
    ]

    print(
        "Anomaly:",
        anomaly["is_anomaly"]
    )

    print(
        "Anomaly score:",
        f"{anomaly['anomaly_score']:.6f}"
    )

    print(
        "Interpretation:",
        anomaly["interpretation"]
    )

    print("\n" + "=" * 70)
    print("MODEL CONSISTENCY")
    print("=" * 70)

    consistency = evidence[
        "model_consistency"
    ]

    print(
        "Random Forest:",
        consistency[
            "random_forest_class"
        ]
    )

    print(
        "Two-Stream:",
        consistency[
            "two_stream_class"
        ]
    )

    print(
        "Status:",
        consistency[
            "agreement_status"
        ]
    )

    print("\n" + "=" * 70)
    print("EVIDENCE STRENGTH")
    print("=" * 70)

    strength = evidence[
        "evidence_strength"
    ]

    print(
        "Level:",
        strength["level"]
    )

    print(
        "Score:",
        f"{strength['score']}/100"
    )

    print("\n" + "=" * 70)
    print("ANALYST SUMMARY")
    print("=" * 70)

    for item in evidence[
        "analyst_summary"
    ]:

        print(
            f"- {item}"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    extractor.save_evidence(
        evidence
    )

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)