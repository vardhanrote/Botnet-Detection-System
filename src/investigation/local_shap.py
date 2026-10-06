"""
Local SHAP explanation for CyberAgent investigations.

Generates a fresh explanation for the requested sample.
"""

from pathlib import Path

import joblib
import numpy as np
import shap


class LocalSHAPExplainer:

    def __init__(self, project_root=None):

        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        else:
            project_root = Path(project_root)

        self.project_root = project_root

        model_path = (
            project_root
            / "models"
            / "baselines"
            / "random_forest_multiclass.pkl"
        )

        self.model = joblib.load(model_path)

        self.explainer = shap.TreeExplainer(
            self.model
        )

        self.feature_names = [
            "inter_arrival_time",
            "protocol_distribution",
            "packet_rate",
            "flow_duration",
            "packet_size",
            "query_type_distribution",
            "dns_query_frequency",
            "query_rate",
            "dns_response_activity",
            "query_length",
        ]

        self.attack_classes = [
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

    def explain(self, features):

        features = np.asarray(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        prediction = int(
            self.model.predict(features)[0]
        )

        probabilities = (
            self.model.predict_proba(
                features
            )[0]
        )

        shap_values = self.explainer.shap_values(
            features
        )

        # Different SHAP versions can return:
        # [samples, features, classes]
        # or a list of arrays.

        if isinstance(shap_values, list):

            class_values = np.asarray(
                shap_values[prediction]
            )[0]

        else:

            shap_values = np.asarray(
                shap_values
            )

            if shap_values.ndim == 3:

                class_values = shap_values[
                    0,
                    :,
                    prediction
                ]

            else:

                class_values = shap_values[
                    0
                ]

        ranked_indices = np.argsort(
            np.abs(class_values)
        )[::-1]

        contributions = []

        for index in ranked_indices:

            contribution = float(
                class_values[index]
            )

            contributions.append(
                {
                    "feature": self.feature_names[
                        index
                    ],

                    "feature_value": float(
                        features[0, index]
                    ),

                    "contribution": contribution,

                    "direction": (
                        "supports_prediction"
                        if contribution > 0
                        else "opposes_prediction"
                    ),
                }
            )

        return {

            "predicted_class":
                self.attack_classes[prediction],

            "confidence":
                float(probabilities[prediction]),

            "top_features":
                contributions[:10],
        }