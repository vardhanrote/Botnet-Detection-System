"""
Fresh LIME explanation for a CyberAgent investigation sample.
"""

from pathlib import Path

import joblib
import numpy as np

from lime.lime_tabular import LimeTabularExplainer


class LocalLIMEExplainer:

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

        train_network = np.load(
            project_root
            / "data"
            / "processed"
            / "final_network_train.npy"
        )

        train_dns = np.load(
            project_root
            / "data"
            / "processed"
            / "final_dns_train.npy"
        )

        X_train = np.concatenate(
            [
                train_network,
                train_dns
            ],
            axis=1
        )

        # Use a manageable reference subset.
        rng = np.random.default_rng(42)

        sample_size = min(
            10000,
            len(X_train)
        )

        indices = rng.choice(
            len(X_train),
            size=sample_size,
            replace=False
        )

        X_reference = X_train[
            indices
        ]

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

        self.explainer = LimeTabularExplainer(
            X_reference,
            feature_names=self.feature_names,
            class_names=self.attack_classes,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

    def explain(
        self,
        features,
        num_features=10,
        num_samples=2000,
    ):

        features = np.asarray(
            features,
            dtype=np.float32
        )

        probabilities = self.model.predict_proba(
            features.reshape(1, -1)
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        explanation = self.explainer.explain_instance(
            features,
            self.model.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=[predicted_class],
        )

        local_values = explanation.as_map()[
            predicted_class
        ]

        contributions = []

        for feature_index, contribution in local_values:

            contributions.append(
                {
                    "feature":
                        self.feature_names[
                            feature_index
                        ],

                    "feature_value":
                        float(
                            features[
                                feature_index
                            ]
                        ),

                    "contribution":
                        float(contribution),

                    "direction": (
                        "supports_prediction"
                        if contribution > 0
                        else "opposes_prediction"
                    ),
                }
            )

        return {
            "predicted_class":
                self.attack_classes[
                    predicted_class
                ],

            "confidence":
                float(
                    probabilities[
                        predicted_class
                    ]
                ),

            "top_features":
                contributions,
        }