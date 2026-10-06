# CyberAgent: Agent-Assisted Network Threat Detection and Investigation using Machine Learning

## Overview

**CyberAgent** is an intelligent network security system designed to detect, classify, investigate, and explain suspicious network traffic using machine learning, deep learning, anomaly detection, and explainable AI. The project addresses a major limitation of traditional intrusion detection systems: detecting malicious traffic is only the first step, while security analysts also need to understand why traffic was flagged, what type of activity it may represent, how confident the models are, whether different detection methods agree, and what evidence supports the alert. CyberAgent therefore combines supervised machine learning, a custom two-stream deep learning architecture, unsupervised anomaly detection, SHAP/LIME explainability, and an evidence extraction engine into a unified investigation pipeline. The current implementation uses the **UNSW-NB15** network intrusion dataset and focuses on both binary attack detection and multiclass attack classification. The system is designed as a research-oriented foundation for a future SOC-style cybersecurity platform with threat intelligence, MITRE ATT&CK mapping, an AI investigation agent, security regression testing, attack replay, and automated incident reporting.

## Problem Statement

Traditional network intrusion detection systems generally focus on answering a simple question: **"Is this traffic malicious?"** However, real-world cybersecurity investigation requires more information. A security analyst needs to know whether the traffic is anomalous, what type of attack it resembles, which features influenced the prediction, whether multiple models agree, and how strong the available evidence is. CyberAgent addresses this problem by combining multiple detection and investigation techniques into a single pipeline. Instead of relying on one model, the system creates an evidence-based view of suspicious traffic by combining binary classification, multiclass classification, anomaly detection, explainability, model consistency analysis, and evidence scoring.

## Main Objectives

The main objectives of CyberAgent are to: detect network traffic as normal or malicious; classify malicious traffic into multiple attack categories; compare classical machine-learning and deep-learning approaches; use network and DNS-derived flow features through separate feature streams; improve representation learning using CNN layers, feature fusion, multi-head attention, and BiLSTM; identify unusual traffic using unsupervised anomaly detection; explain model predictions using SHAP and LIME; generate local explanations for individual suspicious samples; compare predictions from different models; calculate an internal evidence-strength score; generate an analyst-friendly investigation summary; store investigation evidence in a structured JSON format; and provide a foundation for future threat profiling, threat intelligence, MITRE ATT&CK mapping, AI-assisted investigation, SOC visualization, and incident reporting.

## Key Features

CyberAgent currently includes a complete machine-learning detection pipeline, feature engineering, Ridge-based feature selection, class balancing using SMOTE, a custom Two-Stream CNN architecture, Multi-Head Attention, BiLSTM-based representation learning, binary attack detection, multiclass attack classification, Random Forest and XGBoost baselines, a CNN-LSTM baseline, Isolation Forest anomaly detection, SHAP global explainability, LIME local explainability, fresh local SHAP explanations, fresh local LIME explanations, model consistency analysis, evidence-strength scoring, analyst-oriented evidence summaries, and persistent investigation results in JSON format.

## System Architecture

```text
                         ┌──────────────────────────┐
                         │       UNSW-NB15          │
                         │  Training + Testing Data │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Data Preprocessing     │
                         │                          │
                         │ • Cleaning               │
                         │ • Encoding              │
                         │ • Feature preparation   │
                         │ • Train/Validation split│
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  Feature Engineering     │
                         │                          │
                         │ • Network features      │
                         │ • DNS-derived proxies   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Ridge Feature Selection  │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
           ┌──────────────────┐               ┌──────────────────┐
           │  Network Stream  │               │    DNS Stream    │
           │    5 Features    │               │    5 Features    │
           └────────┬─────────┘               └────────┬─────────┘
                    │                                  │
                    ▼                                  ▼
           ┌──────────────────┐               ┌──────────────────┐
           │ Conv1D + BN      │               │ Conv1D + BN      │
           │ ReLU + Pooling   │               │ ReLU + Pooling   │
           └────────┬─────────┘               └────────┬─────────┘
                    │                                  │
                    └────────────────┬─────────────────┘
                                     ▼
                            ┌──────────────────┐
                            │ Feature Fusion   │
                            └────────┬─────────┘
                                     ▼
                            ┌──────────────────┐
                            │ Multi-Head       │
                            │ Attention        │
                            └────────┬─────────┘
                                     ▼
                            ┌──────────────────┐
                            │ BiLSTM           │
                            └────────┬─────────┘
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
                 ┌───────────────┐       ┌────────────────┐
                 │ Binary Output │       │ Multiclass     │
                 │ Normal/Attack │       │ 10 Classes     │
                 └───────┬───────┘       └───────┬────────┘
                         │                       │
                         └───────────┬───────────┘
                                     ▼
                       ┌──────────────────────────┐
                       │ Additional ML Analysis   │
                       │                          │
                       │ • Random Forest          │
                       │ • XGBoost                │
                       │ • CNN-LSTM               │
                       │ • Isolation Forest       │
                       └────────────┬─────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │ Explainability Layer     │
                       │                          │
                       │ • SHAP                   │
                       │ • LIME                   │
                       │ • Local SHAP             │
                       │ • Local LIME             │
                       └────────────┬─────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │ Evidence Extraction      │
                       │ Engine                   │
                       │                          │
                       │ • Binary prediction     │
                       │ • Attack classification │
                       │ • Confidence             │
                       │ • Anomaly detection     │
                       │ • Model agreement       │
                       │ • Evidence strength     │
                       │ • Analyst summary       │
                       └────────────┬─────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │ Investigation Evidence  │
                       │          JSON            │
                       └────────────┬─────────────┘
                                    ▼
                 ┌────────────────────────────────────┐
                 │        Future CyberAgent           │
                 │                                    │
                 │ • Threat Profiling                  │
                 │ • Severity Engine                   │
                 │ • Threat Intelligence               │
                 │ • MITRE ATT&CK Mapping              │
                 │ • AI Investigation Agent            │
                 │ • Attack Replay                     │
                 │ • Security Regression Testing       │
                 │ • SOC Application                   │
                 │ • Incident Reporting                │
                 └────────────────────────────────────┘
```

## End-to-End Workflow

The system begins with the UNSW-NB15 dataset containing network-flow information and attack labels. The raw data is inspected and prepared for machine-learning processing. Relevant categorical and numerical information is transformed into model-compatible representations. The dataset is divided into training, validation, and testing portions while keeping the test set untouched. Network and DNS-derived flow features are then engineered from the available flow-level information. Ridge-based feature selection is applied using the training data to identify the most useful features for each stream. Class imbalance in the training data is addressed using SMOTE. The resulting network and DNS feature groups are passed to the Two-Stream deep-learning architecture. Each stream independently learns feature representations through Conv1D layers before the learned representations are fused. Multi-Head Attention learns relationships between the fused representations, followed by BiLSTM processing and separate binary and multiclass prediction heads. Classical machine-learning models are also trained as baselines. Isolation Forest is trained only on normal training traffic to provide a second, unsupervised anomaly signal. SHAP and LIME are then used to understand model decisions. During investigation, the system combines predictions from the trained models, anomaly information, explainability information, model agreement, and evidence scoring into a structured investigation result. The final evidence is stored as JSON for further analysis or future integration with a SOC interface and AI investigation agent.

## Dataset

CyberAgent currently uses the **UNSW-NB15** network intrusion dataset. The dataset contains network-flow records representing normal traffic and multiple categories of malicious activity. The project uses the provided training and testing datasets.

Training dataset shape:

```text
175,341 rows × 45 columns
```

Testing dataset shape:

```text
82,332 rows × 45 columns
```

The binary classification task contains:

| Label  | Samples |
| ------ | ------: |
| Normal |  56,000 |
| Attack | 119,341 |

The multiclass classification task contains the following classes:

| Attack Class   | Samples |
| -------------- | ------: |
| Normal         |  56,000 |
| Generic        |  40,000 |
| Exploits       |  33,393 |
| Fuzzers        |  18,184 |
| DoS            |  12,264 |
| Reconnaissance |  10,491 |
| Analysis       |   2,000 |
| Backdoor       |   1,746 |
| Shellcode      |   1,133 |
| Worms          |     130 |

The severe imbalance among minority classes is one of the challenges addressed by the project.

## Feature Engineering

CyberAgent separates the selected features into two streams.

### Network Features

The current network stream contains:

```text
inter_arrival_time
protocol_distribution
packet_rate
flow_duration
packet_size
```

These features represent characteristics such as traffic timing, protocol behavior, packet frequency, duration, and packet-size information.

### DNS-Derived Flow Features

The current DNS stream contains:

```text
query_type_distribution
dns_query_frequency
query_rate
dns_response_activity
query_length
```

These are **DNS-derived flow-level proxies** calculated from available network-flow information.

The current implementation does not claim to process raw DNS packets or complete DNS protocol-level telemetry.

## Ridge Feature Selection

Ridge-based feature selection is applied before the deep-learning stage. The purpose is to reduce unnecessary features and retain a compact set of features with useful predictive information. Feature selection is fitted using the training data rather than the untouched test set.

The final selected network features are:

```text
1. inter_arrival_time
2. protocol_distribution
3. packet_rate
4. flow_duration
5. packet_size
```

The final selected DNS-derived features are:

```text
1. query_type_distribution
2. dns_query_frequency
3. query_rate
4. dns_response_activity
5. query_length
```

## Class Balancing

The training data contains significant class imbalance, particularly for classes such as Worms, Shellcode, and Backdoor. SMOTE is applied to the training data to reduce this imbalance.

The validation and test datasets remain untouched by SMOTE so that evaluation represents the original data distribution.

After SMOTE, each training class contains approximately:

```text
50,400 samples
```

resulting in approximately:

```text
504,000 training samples
```

The validation set contains:

```text
17,535 samples
```

and the test set contains:

```text
82,332 samples
```

## Two-Stream Deep Learning Architecture

The main deep-learning model is a custom Two-Stream architecture.

The architecture contains two independent feature branches:

```text
Network Features
       ↓
Conv1D
       ↓
Batch Normalization
       ↓
ReLU
       ↓
Max Pooling
       ↓
Conv1D
       ↓
Batch Normalization
       ↓
ReLU
```

and:

```text
DNS Features
       ↓
Conv1D
       ↓
Batch Normalization
       ↓
ReLU
       ↓
Max Pooling
       ↓
Conv1D
       ↓
Batch Normalization
       ↓
ReLU
```

The two learned representations are then fused.

The fused representation is processed by:

```text
Feature Fusion
      ↓
Multi-Head Attention
      ↓
Residual Connection
      ↓
BiLSTM
      ↓
Mean Pooling
      ↓
Fully Connected Layer
      ↓
Binary Head + Multiclass Head
```

The attention layer uses four attention heads and the BiLSTM uses a hidden size of 64.

The model produces two primary outputs:

1. Binary prediction: Normal vs Attack
2. Multiclass prediction: one of the 10 traffic classes

## Baseline Models

To evaluate the usefulness of the proposed architecture, several baseline approaches are included.

### Random Forest

Random Forest provides a strong classical machine-learning baseline and is particularly useful for multiclass classification.

Current test results:

```text
Binary:
Accuracy  : 59.17%
Precision : 58.55%
Recall    : 88.55%
F1        : 70.49%

Multiclass:
Accuracy      : 63.71%
Weighted F1   : 71.36%
Precision     : 86.50%
```

### XGBoost

XGBoost provides another tree-based baseline.

Current test results:

```text
Binary:
Accuracy  : 59.09%
Precision : 58.00%
Recall    : 93.15%
F1        : 71.49%

Multiclass:
Accuracy    : 52.21%
Weighted F1 : 59.85%
Precision   : 81.52%
```

### CNN-LSTM

A CNN-LSTM baseline is also included to compare a simpler deep-learning architecture against the Two-Stream architecture.

Current multiclass results:

```text
Accuracy    : 58.05%
Weighted F1 : 63.87%
```

## Two-Stream Model Results

The original Two-Stream model achieved the following test results.

Binary classification:

```text
Accuracy  : 70.98%
Precision : 65.51%
Recall    : 99.86%
F1        : 79.12%
FPR        : 64.40%
FNR        : 0.14%
```

The weighted Two-Stream version was also developed to address class imbalance through class-weighted learning.

Weighted Two-Stream test results:

```text
Accuracy  : 72.98%
Precision : 67.34%
Recall    : 98.87%
F1        : 80.12%
FPR        : 58.74%
FNR        : 1.13%
```

The weighted Two-Stream model is therefore used as the primary deep-learning binary detector.

For multiclass classification, Random Forest currently performs better than the Two-Stream model on the available evaluation results, so the project does not artificially claim that the deep-learning model is superior for every task.

## Model Selection Strategy

The project follows a practical model-selection approach instead of forcing one model to perform every task.

For binary detection:

```text
Primary:
Weighted Two-Stream Model

Supporting:
XGBoost
Random Forest
Isolation Forest
```

For multiclass classification:

```text
Primary:
Random Forest

Supporting:
Two-Stream Model
XGBoost
CNN-LSTM
```

This allows CyberAgent to use the strongest available model for each specific task while retaining multiple independent signals for investigation.

## Ensemble Experiment

Probability-level fusion was also investigated using Random Forest, XGBoost, and the Two-Stream model.

The validation set produced a useful fusion configuration, but the same configuration did not generalize sufficiently to the untouched test set. Therefore, the ensemble is currently retained as an **experimental/ablation component rather than the primary production classifier**.

This prevents the system from selecting a model only because it performs well on a validation set.

## Isolation Forest Anomaly Detection

CyberAgent includes an unsupervised Isolation Forest module.

Unlike the supervised classifiers, Isolation Forest is trained only using normal training traffic.

Configuration:

```text
Trees          : 200
Max Samples    : 10,000
Random State   : 42
Training Data  : Normal training traffic only
Threshold      : 95th percentile of normal training anomaly scores
```

The resulting test performance was:

```text
Accuracy  : 71.22%
Precision : 91.12%
Recall    : 52.89%
F1        : 66.93%
FPR       : 6.31%
FNR       : 47.11%
```

The anomaly detector provides an independent signal to the investigation layer.

An anomaly does **not** automatically mean that the traffic is malicious. It means that the traffic differs from the normal distribution learned by the Isolation Forest.

## Explainable AI

CyberAgent uses two complementary explainability methods.

### SHAP

SHAP is used for global and local model explanation.

The current global SHAP analysis uses the Random Forest multiclass model and 1,000 randomly selected test samples.

The global feature ranking is:

```text
1. dns_response_activity
2. protocol_distribution
3. inter_arrival_time
4. query_type_distribution
5. query_length
6. flow_duration
7. packet_size
8. query_rate
9. dns_query_frequency
10. packet_rate
```

The results indicate that several DNS-derived features have substantial influence on the Random Forest's multiclass decisions.

SHAP values represent feature contributions to model predictions. They should not be interpreted as proof of causal relationships.

### LIME

LIME provides local explanations for individual predictions. It approximates the model locally around a selected sample and identifies features that contribute to that specific prediction.

The investigation layer also supports fresh local LIME explanations instead of relying only on previously stored explanations.

## Investigation Layer

The investigation layer transforms raw model predictions into a more meaningful evidence package.

For an individual network-flow sample, the investigation engine combines:

```text
Binary Detection
        +
Random Forest Classification
        +
Two-Stream Classification
        +
Isolation Forest
        +
Model Consistency
        +
Local SHAP
        +
Local LIME
        +
Evidence Strength
        +
Analyst Summary
```

The result is stored in:

```text
results/investigation/evidence_extraction_results.json
```

## Evidence Extraction

The evidence extraction engine currently performs the following operations:

1. Loads network and DNS test features.
2. Loads the Random Forest classifier.
3. Loads the Two-Stream deep-learning model.
4. Loads the Isolation Forest model.
5. Initializes the local SHAP explainer.
6. Initializes the local LIME explainer.
7. Selects an individual test sample.
8. Generates a binary attack prediction.
9. Generates a Random Forest multiclass prediction.
10. Generates a Two-Stream multiclass prediction.
11. Calculates anomaly information.
12. Compares model predictions.
13. Calculates an evidence-strength score.
14. Generates an analyst-readable summary.
15. Saves the investigation evidence to JSON.

## Model Consistency

CyberAgent compares the predictions of multiple supervised models.

For example:

```text
Random Forest → Exploits
Two-Stream    → Exploits

Result → Agreement
```

or:

```text
Random Forest → Normal
Two-Stream    → Shellcode

Result → Disagreement
```

Model disagreement is deliberately preserved rather than hidden.

This is important because an investigation system should communicate uncertainty instead of presenting every prediction as a confirmed attack.

## Evidence Strength

CyberAgent calculates an internal evidence-strength score based on available signals such as detection confidence, model agreement, and anomaly information.

The score is used to prioritize investigations.

It is important to note that:

> Evidence strength is an internal investigation-prioritization score and is not an attack probability.

A low evidence score does not necessarily mean that an attack is impossible. It means that the available model evidence is not sufficiently consistent or strong.

## Example Investigation

A sample investigation can produce results such as:

```text
Binary result       : Attack
Attack probability  : 0.7393
Random Forest class : Normal
RF confidence       : 0.1717
Two-Stream class    : Shellcode
Anomaly detected    : False
Class agreement     : Disagreement
Evidence strength   : Low (50/100)
```

The system does not simply label this sample as "Shellcode attack."

Instead, the investigation engine recognizes that:

```text
Binary detector → Attack
Random Forest   → Normal
Two-Stream      → Shellcode
Isolation Forest→ Not anomalous
```

Therefore, the evidence is inconsistent and should be treated cautiously.

This is one of the central ideas behind CyberAgent: **detection should lead to investigation rather than automatically being treated as confirmation.**

## Technology Stack

### Programming Language

```text
Python
```

### Machine Learning

```text
scikit-learn
XGBoost
imbalanced-learn
```

### Deep Learning

```text
PyTorch
```

### Explainable AI

```text
SHAP
LIME
```

### Data Processing

```text
NumPy
Pandas
```

### Development

```text
Visual Studio Code
Git
GitHub
Jupyter / Google Colab where required
```

### Future Application Layer

```text
Streamlit
FastAPI
```

The current core project is CPU-compatible and does not require cloud infrastructure.

## Repository Structure

```text
Botnet-Detection-System/
│
├── data/
│   ├── raw/
│   │   ├── UNSW_NB15_training-set.csv
│   │   └── UNSW_NB15_testing-set.csv
│   │
│   └── processed/
│       ├── final_binary_train.npy
│       ├── final_binary_val.npy
│       ├── final_binary_test.npy
│       ├── final_multiclass_train.npy
│       ├── final_multiclass_val.npy
│       ├── final_multiclass_test.npy
│       ├── final_network_train.npy
│       ├── final_network_val.npy
│       ├── final_network_test.npy
│       ├── final_dns_train.npy
│       ├── final_dns_val.npy
│       ├── final_dns_test.npy
│       ├── feature_metadata.json
│       ├── network_ridge_ranking.csv
│       └── dns_ridge_ranking.csv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── utils.py
│   ├── inspect_dataset.py
│   ├── run_preprocessing.py
│   ├── prepare_final_data.py
│   ├── train_two_stream.py
│   ├── analyze_two_stream.py
│   ├── train_two_stream_weighted.py
│   ├── compare_two_stream_models.py
│   ├── analyze_prediction_distribution.py
│   ├── error_analysis_comparison.py
│   ├── generate_baseline_probabilities.py
│   ├── generate_validation_probabilities.py
│   │
│   ├── models/
│   │   └── two_stream_model.py
│   │
│   ├── baselines/
│   │   ├── evaluate.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── cnn_lstm.py
│   │   └── compare_results.py
│   │
│   ├── ensemble/
│   │   ├── probability_fusion.py
│   │   ├── optimize_fusion.py
│   │   └── evaluate_frozen_fusion.py
│   │
│   ├── anomaly/
│   │   ├── isolation_forest.py
│   │   └── analyze_anomalies.py
│   │
│   ├── explainability/
│   │   ├── feature_names.py
│   │   ├── shap_random_forest.py
│   │   └── lime_random_forest.py
│   │
│   └── investigation/
│       ├── evidence_extractor.py
│       ├── run_evidence_extraction.py
│       ├── local_shap.py
│       └── local_lime.py
│
├── models/
│   ├── baselines/
│   │   ├── random_forest_binary.pkl
│   │   ├── random_forest_multiclass.pkl
│   │   ├── xgboost_binary.pkl
│   │   ├── xgboost_multiclass.pkl
│   │   └── cnn_lstm_multiclass.pth
│   │
│   ├── two_stream_best.pth
│   ├── two_stream_weighted_best.pth
│   │
│   └── anomaly/
│       └── isolation_forest_normal.pkl
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   ├── comparison/
│   ├── ensemble/
│   ├── anomaly/
│   ├── explainability/
│   └── investigation/
│
├── app/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```powershell
git clone https://github.com/vardhanrote/Botnet-Detection-System.git
cd Botnet-Detection-System
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify Python:

```powershell
python --version
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Running the Project

### Inspect Dataset

```powershell
python -m src.inspect_dataset
```

### Run Preprocessing

```powershell
python -m src.run_preprocessing
```

### Prepare Final Data

```powershell
python -m src.prepare_final_data
```

### Train the Original Two-Stream Model

```powershell
python -m src.train_two_stream
```

### Train the Weighted Two-Stream Model

```powershell
python -m src.train_two_stream_weighted
```

### Train / Evaluate Baselines

Random Forest, XGBoost, and CNN-LSTM implementations are available under:

```text
src/baselines/
```

### Run Isolation Forest

```powershell
python -m src.anomaly.isolation_forest
```

### Analyze Anomalies

```powershell
python -m src.anomaly.analyze_anomalies
```

### Run Global SHAP

```powershell
python -m src.explainability.shap_random_forest
```

### Run LIME

```powershell
python -m src.explainability.lime_random_forest
```

### Run Evidence Extraction

```powershell
python -m src.investigation.run_evidence_extraction
```

The investigation results are saved to:

```text
results/investigation/evidence_extraction_results.json
```

## Current Project Results

The current implementation demonstrates that different models provide different strengths.

Random Forest currently provides the strongest multiclass baseline with approximately:

```text
63.71% multiclass accuracy
71.36% weighted F1
```

The weighted Two-Stream model currently provides strong binary detection performance with approximately:

```text
72.98% accuracy
98.87% recall
80.12% F1
```

Isolation Forest provides an independent anomaly signal with approximately:

```text
91.12% precision
52.89% recall
66.93% F1
6.31% false-positive rate
```

These results demonstrate why CyberAgent uses a multi-model investigation architecture instead of relying on a single classifier.

## Limitations

### DNS Features

The current DNS features are DNS-derived flow-level proxies. The system does not currently process complete raw DNS packet payloads or detailed DNS transaction logs.

### Temporal Modeling

UNSW-NB15 does not provide a clean timestamped sequence structure suitable for claiming true temporal traffic modeling. Therefore, the current BiLSTM should not be described as learning genuine chronological network sessions. Future versions can incorporate timestamped traffic windows to support real temporal modeling.

### Anomaly Detection

Isolation Forest detects deviation from the learned normal distribution. An anomaly is not automatically malicious.

### Explainability

SHAP explains feature contribution to model predictions but does not establish causality. LIME provides a local approximation of the model and should also not be treated as a definitive explanation of the real-world attack mechanism.

### Evidence Strength

Evidence strength is an internal prioritization mechanism. It is not a calibrated probability of attack.

### Dataset Dependence

The current models are trained and evaluated on UNSW-NB15. Real-world deployment would require additional datasets and validation against live or independently collected network traffic.

### Cloud

The current implementation intentionally avoids dependency on paid cloud infrastructure. The core system is designed to run locally.

## Security and Privacy

CyberAgent is designed for defensive cybersecurity research. The current attack replay and simulation roadmap is intended to operate at the dataset or controlled-simulation level rather than generating real-world malicious traffic. The system should only be tested against networks and systems for which the user has explicit authorization.

Sensitive information such as API keys, credentials, private datasets, environment files, or personal information should never be committed to the repository.

## Current Development Status

The current project has completed the core machine-learning and investigation foundation.

### Completed

```text
✓ UNSW-NB15 dataset preparation
✓ Dataset inspection
✓ Feature engineering
✓ Network feature stream
✓ DNS-derived feature stream
✓ Ridge feature selection
✓ Training/validation/test separation
✓ SMOTE-based training balancing
✓ Two-Stream CNN architecture
✓ Multi-Head Attention
✓ BiLSTM
✓ Binary detection
✓ Multiclass classification
✓ Random Forest baseline
✓ XGBoost baseline
✓ CNN-LSTM baseline
✓ Model comparison
✓ Ensemble experiment
✓ Isolation Forest
✓ SHAP global analysis
✓ LIME local analysis
✓ Local SHAP
✓ Local LIME
✓ Evidence extraction engine
✓ Model consistency analysis
✓ Evidence strength scoring
✓ Analyst-oriented investigation summary
✓ JSON investigation output
```

## Future Roadmap

The next stage of CyberAgent will extend the current evidence extraction engine into a complete cybersecurity investigation platform.

### Threat Profiling

Create structured threat profiles based on detected attack classes, model evidence, anomaly information, and feature behavior.

### Severity Engine

Develop a more detailed severity framework that considers detection confidence, attack category, anomaly level, model agreement, and other investigation signals.

### Threat Intelligence

Build a local threat-intelligence knowledge base that can associate detected behaviors with known indicators, attack patterns, and defensive information.

### MITRE ATT&CK Mapping

Map detected attack categories and behaviors to relevant MITRE ATT&CK techniques where sufficient evidence exists.

### AI Investigation Agent

Develop the CyberAgent investigation layer so that an analyst can provide a suspicious event and receive a structured investigation containing:

```text
Detection
↓
Classification
↓
Evidence
↓
Explanation
↓
Threat Profile
↓
MITRE Mapping
↓
Recommended Investigation Steps
↓
Final Assessment
```

The agent should use the available evidence rather than generating unsupported conclusions.

### Attack Replay

Implement dataset-level attack replay so previously observed suspicious samples can be replayed through the detection and investigation pipeline.

### Security Regression Testing

Create a regression-testing framework to ensure that future model or rule changes do not silently reduce detection performance.

### SOC Application

Build a Streamlit-based SOC interface for:

```text
Alerts
Investigations
Model Predictions
Anomaly Scores
SHAP Explanations
LIME Explanations
Threat Profiles
MITRE ATT&CK
Evidence
Incident Reports
```

### Incident Reporting

Generate structured incident reports containing:

```text
Incident Summary
Detection Details
Predicted Attack Class
Model Confidence
Anomaly Information
Feature Evidence
Model Agreement
Threat Profile
MITRE Mapping
Recommended Actions
Investigation Conclusion
```

## Research Contribution

The main contribution of CyberAgent is not simply another network classifier. The project combines **detection and investigation** into one workflow. Instead of treating a machine-learning prediction as the final answer, CyberAgent combines multiple supervised models, unsupervised anomaly detection, explainability, model-consistency analysis, and evidence scoring to produce a more transparent investigation result. The architecture therefore moves from a traditional "detect and classify" approach toward an "detect, explain, compare, investigate, and report" approach.

## Conclusion

CyberAgent provides a modular foundation for intelligent network-threat detection and investigation. The current system combines the UNSW-NB15 dataset, engineered network and DNS-derived flow features, Ridge-based feature selection, SMOTE-based class balancing, a Two-Stream CNN architecture, Multi-Head Attention, BiLSTM representation learning, Random Forest and XGBoost baselines, Isolation Forest anomaly detection, SHAP, LIME, local explanations, model consistency analysis, and evidence extraction. The current implementation demonstrates that no single model provides the complete picture: the deep-learning detector can provide strong binary detection, Random Forest provides stronger multiclass performance, and Isolation Forest contributes an independent anomaly signal. CyberAgent therefore treats model outputs as pieces of evidence that should be combined and investigated rather than blindly accepted. The project is designed to evolve into a complete AI-assisted SOC investigation platform incorporating threat profiling, threat intelligence, MITRE ATT&CK mapping, security regression testing, controlled attack replay, an AI investigation agent, SOC visualization, and automated incident reporting.

## Project

**CyberAgent: Agent-Assisted Network Threat Detection and Investigation using Machine Learning**

**Repository:** `vardhanrote/Botnet-Detection-System`

**Primary Dataset:** UNSW-NB15

**Development Environment:** Python + PyTorch + scikit-learn + XGBoost + SHAP + LIME + VS Code

**Project Focus:** Network Threat Detection, Machine Learning, Deep Learning, Anomaly Detection, Explainable AI, and AI-Assisted Cybersecurity Investigation
