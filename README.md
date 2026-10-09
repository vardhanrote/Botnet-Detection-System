# CyberAgent: Agent-Assisted Network Threat Detection and Investigation Using Machine Learning

## Overview

CyberAgent is a research-oriented network security system designed to detect, classify, investigate, and explain suspicious network traffic using machine learning, deep learning, anomaly detection, and explainable AI.

Traditional intrusion detection systems focus primarily on identifying malicious traffic. CyberAgent extends this process by examining the evidence behind predictions, comparing independent model outputs, identifying anomalous behavior, and communicating uncertainty to support security investigation.

The project uses the **UNSW-NB15 network intrusion dataset** and includes two related development tracks:

1. **Network detection and investigation:** Binary and multiclass classification, a two-stream deep-learning architecture, classical machine-learning baselines, anomaly detection, explainability, and structured evidence extraction.
2. **Consensus-Stable Evidence Pipeline (CSEP):** An experimental research pipeline combining stability-selected features, anomaly-validated oversampling, explanation agreement, and uncertainty-aware verdict routing.

The long-term objective is to develop a local, SOC-style investigation application that converts network alerts into structured, evidence-based investigations.

## Problem Statement

A malicious-traffic prediction alone is insufficient for a reliable security investigation. Analysts also need to understand the predicted attack category, the evidence influencing the prediction, the level of confidence, whether different analysis methods agree, and whether the event requires further investigation.

CyberAgent addresses this problem through a modular pipeline that combines supervised classification, unsupervised anomaly detection, explainable AI, model-consistency analysis, and evidence-based investigation.

The CSEP research track further investigates whether stable feature selection, quality-controlled oversampling, explanation agreement, and uncertainty-aware routing can improve the reliability of automated alert handling.

## Main Objectives

- Detect normal and potentially malicious network traffic.
- Classify traffic into multiple attack categories.
- Compare classical machine-learning and deep-learning models.
- Investigate network-flow characteristics using engineered features.
- Identify unusual traffic using unsupervised anomaly detection.
- Explain model predictions using SHAP and LIME.
- Compare model predictions and explanation rankings.
- Evaluate confidence, calibration, and feature-selection stability.
- Route uncertain alerts for analyst review.
- Produce structured investigation evidence.
- Develop a foundation for threat profiling, MITRE ATT&CK mapping, an AI investigation agent, and incident reporting.

## Key Features

### Existing detection and investigation track

The project includes implementations for:

- UNSW-NB15 data preparation and feature engineering.
- Binary attack detection and multiclass classification.
- Random Forest and XGBoost baselines.
- A custom two-stream CNN architecture with feature fusion, multi-head attention, and BiLSTM.
- A CNN-LSTM comparison model.
- Isolation Forest anomaly detection.
- SHAP global and local explanations.
- LIME local explanations.
- Model-consistency analysis and evidence-strength scoring.
- Structured investigation evidence saved as JSON.

These components belong to the existing detection and investigation codebase. Their availability does not imply that every component has been revalidated in the current CSEP experiment.

### CSEP research track

CSEP is the current experimental focus. Its components are:

| Component                            | Purpose                                                                                                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stability-Selected Features (SSF)    | Identify features that remain useful across repeated training folds.                                                                                       |
| Anomaly-Validated Oversampling (AVO) | Filter candidate synthetic minority samples using class locality and anomaly checks.                                                                       |
| Consensus Attribution                | Compare SHAP and LIME feature rankings using top-k Jaccard similarity and Kendall's tau.                                                                   |
| Agreement-Gated Verdict Routing      | Combine confidence, attribution agreement, anomaly score, and class plausibility to determine whether an alert should be automatically routed or reviewed. |
| Evaluation Metrics                   | Measure classification quality, calibration, feature stability, and alert-routing behavior.                                                                |

The verdict gate is a decision-support mechanism. An automatic routing label is not proof that an incident occurred, and explanation agreement does not establish that an explanation is correct.

## System Architecture

The high-level architecture contains two connected development tracks.

### Track A: Detection and Investigation

```text
UNSW-NB15 Dataset
        |
        v
Data Cleaning and Preprocessing
        |
        v
Feature Engineering
        |
        v
Feature Selection and Training Preparation
        |
        +---------------------------+
        |                           |
        v                           v
Network Feature Stream       DNS-Derived Feature Stream
        |                           |
        v                           v
Conv1D and Pooling           Conv1D and Pooling
        |                           |
        +-------------+-------------+
                      |
                      v
                 Feature Fusion
                      |
                      v
              Multi-Head Attention
                      |
                      v
                    BiLSTM
                      |
             +--------+--------+
             |                 |
             v                 v
       Binary Output     Multiclass Output
             |                 |
             +--------+--------+
                      |
                      v
       Classical ML and Anomaly Signals
                      |
                      v
              SHAP and LIME
                      |
                      v
       Evidence Extraction and Scoring
                      |
                      v
          Structured Investigation JSON
```

### Track B: CSEP Research Pipeline

```text
UNSW-NB15 Training Data
          |
          v
Training-Only Preprocessing
          |
          v
Baseline Feature Selection
          |
          +-----------------------------+
          |                             |
          v                             v
       SSF + SMOTE                   SSF + AVO
          |                             |
          +--------------+--------------+
                         |
                         v
                Model Evaluation
                         |
                         v
           Classification and Calibration
                         |
                         v
            SHAP/LIME Attribution Agreement
                         |
                         v
             Agreement-Gated Verdict Routing
                         |
                         v
            Metrics and Experiment Reports
```

The CSEP experiment runner currently compares a Ridge-selection plus SMOTE baseline, SSF plus SMOTE, and SSF plus AVO. The integrated runner is still under research validation.

## Dataset

CyberAgent uses the UNSW-NB15 network intrusion dataset.

The supplied training and testing files have the following dimensions:

| Dataset  |    Rows | Columns |
| -------- | ------: | ------: |
| Training | 175,341 |      45 |
| Testing  |  82,332 |      45 |

The dataset contains normal traffic and multiple attack categories.

| Attack category | Training samples |
| --------------- | ---------------: |
| Normal          |           56,000 |
| Generic         |           40,000 |
| Exploits        |           33,393 |
| Fuzzers         |           18,184 |
| DoS             |           12,264 |
| Reconnaissance  |           10,491 |
| Analysis        |            2,000 |
| Backdoor        |            1,746 |
| Shellcode       |            1,133 |
| Worms           |              130 |

The dataset is highly imbalanced, particularly for rare attack categories. This makes per-class recall, macro-F1, and evaluation on rare attacks important in addition to overall accuracy.

The CSEP smoke experiment uses stratified subsets of 8,000 training rows and 3,000 test rows. These subsets are intended for debugging and preliminary comparisons, not final research claims.

## Feature Engineering

The existing two-stream track groups engineered features into two streams.

### Network features

- `inter_arrival_time`
- `protocol_distribution`
- `packet_rate`
- `flow_duration`
- `packet_size`

### DNS-derived flow features

- `query_type_distribution`
- `dns_query_frequency`
- `query_rate`
- `dns_response_activity`
- `query_length`

**Important limitation:** These DNS-related features are derived proxies calculated from available flow-level fields. The current feature-engineering approach does not establish that the system processes raw DNS packets or complete DNS transaction telemetry.

The CSEP experiment runner separately processes categorical and numerical fields from the supplied dataset and encodes them for its classification experiments. Its smoke-test configuration produced 194 encoded features. These 194 features should not be confused with the two-stream architecture's selected feature groups.

## Existing Two-Stream Deep-Learning Model

The existing deep-learning architecture contains two independent feature branches.

Each branch applies convolutional layers, batch normalization, activation functions, and pooling. The learned representations are fused and processed using multi-head attention, a residual connection, and BiLSTM layers before the output heads generate predictions.

The architecture is intended to support:

- Binary classification: Normal versus Attack.
- Multiclass classification: Ten traffic classes.

The configured attention layer uses four heads, and the BiLSTM uses a hidden size of 64.

Because UNSW-NB15 does not provide a clean chronological sequence of network sessions for this project, the BiLSTM should not be described as demonstrating genuine chronological traffic modeling without additional timestamped sequence data.

## Baseline Models and Previously Reported Results

The existing project documentation reports the following results from earlier model evaluations. These are historical project results, separate from the new CSEP smoke experiment.

### Random Forest

Previously reported multiclass results:

- Accuracy: 63.71%
- Weighted F1: 71.36%
- Precision: 86.50%

### XGBoost

Previously reported results:

- Binary accuracy: 59.09%
- Binary recall: 93.15%
- Binary F1: 71.49%
- Multiclass accuracy: 52.21%
- Multiclass weighted F1: 59.85%

### CNN-LSTM

Previously reported multiclass results:

- Accuracy: 58.05%
- Weighted F1: 63.87%

### Weighted Two-Stream Model

Previously reported binary results:

- Accuracy: 72.98%
- Precision: 67.34%
- Recall: 98.87%
- F1: 80.12%
- False-positive rate: 58.74%
- False-negative rate: 1.13%

### Isolation Forest

Previously reported results:

- Accuracy: 71.22%
- Precision: 91.12%
- Recall: 52.89%
- F1: 66.93%
- False-positive rate: 6.31%
- False-negative rate: 47.11%

These metrics should be rechecked against their original experiment reports before being used in a final publication or presentation. Results from different experiments should not be compared as if they came from an identical split and evaluation protocol unless that has been verified.

## CSEP Smoke Experiment

### Experimental configuration

The first integrated binary smoke experiment completed successfully.

| Setting                          | Value                 |
| -------------------------------- | --------------------- |
| Dataset                          | UNSW-NB15             |
| Target                           | Binary classification |
| Training subset                  | 8,000 rows            |
| Test subset                      | 3,000 rows            |
| Encoded features                 | 194                   |
| SSF folds                        | 2                     |
| Selected features per experiment | 10                    |

The experiment compares:

1. Ridge feature selection + SMOTE.
2. Stability-Selected Features + SMOTE.
3. Stability-Selected Features + Anomaly-Validated Oversampling.

### Preliminary results

| Experiment    | Accuracy | Macro-F1 | Weighted F1 |    ECE | Brier score |
| ------------- | -------: | -------: | ----------: | -----: | ----------: |
| Ridge + SMOTE |   83.73% |   83.18% |      83.49% | 0.0787 |      0.2399 |
| SSF + SMOTE   |   86.20% |   85.66% |      85.95% | 0.0435 |      0.1807 |
| SSF + AVO     |   86.07% |   85.46% |      85.76% | 0.0489 |      0.1829 |

Lower ECE and Brier scores generally indicate better probability calibration and probabilistic prediction quality, respectively.

In this smoke test, SSF + SMOTE achieved the highest macro-F1 and accuracy among the three configurations. SSF + AVO produced a slightly lower macro-F1 and used fewer training rows after resampling.

These results are **preliminary debugging results only**. The experiment used a small subset and two SSF folds. They do not establish final model performance or prove that CSEP improves generalization. The full experiments, methodological checks, and ablation studies remain necessary.

### Runtime

| Experiment    | Reported training time |
| ------------- | ---------------------: |
| Ridge + SMOTE |           1.68 seconds |
| SSF + SMOTE   |         474.12 seconds |
| SSF + AVO     |         347.00 seconds |

The SSF procedure is computationally expensive because it repeatedly estimates feature importance across folds. Runtime is therefore an important practical consideration alongside predictive performance.

### Output files

The completed smoke experiment saved its outputs under `experiments/results/`:

- `csep_experiment_metrics.json`
- `csep_experiment_summary.csv`
- `ssf_smote_feature_report.csv`
- `ssf_avo_feature_report.csv`
- `ssf_avo_oversampling_report.json`

These reports are currently local experiment artifacts. They are not yet the final research results.

### Research evaluation plan

The next CSEP experiments should investigate:

- Baseline versus SSF, AVO, and the combined configuration.
- Ablations that isolate each component's contribution.
- Macro-F1 and per-class recall.
- Calibration using ECE, Brier score, and reliability analysis.
- Feature-selection stability across random seeds and folds.
- SHAP/LIME top-k ranking agreement.
- Automatic-routing coverage and error rates.
- The number of incorrect predictions routed for human review.
- Generalization to held-out attack categories, using strict exclusion of those categories from training and all fitted preprocessing, feature selection, oversampling, calibration, and threshold selection.

The final test set should remain untouched during model selection and tuning.

## Explainable AI

### SHAP

SHAP estimates feature contributions to model predictions and can support global and local interpretation.

The earlier project documentation reports a global SHAP analysis using the Random Forest multiclass model and 1,000 randomly selected test samples.

Reported feature ranking:

1. `dns_response_activity`
2. `protocol_distribution`
3. `inter_arrival_time`
4. `query_type_distribution`
5. `query_length`
6. `flow_duration`
7. `packet_size`
8. `query_rate`
9. `dns_query_frequency`
10. `packet_rate`

These results should be verified against the saved explanation outputs before being reused in final claims.

### LIME

LIME provides a local approximation of a model around a selected sample to identify features influencing that prediction.

### CSEP attribution consensus

The CSEP consensus module compares SHAP and LIME feature rankings using:

- Top-k Jaccard similarity.
- Kendall's tau.
- Shared-feature counts.
- A combined agreement score.

Agreement indicates consistency between explanation rankings, not proof that the explanations are correct or that the model's prediction is true.

## Anomaly Detection and Verdict Routing

Isolation Forest provides an independent anomaly signal by estimating how unusual a sample is relative to its learned reference distribution.

An anomalous sample is not automatically malicious, and an ordinary-looking sample is not necessarily benign.

CSEP's verdict gate combines classifier confidence, attribution agreement, anomaly score, and class plausibility to route alerts. Its intended routes include automatic handling, analyst review, and possible novel-threat investigation.

These are investigation-routing decisions, not ground-truth incident confirmations. Thresholds and routing performance must be validated before the mechanism can be described as production-ready.

## Investigation Evidence

The existing investigation track is designed to combine:

- Binary classification.
- Multiclass predictions.
- Anomaly information.
- Model consistency.
- Local SHAP and LIME explanations.
- Internal evidence-strength scoring.
- An analyst-readable summary.

The earlier project documentation identifies the output location as:

`results/investigation/evidence_extraction_results.json`

Evidence strength is an internal prioritization score, not a calibrated probability of attack. Disagreement between models should be preserved and communicated rather than hidden.

## Technology Stack

| Area                     | Technologies                       |
| ------------------------ | ---------------------------------- |
| Programming              | Python                             |
| Data processing          | NumPy, Pandas                      |
| Machine learning         | scikit-learn, XGBoost              |
| Class balancing          | imbalanced-learn                   |
| Deep learning            | PyTorch                            |
| Explainability           | SHAP, LIME                         |
| Research experiments     | CSEP modules, JSON and CSV reports |
| Development              | Visual Studio Code, Git, GitHub    |
| Future application layer | Streamlit, FastAPI                 |

The core project is designed to run locally and does not require paid cloud infrastructure.

## Repository Structure

```text
Botnet-Detection-System/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── models/
│   ├── baselines/
│   ├── ensemble/
│   ├── anomaly/
│   ├── explainability/
│   ├── investigation/
│   └── features/
├── csep/
│   ├── __init__.py
│   ├── ssf.py
│   ├── avo.py
│   ├── consensus_attribution.py
│   ├── verdict_gate.py
│   └── metrics.py
├── experiments/
│   ├── run_csep_experiments.py
│   └── results/
├── models/
├── results/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

The structure above is a high-level overview. Some folders contain existing model artifacts or generated files that may not be tracked in Git.

## Installation

Clone the repository:

```powershell
git clone https://github.com/vardhanrote/Botnet-Detection-System.git
cd Botnet-Detection-System
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Check the Python installation:

```powershell
python --version
```

Check PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Ensure the UNSW-NB15 training and testing files are present under `data/raw/` before running the dataset-dependent experiments.

## Running the Project

### Inspect the dataset

```powershell
python -m src.inspect_dataset
```

### Run preprocessing

```powershell
python -m src.run_preprocessing
```

### Prepare final model data

```powershell
python -m src.prepare_final_data
```

### Train the two-stream models

```powershell
python -m src.train_two_stream
python -m src.train_two_stream_weighted
```

### Run anomaly detection

```powershell
python -m src.anomaly.isolation_forest
python -m src.anomaly.analyze_anomalies
```

### Run explainability

```powershell
python -m src.explainability.shap_random_forest
python -m src.explainability.lime_random_forest
```

### Run investigation evidence extraction

```powershell
python -m src.investigation.run_evidence_extraction
```

The previously documented output path is:

`results/investigation/evidence_extraction_results.json`

### Run CSEP smoke experiment

From the project root, with the virtual environment activated:

```powershell
python experiments/run_csep_experiments.py --smoke --target binary --ssf-folds 2
```

This runs a smaller binary experiment for pipeline verification. Its metrics are preliminary and must not be used as final performance claims.

### Run CSEP binary experiments

```powershell
python experiments/run_csep_experiments.py --target binary
```

### Run CSEP multiclass experiments

```powershell
python experiments/run_csep_experiments.py --target multiclass
```

The full runs may be computationally expensive. Review the experiment configuration and data-handling methodology before treating their output as final research results.

## Current Development Status

### Implemented

- UNSW-NB15 data preparation and existing preprocessing modules.
- Existing two-stream deep-learning and baseline model code.
- Existing anomaly detection and explainability modules.
- Existing investigation and evidence-extraction code.
- CSEP feature-selection, oversampling, consensus, verdict-gating, and metric modules.
- CSEP experiment runner.
- Supporting tests for the CSEP modules.
- Successful binary smoke experiment with three compared configurations.
- Saved preliminary metrics and feature/oversampling reports.

### In progress

- Verification of the CSEP experimental methodology.
- Full-data binary and multiclass experiments.
- Ablation studies and feature-stability analysis.
- Calibration and alert-routing evaluation.
- Rare-class and held-out attack-category generalization.
- Integration and validation of the research pipeline with the complete investigation workflow.

### Planned

- A complete investigator-facing application.
- Threat profiles and a severity framework.
- A local threat-intelligence knowledge base.
- Evidence-grounded MITRE ATT&CK mapping.
- An AI-assisted investigation agent.
- Dataset-level attack replay.
- Security regression testing.
- Structured incident-report generation.
- End-to-end application tests and a final demonstration.

## Limitations

### Dataset dependence

The current experiments use UNSW-NB15. Additional datasets and independent validation are needed before making claims about real-world deployment.

### DNS-derived features

The DNS-related features are flow-derived proxies and should not be represented as complete raw DNS telemetry.

### Temporal modeling

The current dataset does not establish a clean chronological sequence of network sessions for this project. Genuine temporal modeling claims require appropriate timestamped sequence data.

### Anomaly detection

Anomaly scores represent deviation from a learned reference distribution, not a definitive maliciousness judgment.

### Explainability

SHAP and LIME provide model-oriented explanations, not causal proof of an attack mechanism.

### Synthetic oversampling

Synthetic examples can introduce artifacts or distort class boundaries. AVO requires further validation to establish whether its filtering improves generalization.

### Verdict routing

Automatic routing thresholds must be calibrated and evaluated on appropriate validation data. Automatic handling must not be presented as confirmed incident detection.

### Research results

The CSEP smoke-test results use a reduced dataset and a small number of SSF folds. Full-data experiments, ablations, and leakage checks are required before drawing final conclusions.

### Cloud and security

The current implementation is designed for local execution without a dependency on paid cloud infrastructure. Testing should remain within datasets, controlled simulations, and networks for which explicit authorization exists.

Sensitive information such as API keys, credentials, private datasets, and environment files must not be committed to the repository.

## Research Direction

CyberAgent aims to move beyond a simple detection-and-classification pipeline toward evidence-based investigation.

The CSEP research question is:

> Does a stability- and consensus-aware investigation pipeline route uncertain network alerts more reliably than a conventional detector, without causing an unacceptable increase in false alarms or missed attacks?

The experiments will evaluate predictive performance, calibration, feature stability, explanation agreement, and routing behavior. The results must determine which components provide measurable benefits and which introduce additional cost or limitations.

The project does not claim that CSEP is novel or production-ready solely because the modules have been implemented. Those claims require further comparative evaluation and literature review.

## Conclusion

CyberAgent combines network-threat detection, multiclass classification, anomaly detection, explainable AI, and evidence extraction with a developing research pipeline for more reliable alert investigation.

The existing two-stream and classical-model implementations provide the detection foundation. CSEP adds stability-selected features, anomaly-validated oversampling, explanation-ranking agreement, and uncertainty-aware verdict routing.

The first integrated CSEP binary smoke experiment has completed and produced preliminary results. The next priorities are to validate the experimental methodology, conduct full-data comparisons and ablations, measure rare-class performance, and integrate the validated research pipeline into the investigator-facing application.

**Project:** CyberAgent — Agent-Assisted Network Threat Detection and Investigation Using Machine Learning  
**Repository:** [vardhanrote/Botnet-Detection-System](https://github.com/vardhanrote/Botnet-Detection-System)  
**Primary dataset:** UNSW-NB15  
**Development environment:** Python, PyTorch, scikit-learn, XGBoost, SHAP, LIME, and Visual Studio Code  
**Research focus:** Network threat detection, anomaly detection, explainable AI, feature stability, and AI-assisted cybersecurity investigation
