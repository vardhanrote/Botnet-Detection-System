"""
CyberAgent SOC Application

Local Streamlit interface for:
- Threat detection
- Evidence review
- Threat classification
- Severity assessment
- Explainability
- MITRE ATT&CK candidate mappings
- Investigation Agent results
- Incident report generation
"""

import json
import streamlit as st

from data_loader import (
    build_sample_dataset,
)

from report_generator import (
    generate_incident_report,
    save_incident_report,
)


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="CyberAgent SOC",
    page_icon="🛡️",
    layout="wide",
)


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 16px;
        color: #777777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.markdown(
    '<div class="main-title">CyberAgent SOC</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Agent-Assisted Network Threat Detection and Investigation
    using Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------

try:

    samples = build_sample_dataset()

except Exception as error:

    st.error(
        f"Unable to load CyberAgent results:\n\n{error}"
    )

    st.stop()


if not samples:

    st.warning(
        "No investigation results were found."
    )

    st.stop()


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.title(
    "Investigation Control"
)

st.sidebar.write(
    "Select a network traffic sample "
    "to investigate."
)


sample_numbers = [
    sample["sample_number"]
    for sample in samples
]


selected_sample_number = st.sidebar.selectbox(
    "Select Sample",
    sample_numbers,
)


selected_sample = next(
    sample
    for sample in samples
    if sample["sample_number"]
    == selected_sample_number
)


investigation = selected_sample.get(
    "investigation",
    {},
)

evidence = selected_sample.get(
    "evidence",
    {},
)

threat_profile = selected_sample.get(
    "threat_profile",
    {},
)

threat_intelligence = selected_sample.get(
    "threat_intelligence",
    {},
)


# ----------------------------------------------------------------------
# Extract main information
# ----------------------------------------------------------------------

investigation_verdict = investigation.get(
    "investigation_verdict",
    {},
)

verdict = investigation_verdict.get(
    "verdict",
    "Unknown",
)

threat_class = investigation.get(
    "threat_class",
    "Unknown",
)

severity = investigation.get(
    "severity",
    {},
)

severity_level = severity.get(
    "level",
    "Unknown",
)

severity_score = severity.get(
    "score",
    "N/A",
)

agent_confidence = investigation.get(
    "agent_confidence",
    investigation.get(
        "confidence",
        {},
    ),
)

confidence_score = agent_confidence.get(
    "score",
    "N/A",
)

confidence_level = agent_confidence.get(
    "level",
    "Unknown",
)


binary = evidence.get(
    "binary_detection",
    {},
)

classification = evidence.get(
    "attack_classification",
    {},
)

rf = classification.get(
    "random_forest",
    {},
)

two_stream = classification.get(
    "two_stream",
    {},
)

anomaly = evidence.get(
    "anomaly_detection",
    {},
)

consistency = evidence.get(
    "model_consistency",
    {},
)

evidence_strength = evidence.get(
    "evidence_strength",
    {},
)


# ----------------------------------------------------------------------
# Top metrics
# ----------------------------------------------------------------------

st.subheader(
    f"Sample {selected_sample_number}"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Threat Class",
        threat_class,
    )


with col2:

    st.metric(
        "Severity",
        severity_level,
        f"{severity_score}/100",
    )


with col3:

    st.metric(
        "Agent Confidence",
        confidence_level,
        f"{confidence_score}/100",
    )


with col4:

    prediction = binary.get(
        "prediction",
        "Unknown",
    )

    st.metric(
        "Binary Detection",
        prediction,
    )


st.divider()


# ----------------------------------------------------------------------
# Investigation verdict
# ----------------------------------------------------------------------

st.subheader(
    "Investigation Verdict"
)

st.info(verdict)


# ----------------------------------------------------------------------
# Detection assessment
# ----------------------------------------------------------------------

st.subheader(
    "Detection Assessment"
)

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "### Binary Detector"
    )

    st.write(
        f"Prediction: "
        f"**{binary.get('prediction', 'Unknown')}**"
    )

    attack_probability = binary.get(
        "attack_probability"
    )

    if attack_probability is not None:

        st.progress(
            min(
                max(
                    float(attack_probability),
                    0.0,
                ),
                1.0,
            )
        )

        st.write(
            f"Attack probability: "
            f"**{attack_probability:.2%}**"
        )


with col2:

    st.markdown(
        "### Supervised Classifiers"
    )

    st.write(
        f"Random Forest: "
        f"**{rf.get('predicted_class', 'Unknown')}**"
    )

    if rf.get("confidence") is not None:

        st.write(
            f"RF confidence: "
            f"**{rf['confidence']:.2%}**"
        )

    st.write(
        f"Two-Stream: "
        f"**{two_stream.get('multiclass_prediction', 'Unknown')}**"
    )

    if two_stream.get(
        "multiclass_confidence"
    ) is not None:

        st.write(
            f"Two-Stream confidence: "
            f"**{two_stream['multiclass_confidence']:.2%}**"
        )


# ----------------------------------------------------------------------
# Evidence assessment
# ----------------------------------------------------------------------

st.subheader(
    "Evidence Assessment"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Anomaly Detected",
        "Yes"
        if anomaly.get("is_anomaly")
        else "No",
    )


with col2:

    st.metric(
        "Classifier Disagreement",
        "Yes"
        if consistency.get(
            "model_disagreement"
        )
        else "No",
    )


with col3:

    st.metric(
        "Evidence Strength",
        evidence_strength.get(
            "level",
            "Unknown",
        ),
        (
            f"{evidence_strength.get('score', 'N/A')}/100"
        ),
    )


if anomaly.get("anomaly_score") is not None:

    st.write(
        f"Isolation Forest anomaly score: "
        f"**{anomaly['anomaly_score']:.6f}**"
    )


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Key Findings",
        "Explainability",
        "MITRE ATT&CK",
        "Recommendations",
        "Raw Data",
    ]
)


# ----------------------------------------------------------------------
# Tab 1 - Findings
# ----------------------------------------------------------------------

with tab1:

    st.markdown(
        "### Investigation Findings"
    )

    findings = investigation.get(
        "key_findings",
        investigation.get(
            "findings",
            [],
        ),
    )

    if findings:

        for index, finding in enumerate(
            findings,
            start=1,
        ):

            st.write(
                f"**{index}.** {finding}"
            )

    else:

        st.write(
            "No findings available."
        )

    st.markdown(
        "### Investigation Reasoning"
    )

    reasoning = investigation.get(
        "reasoning",
        [],
    )

    if reasoning:

        for item in reasoning:

            st.write(
                f"- {item}"
            )

    else:

        st.write(
            "No reasoning available."
        )


# ----------------------------------------------------------------------
# Tab 2 - Explainability
# ----------------------------------------------------------------------

with tab2:

    st.markdown(
        "### Model Explainability"
    )

    explainability = investigation.get(
        "explainability",
        {},
    )

    shap_data = explainability.get(
        "shap",
        {},
    )

    lime_data = explainability.get(
        "lime",
        {},
    )

    st.markdown(
        "#### SHAP"
    )

    if shap_data:

        st.json(
            shap_data
        )

    else:

        st.info(
            "No local SHAP information available."
        )

    st.markdown(
        "#### LIME"
    )

    if lime_data:

        st.json(
            lime_data
        )

    else:

        st.info(
            "No local LIME information available."
        )

    st.caption(
        "SHAP and LIME explain model behavior. "
        "They should not be interpreted as causal evidence."
    )


# ----------------------------------------------------------------------
# Tab 3 - MITRE
# ----------------------------------------------------------------------

with tab3:

    st.markdown(
        "### MITRE ATT&CK Candidate Mappings"
    )

    mappings = threat_intelligence.get(
        "mitre_mappings",
        threat_intelligence.get(
            "mappings",
            [],
        ),
    )

    if mappings:

        for mapping in mappings:

            technique_id = mapping.get(
                "technique_id",
                "N/A",
            )

            technique_name = mapping.get(
                "technique_name",
                "Unknown",
            )

            st.markdown(
                f"#### {technique_id} — "
                f"{technique_name}"
            )

            st.write(
                f"Tactic: "
                f"**{mapping.get('tactic', 'Unknown')}**"
            )

            st.write(
                f"Relevance: "
                f"**{mapping.get('relevance', 'Unknown')}**"
            )

            st.write(
                f"Confidence: "
                f"**{mapping.get('confidence', 'Unknown')}**"
            )

            if mapping.get("reason"):

                st.write(
                    f"Reason: "
                    f"{mapping['reason']}"
                )

            if mapping.get(
                "required_corroborating_evidence"
            ):

                st.write(
                    "Required corroborating evidence:"
                )

                for item in mapping[
                    "required_corroborating_evidence"
                ]:

                    st.write(
                        f"- {item}"
                    )

            st.divider()

    else:

        st.info(
            "No candidate MITRE ATT&CK mappings."
        )

    mapping_policy = threat_intelligence.get(
        "mapping_policy"
    )

    if mapping_policy:

        st.warning(
            mapping_policy
        )


# ----------------------------------------------------------------------
# Tab 4 - Recommendations
# ----------------------------------------------------------------------

with tab4:

    st.markdown(
        "### Recommended Actions"
    )

    recommendations = investigation.get(
        "recommended_actions",
        [],
    )

    if recommendations:

        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):

            st.write(
                f"**{index}.** {recommendation}"
            )

    else:

        st.info(
            "No recommendations available."
        )

    st.markdown(
        "### Investigation Limitations"
    )

    limitations = investigation.get(
        "limitations",
        [],
    )

    if limitations:

        for limitation in limitations:

            st.write(
                f"- {limitation}"
            )


# ----------------------------------------------------------------------
# Tab 5 - Raw data
# ----------------------------------------------------------------------

with tab5:

    st.markdown(
        "### Investigation Agent Output"
    )

    st.json(
        investigation
    )

    st.markdown(
        "### Evidence Output"
    )

    st.json(
        evidence
    )

    st.markdown(
        "### Threat Profile"
    )

    st.json(
        threat_profile
    )

    st.markdown(
        "### Threat Intelligence"
    )

    st.json(
        threat_intelligence
    )


# ----------------------------------------------------------------------
# Incident report
# ----------------------------------------------------------------------

st.divider()

st.subheader(
    "Incident Report"
)

st.write(
    "Generate a structured Markdown report "
    "for the selected sample."
)


if st.button(
    "Generate Incident Report",
    type="primary",
):

    report = generate_incident_report(
        selected_sample
    )

    report_path = save_incident_report(
        selected_sample
    )

    st.success(
        f"Incident report generated successfully."
    )

    st.write(
        f"Saved to: `{report_path}`"
    )

    st.download_button(
        label="Download Incident Report",
        data=report,
        file_name=(
            f"incident_report_sample_"
            f"{selected_sample_number}.md"
        ),
        mime="text/markdown",
    )