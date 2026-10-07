# CyberAgent Incident Investigation Report

**Sample:** 2
**Generated:** 2026-10-07 12:46:55

---

## 1. Executive Summary

**Investigation Verdict:** Potential Exploits activity requires further investigation.

**Threat Classification:** Exploits

**Severity:** Medium (50.95/100)

**Agent Confidence:** Very Low (34.89/100)

## 2. Detection Assessment

- Binary prediction: **Attack**
- Attack probability: **46.47%**
- Random Forest prediction: **Exploits**
- Random Forest confidence: **18.02%**
- Two-Stream prediction: **Exploits**
- Two-Stream confidence: **45.09%**

## 3. Anomaly Assessment

- Isolation Forest anomaly: **No**
- Anomaly score: **0.137507**

## 4. Model Consistency

- Classifier disagreement: **No**
- Evidence strength: **Low (55/100)**

## 5. Key Findings

- Binary detection classified the traffic as attack activity with 46.47% attack probability.
- Random Forest predicted Exploits with 18.02% confidence.
- Two-Stream model predicted Exploits with 45.09% confidence.
- The supervised classifiers agree on the predicted threat category.
- Isolation Forest did not flag the sample as anomalous.
- Overall evidence strength is Low (55/100).

## 6. Investigation Reasoning

- The binary detector provides evidence that the traffic may represent malicious activity.
- Random Forest and the Two-Stream classifier produce the same threat category, increasing classification consistency.
- Isolation Forest does not independently support the presence of an anomaly.
- The overall evidence strength is low, so the investigation result should be treated as preliminary.

## 7. MITRE ATT&CK Candidate Mappings

- No candidate MITRE ATT&CK mappings.

> ATT&CK mappings are candidate mappings based on flow-level evidence and require corroboration with additional telemetry before being treated as confirmed techniques.

## 8. Recommended Actions

- Review the event and collect additional evidence before escalation.
- Check affected services and correlate with host or application logs for exploitation evidence.

## 9. Investigation Limitations

- The investigation agent operates on flow-level network features and does not have direct host, process, authentication, or packet-payload telemetry.
- Isolation Forest anomaly detection indicates deviation from learned normal traffic and does not by itself prove malicious activity.
- SHAP and LIME describe model behavior and feature contribution; they do not establish causal relationships.
- MITRE ATT&CK mappings are candidate mappings and require corroborating telemetry before being treated as confirmed techniques.
- Agent confidence represents evidence strength and consistency, not attack probability.

---

**CyberAgent:** Agent-Assisted Network Threat Detection and Investigation using Machine Learning