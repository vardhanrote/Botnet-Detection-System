"""
CyberAgent Investigation Package

Contains the investigation-layer components
used after ML-based threat detection.

Components:
- Evidence extraction
- Local SHAP explanation
- Local LIME explanation
- Threat profiling
- Severity assessment
"""

__version__ = "1.1.0"

__all__ = [
    "evidence_extractor",
    "local_shap",
    "local_lime",
    "threat_profiler",
    "severity_engine",
]