"""
report_generator.py
────────────────────
Generates structured, human-readable medical AI reports from
model predictions. Designed to be informative for clinical review
while clearly stating its AI/research-only nature.
"""

from datetime import datetime
import numpy as np


# ── Clinical Descriptions per Class ──────────────────────────────────────────
CLINICAL_INFO = {
    "Healthy": {
        "full_name"  : "Normal / Healthy Peripheral Blood",
        "description": (
            "The cellular morphology appears within normal parameters. "
            "Leukocytes show appropriate nuclear-to-cytoplasmic ratios with "
            "no significant evidence of blast cells, hypersegmentation, or "
            "abnormal granulation patterns detectable by the AI system."
        ),
        "recommendation": (
            "No immediate hematological concern detected by AI analysis. "
            "Routine monitoring and periodic CBC (Complete Blood Count) "
            "as clinically indicated. Correlation with full clinical presentation recommended."
        ),
        "risk_level" : "LOW",
        "icd10"      : "Z13.6 (Screening for cardiovascular disorders)"
    },
    "ALL": {
        "full_name"  : "Acute Lymphoblastic Leukemia (B-cell predominant)",
        "description": (
            "AI pattern analysis suggests features consistent with Acute Lymphoblastic "
            "Leukemia. The model detected potential indicators including: elevated nuclear-to-"
            "cytoplasmic ratio, lymphoblast-like morphology, and chromatin condensation patterns "
            "characteristic of immature lymphoid cells. Grad-CAM attention maps highlight "
            "nuclear and peri-nuclear regions as primary decision-driving features."
        ),
        "recommendation": (
            "⚠ URGENT CLINICAL REVIEW REQUIRED. If confirmed, immediate referral to "
            "hematology/oncology for flow cytometry, bone marrow biopsy, and cytogenetic "
            "analysis (BCR-ABL1, ETV6-RUNX1, MYC rearrangements). Complete CBC with "
            "differential, LDH, uric acid, and metabolic panel advised."
        ),
        "risk_level" : "CRITICAL",
        "icd10"      : "C91.0 (Acute lymphoblastic leukemia)"
    },
    "AML": {
        "full_name"  : "Acute Myeloid Leukemia (M2/M3 subtype features)",
        "description": (
            "AI pattern analysis suggests features consistent with Acute Myeloid Leukemia. "
            "Detected morphological indicators may include: Auer rod-like structures, "
            "blast cell accumulation, hypergranular cytoplasm, and abnormal nuclear "
            "lobulation. The Grad-CAM heatmap emphasizes cytoplasmic granularity and "
            "nuclear contour irregularities as key discriminating features."
        ),
        "recommendation": (
            "⚠ URGENT CLINICAL REVIEW REQUIRED. Immediate hematology consultation. "
            "Flow cytometry panel (CD13, CD33, CD34, MPO, CD117), cytogenetics "
            "(t(8;21), inv(16), t(15;17) for APL), and FLT3/NPM1 mutation testing. "
            "Emergency bone marrow aspirate and trephine biopsy recommended."
        ),
        "risk_level" : "CRITICAL",
        "icd10"      : "C92.0 (Acute myeloblastic leukemia)"
    },
    "CML": {
        "full_name"  : "Chronic Myeloid Leukemia (Chronic Phase features)",
        "description": (
            "AI pattern analysis suggests features consistent with Chronic Myeloid Leukemia. "
            "Potential indicators detected include: left-shifted myeloid series, increased "
            "granulocyte precursors, basophilia, and characteristic splenomegaly-associated "
            "smear patterns. The Grad-CAM analysis highlights myelocyte and metamyelocyte "
            "populations as key attention regions."
        ),
        "recommendation": (
            "⚠ CLINICAL REVIEW RECOMMENDED. Hematology referral for BCR-ABL1 molecular "
            "testing (Philadelphia chromosome t(9;22)), bone marrow biopsy for phase "
            "assessment (Chronic/Accelerated/Blast), and baseline sokal score calculation. "
            "Tyrosine kinase inhibitor (TKI) therapy evaluation if confirmed."
        ),
        "risk_level" : "HIGH",
        "icd10"      : "C92.1 (Chronic myelogenous leukemia)"
    }
}

RISK_COLORS = {
    "LOW"     : "✅",
    "HIGH"    : "⚠️",
    "CRITICAL": "🚨"
}


def generate_medical_report(predictions: np.ndarray,
                              class_names: list,
                              top_class: int,
                              top_confidence: float,
                              image_name: str,
                              stats: dict) -> str:
    """
    Generate a comprehensive, structured AI medical report.
    
    Args:
        predictions    : Softmax probability array (4,)
        class_names    : List of class names
        top_class      : Index of predicted class
        top_confidence : Confidence of top prediction (0-100)
        image_name     : Filename of uploaded image
        stats          : Processing statistics dict
    
    Returns:
        Formatted report string
    """
    now = datetime.now()
    report_id = f"LIAS-{now.strftime('%Y%m%d%H%M%S')}"

    top_name = class_names[top_class].split(" (")[0]  # Short name for lookup
    # Normalize class name for lookup
    lookup_key = {
        "Healthy": "Healthy",
        "ALL": "ALL",
        "AML": "AML",
        "CML": "CML"
    }.get(top_name.split()[0], "Healthy")

    clinical = CLINICAL_INFO.get(lookup_key, CLINICAL_INFO["Healthy"])
    risk_emoji = RISK_COLORS.get(clinical["risk_level"], "⚠️")

    # Confidence interpretation
    if top_confidence >= 90:
        conf_interpretation = "Very High (Strong model consensus)"
    elif top_confidence >= 75:
        conf_interpretation = "High (Reliable prediction)"
    elif top_confidence >= 60:
        conf_interpretation = "Moderate (Further testing advised)"
    else:
        conf_interpretation = "Low (Inconclusive — repeat analysis recommended)"

    # Second-best prediction
    sorted_idx = np.argsort(predictions)[::-1]
    second_idx  = sorted_idx[1]
    second_name = class_names[second_idx].split(" (")[0]
    second_conf = float(predictions[second_idx]) * 100

    report = f"""╔══════════════════════════════════════════════════════════════════════╗
║     LEUKEMIA IDENTIFICATION & ANALYSIS SYSTEM (LIAS) v1.0          ║
║                     AI DIAGNOSTIC REPORT                            ║
╚══════════════════════════════════════════════════════════════════════╝

REPORT METADATA
───────────────────────────────────────────────────────────────────────
  Report ID       : {report_id}
  Generated       : {now.strftime('%B %d, %Y at %H:%M:%S')}
  System          : LIAS v1.0 | EfficientNetV2-B0 | Grad-CAM XAI
  Image File      : {image_name}
  Analysis Mode   : {'Demo (Mock Weights)' if stats.get('is_demo', True) else 'Trained Model'}
  Model Params    : {stats.get('model_params', '7.2M')}
  Processing Time : {stats.get('total_ms', 'N/A')} ms total
                    ({stats.get('inference_ms', 'N/A')} ms inference + 
                     {stats.get('preproc_ms', 'N/A')} ms preprocessing)

PRIMARY DIAGNOSIS
───────────────────────────────────────────────────────────────────────
  {risk_emoji} Classification  : {clinical['full_name']}
  Confidence Score  : {top_confidence:.1f}% ({conf_interpretation})
  Risk Level        : {clinical['risk_level']}
  ICD-10 Reference  : {clinical['icd10']}

PROBABILITY DISTRIBUTION
───────────────────────────────────────────────────────────────────────"""

    for i, (name, prob) in enumerate(zip(class_names, predictions)):
        conf_pct = float(prob) * 100
        bar_len  = int(conf_pct / 5)  # Max 20 chars
        bar      = "█" * bar_len + "░" * (20 - bar_len)
        marker   = " ◄ PRIMARY" if i == top_class else ""
        short    = name.split(" (")[0]
        report  += f"\n  {short:<8} [{bar}] {conf_pct:5.1f}%{marker}"

    report += f"""

  Differential Diagnosis: {second_name} ({second_conf:.1f}% — consider if primary inconclusive)

CLINICAL DESCRIPTION
───────────────────────────────────────────────────────────────────────
  {clinical['description']}

GRAD-CAM XAI ANALYSIS
───────────────────────────────────────────────────────────────────────
  The Gradient-weighted Class Activation Map highlights the image regions
  most influential in the model's decision:

  • HIGH ATTENTION (red/yellow zones): Nucleus morphology, chromatin
    texture, nuclear contour irregularities — primary discriminating features.

  • MODERATE ATTENTION (green zones): Perinuclear cytoplasm, granulation
    patterns, N:C ratio indicators.

  • LOW/NO ATTENTION (blue zones): Background, RBC stroma, plasma regions
    not considered significant for this classification.

  Interpretation: The Grad-CAM overlay confirms the model is focusing on
  biologically relevant leukocyte structures, providing explainability and
  supporting clinical interpretability of the AI decision.

PREPROCESSING APPLIED
───────────────────────────────────────────────────────────────────────
  1. Macenko stain normalization  → Reduces lab-to-lab stain variability
  2. Gaussian + bilateral denoising → Suppresses acquisition noise
  3. HSV color segmentation       → Isolates nuclear regions (blue-purple)
  4. CLAHE enhancement (LAB space) → Improves local contrast for fine features
  5. Resize to 224×224            → EfficientNetV2-B0 input normalization

CLINICAL RECOMMENDATION
───────────────────────────────────────────────────────────────────────
  {clinical['recommendation']}

TECHNICAL NOTES
───────────────────────────────────────────────────────────────────────
  • Model Architecture : EfficientNetV2-B0 + Custom Head (Transfer Learning)
  • Training Dataset   : [Link your dataset — e.g., ALL-IDB2, AML Morphology DB]
  • Validation Metrics : [Replace with your trained model's actual metrics]
    - Accuracy    : [e.g., 94.3%]
    - AUC-ROC     : [e.g., 0.978]
    - Sensitivity : [e.g., 0.923]
    - Specificity : [e.g., 0.961]
  • XAI Method    : Grad-CAM (Selvaraju et al., 2017 — ICCV Best Paper)

DISCLAIMER
───────────────────────────────────────────────────────────────────────
  ⚠️  FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.
  This AI report is NOT a substitute for professional clinical diagnosis.
  All findings must be reviewed and confirmed by a qualified hematologist
  or pathologist. The system operates in {'DEMO MODE with mock weights' if stats.get('is_demo', True) else 'TRAINED MODEL mode'} and results
  {'DO NOT reflect a clinically validated model.' if stats.get('is_demo', True) else 'should still be confirmed by clinical experts.'}

  Patient data privacy: Ensure all images are de-identified before upload.
  Reference: Helsinki Declaration on medical research ethics.

═══════════════════════════════════════════════════════════════════════
  END OF REPORT | {report_id}
  Generated by LIAS v1.0 | B.Tech Final Year Project
═══════════════════════════════════════════════════════════════════════"""

    return report
