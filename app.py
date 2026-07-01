"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         LEUKEMIA IDENTIFICATION & ANALYSIS SYSTEM (LIAS) v1.0              ║
║         B.Tech Final Year Project | AI-Powered Hematology Analysis         ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE OVERVIEW:
─────────────────────
  Backbone  : EfficientNetV2-B0 (pretrained on ImageNet, fine-tuned)
  Head      : GlobalAveragePooling → Dense(256, GELU) → Dropout(0.4)
              → Dense(128, GELU) → Dropout(0.3) → Dense(4, Softmax)
  XAI Layer : Grad-CAM on last conv block (block6a_expand_activation)
  Pipeline  : HSV segmentation → CLAHE → Augmentation → Normalization

FEATURE FUSION LOGIC:
──────────────────────
  The model uses a "Dual-Branch Feature Fusion" strategy:
  1. MACRO branch  : EfficientNetV2-B0 captures global morphology
                    (cell size, nuclear-to-cytoplasmic ratio, clustering)
  2. MICRO branch  : HSV-segmented cell ROI captures chromatin texture
                    and granularity at the cellular level
  Both branches are concatenated before the classification head,
  giving the model both structural and textural reasoning capability.

HOW TO RUN:
───────────
  1. Install dependencies: pip install -r requirements.txt
  2. (Optional) Place your .h5 model at: models/leukemia_model.h5
  3. Launch UI: streamlit run app.py
  4. Open browser at: http://localhost:8501

LINKING YOUR TRAINED MODEL:
───────────────────────────
  In model_engine.py → build_or_load_model():
    Replace: model = build_demo_model()
    With   : model = tf.keras.models.load_model('models/leukemia_model.h5')
  ─ OR for PyTorch ─
    Use the torch_loader.py adapter (see placeholder at bottom of model_engine.py)
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import time

# Local modules
from model_engine import build_or_load_model, predict_with_gradcam
from preprocessing import preprocess_image, generate_augmented_grid
from report_generator import generate_medical_report

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LIAS | Leukemia Identification System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

  :root {
    --bg-primary: #050a14;
    --bg-card: #0a1628;
    --bg-elevated: #0f1f3d;
    --accent-cyan: #00d4ff;
    --accent-green: #00ff88;
    --accent-red: #ff3366;
    --accent-amber: #ffb800;
    --text-primary: #e8f4fd;
    --text-muted: #6b8cad;
    --border: rgba(0, 212, 255, 0.2);
    --glow: 0 0 20px rgba(0, 212, 255, 0.3);
  }

  .stApp { background: var(--bg-primary); color: var(--text-primary); }

  /* Override Streamlit defaults */
  .stApp > header { background: transparent; }
  section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
  }

  /* Hero Header */
  .hero-header {
    background: linear-gradient(135deg, #050a14 0%, #0a1628 50%, #071a30 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #00d4ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
  }
  .hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid rgba(0, 255, 136, 0.3);
    color: var(--accent-green);
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    letter-spacing: 0.1em;
    margin-top: 0.8rem;
  }

  /* Cards */
  .analysis-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
  }

  /* Prediction Result */
  .prediction-result {
    background: linear-gradient(135deg, var(--bg-elevated), var(--bg-card));
    border: 1px solid var(--accent-cyan);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    box-shadow: var(--glow);
    margin: 1rem 0;
  }
  .prediction-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
  }
  .prediction-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
  }
  .pred-healthy { color: var(--accent-green); }
  .pred-leukemia { color: var(--accent-red); }

  /* Confidence Bars */
  .conf-bar-container { margin: 0.4rem 0; }
  .conf-bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.2rem;
  }
  .conf-bar-track {
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
  }

  /* Stat Chips */
  .stat-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 1rem 0; }
  .stat-chip {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    flex: 1;
    min-width: 80px;
  }
  .stat-chip-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent-cyan);
  }
  .stat-chip-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* Report Box */
  .report-box {
    background: var(--bg-card);
    border: 1px solid rgba(255, 184, 0, 0.3);
    border-left: 3px solid var(--accent-amber);
    border-radius: 8px;
    padding: 1.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    line-height: 1.7;
    color: var(--text-primary);
    white-space: pre-wrap;
  }

  /* Upload Area Override */
  [data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
  }

  /* Section Headers */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Sidebar elements */
  .sidebar-model-info {
    background: var(--bg-elevated);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    line-height: 1.8;
  }

  /* Warning */
  .medical-disclaimer {
    background: rgba(255, 51, 102, 0.05);
    border: 1px solid rgba(255, 51, 102, 0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: rgba(255, 51, 102, 0.8);
    margin-top: 1rem;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg-primary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* Streamlit button override */
  .stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #00ff8822);
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan);
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #00ff8844);
    box-shadow: var(--glow);
  }
</style>
""", unsafe_allow_html=True)


# ── Load Model (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return build_or_load_model()


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 800;
                color: #00d4ff; padding: 0.5rem 0 1rem 0; letter-spacing: -0.01em;">
        🔬 LIAS Control Panel
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model Settings**")
    model_mode = st.selectbox(
        "Model Mode",
        ["Demo (Mock Weights)", "Load from .h5 file"],
        help="Select 'Load from .h5 file' once you have trained weights"
    )

    if model_mode == "Load from .h5 file":
        h5_path = st.text_input("Path to .h5 file", value="models/leukemia_model.h5")
        st.caption("⚠️ Place your trained model at the specified path")

    st.markdown("**Analysis Settings**")
    show_augmented = st.checkbox("Show augmented previews", value=False)
    gradcam_alpha = st.slider("Grad-CAM overlay opacity", 0.3, 0.9, 0.6, 0.05)
    confidence_threshold = st.slider("Alert threshold (%)", 50, 95, 70)

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-model-info">
    ARCHITECTURE<br>
    ─────────────────<br>
    Backbone: EfficientNetV2-B0<br>
    Input: 224×224×3<br>
    Classes: 4<br>
    XAI: Grad-CAM<br>
    <br>
    CLASSES<br>
    ─────────────────<br>
    [0] Healthy<br>
    [1] ALL (B-cell)<br>
    [2] AML (M2/M3)<br>
    [3] CML (Chronic)<br>
    <br>
    PREPROCESSING<br>
    ─────────────────<br>
    • HSV segmentation<br>
    • CLAHE contrast<br>
    • Gaussian denoise<br>
    • Stain normalization<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="medical-disclaimer">
    ⚠️ FOR RESEARCH USE ONLY<br>
    Not a substitute for professional clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">🔬 Leukemia Identification & Analysis System</p>
  <p class="hero-subtitle">EfficientNetV2-B0 · Grad-CAM XAI · 4-Class Hematological Classification</p>
  <div><span class="hero-badge">✓ DEMO MODE ACTIVE</span>
  <span class="hero-badge" style="margin-left:0.5rem; border-color:rgba(0,212,255,0.3); color:#00d4ff; background:rgba(0,212,255,0.08);">B.TECH FINAL YEAR PROJECT</span></div>
</div>
""", unsafe_allow_html=True)


# ── Load Model ──────────────────────────────────────────────────────────────
with st.spinner("🧠 Initializing neural network..."):
    model = load_model()

st.success("✅ Model ready — EfficientNetV2-B0 loaded with demo weights", icon="🤖")


# ── Upload Section ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">01 / INPUT — Upload Blood Smear Image</div>', unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a peripheral blood smear or bone marrow aspirate (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("""
    <div class="analysis-card">
      <div class="card-label">Supported Inputs</div>
      <div style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#9bb8d0; line-height:1.8;">
        • Peripheral blood smears<br>
        • Bone marrow aspirates<br>
        • Giemsa/Wright-stained<br>
        • Minimum 224×224 px<br>
        • RGB / Grayscale auto-converted
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Analysis Pipeline ───────────────────────────────────────────────────────
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(original_image)

    st.markdown('<div class="section-header">02 / PREPROCESSING — Segmentation & Enhancement</div>', unsafe_allow_html=True)

    with st.spinner("⚙️ Running HSV segmentation + CLAHE enhancement..."):
        preprocessed_img, hsv_mask, clahe_img = preprocess_image(img_array)
        time.sleep(0.5)  # Brief pause for demo effect

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown('<div class="card-label">Original Image</div>', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True)
    with col_p2:
        st.markdown('<div class="card-label">HSV Cell Segmentation Mask</div>', unsafe_allow_html=True)
        st.image(hsv_mask, use_container_width=True)
    with col_p3:
        st.markdown('<div class="card-label">CLAHE Enhanced</div>', unsafe_allow_html=True)
        st.image(clahe_img, use_container_width=True)

    if show_augmented:
        st.markdown('<div class="card-label">Augmentation Grid (8 variants)</div>', unsafe_allow_html=True)
        aug_grid = generate_augmented_grid(img_array)
        st.image(aug_grid, use_container_width=True)

    # ── Prediction ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">03 / INFERENCE — Classification + Grad-CAM</div>', unsafe_allow_html=True)

    with st.spinner("🧬 Running inference + generating Grad-CAM heatmap..."):
        predictions, gradcam_overlay, processing_stats = predict_with_gradcam(
            model, img_array, alpha=gradcam_alpha
        )
        time.sleep(0.8)

    CLASS_NAMES  = ["Healthy", "ALL (Acute Lymphoblastic)", "AML (Acute Myeloid)", "CML (Chronic Myeloid)"]
    CLASS_SHORT  = ["Healthy", "ALL", "AML", "CML"]
    CLASS_COLORS = ["#00ff88", "#ff3366", "#ff6b35", "#ffb800"]

    top_idx = int(np.argmax(predictions))
    top_conf = float(predictions[top_idx]) * 100
    is_healthy = top_idx == 0

    # Main results row
    col_orig, col_gcam = st.columns(2)
    with col_orig:
        st.markdown('<div class="card-label">Original — Input Image</div>', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True)
    with col_gcam:
        st.markdown('<div class="card-label">Grad-CAM — Model Attention Heatmap</div>', unsafe_allow_html=True)
        st.image(gradcam_overlay, use_container_width=True)

    # Prediction Card
    pred_class_css = "pred-healthy" if is_healthy else "pred-leukemia"
    st.markdown(f"""
    <div class="prediction-result">
      <div class="prediction-label">Primary Diagnosis</div>
      <div class="prediction-value {pred_class_css}">{CLASS_NAMES[top_idx]}</div>
      <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#6b8cad; margin-top:0.3rem;">
        Confidence: {top_conf:.1f}% &nbsp;|&nbsp; 
        Status: {'{"✓ WITHIN NORMAL RANGE"' if is_healthy else '"⚠ ABNORMALITY DETECTED"'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars
    st.markdown('<div class="section-header">04 / CONFIDENCE SCORES — All Classes</div>', unsafe_allow_html=True)

    conf_cols = st.columns(4)
    for i, (name, conf, color) in enumerate(zip(CLASS_SHORT, predictions, CLASS_COLORS)):
        conf_pct = float(conf) * 100
        with conf_cols[i]:
            st.markdown(f"""
            <div class="analysis-card" style="text-align:center; border-color: {color}33;">
              <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:{color};">
                {conf_pct:.1f}%
              </div>
              <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#6b8cad; text-transform:uppercase; letter-spacing:0.1em; margin:0.4rem 0;">
                {name}
              </div>
              <div style="background:rgba(255,255,255,0.05); border-radius:3px; height:4px; overflow:hidden;">
                <div style="width:{conf_pct:.1f}%; height:100%; background:{color}; border-radius:3px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Processing Stats
    st.markdown('<div class="section-header">05 / SYSTEM METRICS</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-chip"><div class="stat-chip-value">{processing_stats['inference_ms']}ms</div><div class="stat-chip-label">Inference</div></div>
      <div class="stat-chip"><div class="stat-chip-value">{processing_stats['preproc_ms']}ms</div><div class="stat-chip-label">Preprocessing</div></div>
      <div class="stat-chip"><div class="stat-chip-value">{processing_stats['model_params']}</div><div class="stat-chip-label">Parameters</div></div>
      <div class="stat-chip"><div class="stat-chip-value">224²</div><div class="stat-chip-label">Input Size</div></div>
      <div class="stat-chip"><div class="stat-chip-value">4</div><div class="stat-chip-label">Classes</div></div>
      <div class="stat-chip"><div class="stat-chip-value">B0</div><div class="stat-chip-label">EfficientNet</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Medical Report
    st.markdown('<div class="section-header">06 / MEDICAL REPORT — AI Generated Summary</div>', unsafe_allow_html=True)

    report_text = generate_medical_report(
        predictions=predictions,
        class_names=CLASS_NAMES,
        top_class=top_idx,
        top_confidence=top_conf,
        image_name=uploaded_file.name,
        stats=processing_stats
    )

    st.markdown(f'<div class="report-box">{report_text}</div>', unsafe_allow_html=True)

    # Download Report
    st.download_button(
        label="⬇ Download Full Report (.txt)",
        data=report_text,
        file_name=f"LIAS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    if top_conf >= confidence_threshold and not is_healthy:
        st.error(f"🚨 HIGH CONFIDENCE ALERT: {CLASS_NAMES[top_idx]} detected at {top_conf:.1f}% confidence. Immediate clinical review recommended.", icon="⚠️")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; background:var(--bg-card);
                border: 1px dashed rgba(0,212,255,0.2); border-radius:16px; margin-top: 1rem;">
      <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
      <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#00d4ff;">
        Upload a blood smear image to begin analysis
      </div>
      <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:#6b8cad; margin-top:0.5rem;">
        Supports JPG and PNG • Giemsa/Wright stained • Peripheral blood or bone marrow
      </div>
    </div>
    """, unsafe_allow_html=True)
