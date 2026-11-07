"""
app.py - ResumeAI P30: Intelligent Resume Parsing & NER Application.

Phase 2 Dashboard:
- PDF upload + PyMuPDF text extraction
- Privacy-first PII masking (emails, phone numbers)
- Gemini API structured extraction (PERSON, SKILL, DEGREE, COMPANY)
- Fine-Tuned DistilBERT NER extraction (synthetic training data)
- Side-by-side BERT vs Gemini comparison
- Training metrics and loss chart
- Responsible AI disclosures
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from src.evaluator import compare_bert_vs_gemini, compute_set_metrics
from src.gemini_extractor import (
    GeminiExtractionError,
    GeminiExtractor,
    is_placeholder_key,
    load_api_credentials,
)
from src.ner_model import ENTITY_LABELS, ResumeNERArchitecture
from src.pdf_extractor import PDFExtractionError, PDFExtractor
from src.privacy import PrivacyMasker

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeAI P30 - NER & Entity Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ResumeAI — modern AI dashboard theme */

.stApp {
    background: linear-gradient(145deg, #0B1020 0%, #111827 55%, #15132B 100%);
    color: #F8FAFC;
}
<style>
/* ResumeAI — modern AI dashboard theme */

.stApp {
    background: linear-gradient(145deg, #0B1020 0%, #111827 55%, #15132B 100%);
    color: #F8FAFC;
}

/* Main content */
.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3, h4 {
    color: #F8FAFC !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown {
    color: #CBD5E1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #10172A;
    border-right: 1px solid #29334B;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #E9D5FF !important;
}

/* Main header */
.main-header {
    font-size: 2.8rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1.5px;
    margin-bottom: 0.3rem;
}

.sub-header {
    color: #A5B4FC;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1B2540, #171D33);
    border: 1px solid #354365;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.18);
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #8B5CF6;
    box-shadow: 0 12px 30px rgba(139,92,246,0.15);
}

.metric-title {
    color: #A5B4FC;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #FFFFFF;
    font-size: 2rem;
    font-weight: 800;
}

/* Entity badges */
.skill-badge {
    display: inline-block;
    background: #29204D;
    color: #C4B5FD;
    border: 1px solid #7C3AED;
    border-radius: 20px;
    padding: 6px 13px;
    margin: 4px;
    font-size: 0.88rem;
}

.company-badge {
    display: inline-block;
    background: #103A3B;
    color: #99F6E4;
    border: 1px solid #0D9488;
    border-radius: 9px;
    padding: 6px 13px;
    margin: 4px;
}

.degree-badge {
    display: inline-block;
    background: #422C18;
    color: #FDE68A;
    border: 1px solid #B7791F;
    border-radius: 9px;
    padding: 6px 13px;
    margin: 4px;
}

.person-badge {
    display: inline-block;
    background: #172E50;
    color: #BFDBFE;
    border: 1px solid #3B82F6;
    border-radius: 20px;
    padding: 6px 13px;
    margin: 4px;
}

/* Model result cards */
.bert-card, .gemini-card {
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.25s ease;
}

.bert-card {
    background: linear-gradient(135deg, #172D50, #17213A);
    border: 1px solid #3B82F6;
}

.gemini-card {
    background: linear-gradient(135deg, #103A36, #172D32);
    border: 1px solid #14B8A6;
}

.section-header {
    color: #E9D5FF;
    font-size: 1.15rem;
    font-weight: 750;
    border-bottom: 1px solid #354365;
    padding-bottom: 10px;
    margin-bottom: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(110deg, #7C3AED, #4F46E5);
    color: white !important;
    border: none;
    border-radius: 11px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(110deg, #8B5CF6, #6366F1);
    box-shadow: 0 5px 20px rgba(124,58,237,0.35);
    transform: translateY(-2px);
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #151F35;
    border: 1px dashed #6474A3;
    border-radius: 15px;
    padding: 15px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid #354365;
}

.stTabs [data-baseweb="tab"] {
    background: #19243B;
    color: #CBD5E1;
    border-radius: 9px 9px 0 0;
    padding: 10px 16px;
}

.stTabs [aria-selected="true"] {
    background: #33245C !important;
    color: #DDD6FE !important;
}

/* Inputs */
.stTextInput input, .stTextArea textarea,
.stSelectbox [data-baseweb="select"] {
    background: #17213A;
    color: #FFFFFF;
    border: 1px solid #354365;
    border-radius: 9px;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #151F35;
    border: 1px solid #354365;
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #354365;
}

/* Progress indicators */
.stProgress > div > div {
    background: linear-gradient(90deg, #7C3AED, #22D3EE);
}

/* Responsive */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>


""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def initialize_session_state():
    defaults = {
        "extracted_text_data": None,
        "gemini_results": None,
        "bert_results": None,
        "connection_status": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(ner_model: ResumeNERArchitecture):
    st.sidebar.title("⚙️ Configuration")

    env_key, env_model = load_api_credentials()
    initial_key = "" if is_placeholder_key(env_key) else (env_key or "")

    st.sidebar.markdown("### 🔑 Gemini Credentials")
    user_api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=initial_key,
        type="password",
        placeholder="AIzaSy...",
        help="Reads automatically from .env. Paste key here to override.",
    )
    if is_placeholder_key(env_key):
        st.sidebar.caption("⚠️ `.env` has the default placeholder. Paste your real key above.")

    available_models = [
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    current_model = env_model if env_model in available_models else "gemini-3.5-flash"
    selected_model = st.sidebar.selectbox(
        "Gemini Model",
        options=available_models,
        index=available_models.index(current_model) if current_model in available_models else 0,
        help="Model specified in GEMINI_MODEL or selected here.",
    )

    if st.sidebar.button("🔍 Check Gemini Connection", width="stretch"):
        if not user_api_key.strip():
            st.sidebar.error("❌ API key is empty.")
            st.session_state.connection_status = False
        else:
            with st.sidebar.status("Testing Gemini connection...", expanded=True) as status:
                tester = GeminiExtractor(api_key=user_api_key.strip(), model_name=selected_model)
                is_ok, msg = tester.test_connection()
                if is_ok:
                    status.update(label="✅ Connected!", state="complete")
                    st.sidebar.success(f"**Status:** {msg}")
                    st.session_state.connection_status = True
                else:
                    status.update(label="❌ Failed", state="error")
                    st.sidebar.error(f"**Error:** {msg}")
                    st.session_state.connection_status = False

    # Privacy
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Privacy Shield")
    mask_pii = st.sidebar.checkbox(
        "Mask PII before LLM calls",
        value=True,
        help="Redacts emails and phone numbers before sending to Gemini.",
    )

    # Extraction Engine
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Extraction Engine")
    engine_choice = st.sidebar.radio(
        "Run extraction with:",
        options=["Gemini Only", "BERT Only", "Both (Compare)"],
        index=0,
    )

    # BERT Status card in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 BERT Model Status")
    if ner_model.is_fine_tuned:
        st.sidebar.success("✅ Fine-tuned checkpoint loaded")
        meta = ner_model.training_metadata
        st.sidebar.markdown(
            f"**Base:** `{meta['base_model']}`  \n"
            f"**Epochs:** {meta['epochs']}  \n"
            f"**Val F1:** `{meta['eval_f1']}`  \n"
            f"**Precision:** `{meta['eval_precision']}`  \n"
            f"**Recall:** `{meta['eval_recall']}`  \n"
            f"**Device:** {meta['device']}  \n"
            f"**Data:** {meta['training_data']}"
        )
    else:
        st.sidebar.warning("⚠️ BERT checkpoint not found")
        st.sidebar.caption(
            "Run `python training/train_bert.py` to generate the model checkpoint."
        )

    return user_api_key, selected_model, mask_pii, engine_choice


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_entity_badges(entities: Dict[str, List[str]]):
    # Debug information
    st.write("DEBUG - Entity keys:", list(entities.keys()))
    st.write("DEBUG - Company entities:", entities.get("COMPANY", []))

    persons = entities.get("PERSON", [])
    skills = entities.get("SKILL", [])
    degrees = entities.get("DEGREE", [])
    companies = entities.get("COMPANY", [])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">👤 Persons</div>'
            f'<div class="metric-value">{len(persons)}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">🛠️ Skills</div>'
            f'<div class="metric-value">{len(skills)}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">🎓 Degrees</div>'
            f'<div class="metric-value">{len(degrees)}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">🏢 Companies</div>'
            f'<div class="metric-value">{len(companies)}</div></div>',
            unsafe_allow_html=True,
        )

    t1, t2, t3, t4 = st.tabs(["🛠️ Skills", "🎓 Education", "🏢 Companies", "👤 Persons"])
    with t1:
        if skills:
            st.markdown(
                " ".join(f'<span class="skill-badge">{s}</span>' for s in skills),
                unsafe_allow_html=True,
            )
        else:
            st.info("No skills extracted.")
    with t2:
        if degrees:
            for d in degrees:
                st.markdown(f"- 🎓 **{d}**")
        else:
            st.info("No degrees extracted.")
    with t3:
        if companies:
            for c in companies:
                st.markdown(f"- 🏢 **{c}**")
        else:
            st.info("No companies extracted.")
    with t4:
        if persons:
            for p in persons:
                st.markdown(f"- 👤 **{p}**")
        else:
            st.info("No person names extracted.")


def render_metrics_table(bert_entities: Optional[Dict], gemini_entities: Optional[Dict]):
    """Side-by-side comparison table when both results are available."""
    if bert_entities is None or gemini_entities is None:
        return

    comparison = compare_bert_vs_gemini(bert_entities, gemini_entities)
    categories = ["PERSON", "SKILL", "DEGREE", "COMPANY"]

    st.markdown("#### 🔍 Entity Overlap Analysis")
    cols = st.columns(4)
    for i, cat in enumerate(categories):
        overlap_data = comparison["overlap"].get(cat, {})
        count = overlap_data.get("count", 0)
        with cols[i]:
            st.metric(label=f"{cat} in common", value=count)

    st.markdown("#### 📊 BERT-only vs Gemini-only Entities")
    b_col, g_col = st.columns(2)
    with b_col:
        st.markdown("**🤖 BERT-only extractions:**")
        for cat in categories:
            items = comparison["bert_only"].get(cat, [])
            if items:
                st.markdown(f"*{cat}:* {', '.join(items)}")
    with g_col:
        st.markdown("**✨ Gemini-only extractions:**")
        for cat in categories:
            items = comparison["gemini_only"].get(cat, [])
            if items:
                st.markdown(f"*{cat}:* {', '.join(items)}")


def render_training_chart():
    """Renders training + validation loss curve from training_history.json."""
    history_path = (
        Path(__file__).parent / "models" / "bert_ner" / "training_history.json"
    )
    if not history_path.exists():
        st.info("Training history not found. Train the model first.")
        return

    with open(history_path) as f:
        history = json.load(f)

    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    val_f1s = []

    for entry in history.get("training_loss_history", []):
        step = entry.get("step", 0)
        if "train_loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["train_loss"])
        if "eval_loss" in entry:
            val_steps.append(step)
            val_losses.append(entry["eval_loss"])
        if "eval_f1" in entry:
            val_f1s.append({"step": step, "Val F1": entry["eval_f1"]})

    import pandas as pd

    if train_losses:
        df_train = pd.DataFrame({"Step": train_steps, "Train Loss": train_losses}).set_index("Step")
        st.markdown("**Training Loss**")
        st.line_chart(df_train)

    if val_losses:
        df_val = pd.DataFrame({"Step": val_steps, "Val Loss": val_losses}).set_index("Step")
        st.markdown("**Validation Loss**")
        st.line_chart(df_val)

    if val_f1s:
        df_f1 = pd.DataFrame(val_f1s).set_index("step")
        st.markdown("**Validation F1 Score (per epoch)**")
        st.line_chart(df_f1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    initialize_session_state()

    # Load BERT model once and cache via session state
    if "ner_model" not in st.session_state:
        with st.spinner("Loading DistilBERT NER model..."):
            st.session_state.ner_model = ResumeNERArchitecture()

    ner_model: ResumeNERArchitecture = st.session_state.ner_model
    user_api_key, selected_model, mask_pii, engine_choice = render_sidebar(ner_model)

    
    # Premium ResumeAI Hero Header
    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #1E1B4B, #172554, #164E63);
        border: 1px solid #4F46E5;
        border-radius: 20px;
        padding: 30px 32px;
        margin-bottom: 28px;
        box-shadow: 0 12px 35px rgba(79,70,229,0.14);
    ">
        <div style="color:#A5B4FC;font-size:12px;font-weight:700;
                    letter-spacing:2px;margin-bottom:12px;">
            AI-POWERED DOCUMENT INTELLIGENCE
        </div>
        <div style="color:white;font-size:38px;font-weight:800;">
            ResumeAI <span style="color:#A78BFA;">P30</span>
        </div>
        <div style="color:#CBD5E1;font-size:15px;margin-top:10px;">
            Intelligent Resume Parsing · Named Entity Recognition · Privacy-First AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section 1 — Upload
    st.markdown("### 1. Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose a candidate resume PDF",
        type=["pdf"],
        help="Upload a PDF resume. PyMuPDF extracts real text layout.",
    )

    if uploaded_file is not None:
        try:
            pdf_bytes = uploaded_file.getvalue()
            extractor = PDFExtractor()
            text_data = extractor.extract_from_bytes(
                pdf_bytes, filename=uploaded_file.name
            )
            st.session_state.extracted_text_data = text_data
            st.session_state.gemini_results = None
            st.session_state.bert_results = None
            st.success(f"Loaded {uploaded_file.name} using PyMuPDF.")
        except PDFExtractionError as e:
            st.error(f"PDF Extraction Error: {e}")
            st.session_state.extracted_text_data = None
        except Exception as e:
            st.error(f"Unexpected Error reading PDF: {e}")
            st.session_state.extracted_text_data = None




    # -----------------------------------------------------------------------
    # Section 2 — Text & Privacy
    # -----------------------------------------------------------------------
    text_data = st.session_state.extracted_text_data
    if text_data is not None:
        raw_text = text_data["text"]
        masker = PrivacyMasker()
        masked_text, privacy_meta = masker.mask_pii(raw_text)

        # Document metrics row
        m1, m2, m3, m4 = st.columns(4)
        for col, label, value in [
            (m1, "Pages", text_data["page_count"]),
            (m2, "Words", f"{text_data['word_count']:,}"),
            (m3, "Characters", f"{text_data['char_count']:,}"),
            (m4, "PII Redacted", privacy_meta["total_redactions"]),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">{label}</div>'
                    f'<div class="metric-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        if text_data.get("is_scanned_likely"):
            st.warning("⚠️ Low text density — this may be a scanned / image-based PDF.")

        st.markdown("### 2. Extracted Text & Privacy Inspector")
        tab_priv, tab_raw, tab_audit = st.tabs([
            "🛡️ Privacy-Masked (LLM Payload)", "📄 Raw Text", "🔍 Privacy Audit",
        ])
        with tab_priv:
            st.caption("Sanitized text sent to Gemini — PII is redacted.")
            st.text_area("Masked Text", value=masked_text if mask_pii else raw_text,
                         height=220, disabled=True, key="ta_masked")
        with tab_raw:
            st.caption("Text extracted directly by PyMuPDF.")
            st.text_area("Raw Text", value=raw_text, height=220, disabled=True, key="ta_raw")
        with tab_audit:
            ca, cb = st.columns(2)
            with ca:
                st.write(f"📧 Emails ({privacy_meta['emails_count']}):")
                for em in privacy_meta.get("detected_emails", []):
                    st.code(em)
                if not privacy_meta.get("detected_emails"):
                    st.write("_None found._")
            with cb:
                st.write(f"📞 Phones ({privacy_meta['phones_count']}):")
                for ph in privacy_meta.get("detected_phones", []):
                    st.code(ph)
                if not privacy_meta.get("detected_phones"):
                    st.write("_None found._")

        # -------------------------------------------------------------------
        # Section 3 — Entity Extraction
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 3. Entity Extraction")

        btn_label = {
            "Gemini Only": "⚡ Extract with Gemini",
            "BERT Only": "🤖 Extract with BERT",
            "Both (Compare)": "⚡🤖 Extract with Both (Gemini + BERT)",
        }.get(engine_choice, "⚡ Extract Entities")

        if st.button(btn_label, type="primary", width="stretch"):
            run_gemini = engine_choice in ("Gemini Only", "Both (Compare)")
            run_bert = engine_choice in ("BERT Only", "Both (Compare)")

            # Gemini extraction
            if run_gemini:
                if not user_api_key or not user_api_key.strip():
                    st.error("❌ Gemini API key missing. Add it in the sidebar or .env.")
                else:
                    with st.spinner(f"Extracting with {selected_model}..."):
                        try:
                            client = GeminiExtractor(
                                api_key=user_api_key.strip(),
                                model_name=selected_model,
                            )
                            result = client.extract_entities(
                                resume_text=raw_text,
                                apply_privacy_masking=mask_pii,
                            )
                            st.session_state.gemini_results = result
                            st.success(f"✅ Gemini extraction done using **{result['model_used']}**.")
                        except GeminiExtractionError as e:
                            st.error(f"❌ Gemini Error: {e}")
                        except Exception as e:
                            st.error(f"❌ Unexpected Gemini failure: {e}")

            # BERT extraction
            if run_bert:
                if not ner_model.is_fine_tuned:
                    st.warning(
                        "⚠️ BERT checkpoint not loaded. Run `python training/train_bert.py` first."
                    )
                else:
                    with st.spinner("Running DistilBERT NER inference..."):
                        bert_pred = ner_model.predict(raw_text)
                        st.session_state.bert_results = bert_pred
                        st.success(f"✅ BERT extraction done. {bert_pred.get('status', '')}")

    # -----------------------------------------------------------------------
    # Section 4 — Display Results
    # -----------------------------------------------------------------------
    gemini_results = st.session_state.gemini_results
    bert_results = st.session_state.bert_results

    if gemini_results is not None or bert_results is not None:
        st.markdown("---")

        # Tabs: Gemini | BERT | Compare | Charts | Export
        result_tabs = st.tabs([
            "✨ Gemini Extraction",
            "🤖 BERT Extraction",
            "📊 Side-by-Side Comparison",
            "📈 Training Charts",
            "📥 Export",
        ])

        # --- Gemini tab ---
        with result_tabs[0]:
            if gemini_results is None:
                st.info("Run extraction with Gemini to see results here.")
            else:
                st.markdown(
                    f'<div class="gemini-card">'
                    f'<div class="section-header">✨ Gemini Extraction — {gemini_results.get("model_used","")}</div>'
                    f'Zero-shot structured extraction via Gemini API.</div>',
                    unsafe_allow_html=True,
                )
                render_entity_badges(gemini_results.get("entities", {}))

        # --- BERT tab ---
        with result_tabs[1]:
            if bert_results is None:
                st.info("Run extraction with BERT to see results here.")
            else:
                meta = ner_model.training_metadata
                st.markdown(
                    f'<div class="bert-card">'
                    f'<div class="section-header">🤖 DistilBERT NER — Fine-Tuned</div>'
                    f'Base: <code>{meta["base_model"]}</code> | '
                    f'Epochs: {meta["epochs"]} | '
                    f'Val F1: <strong>{meta["eval_f1"]}</strong> | '
                    f'<em>Trained on SYNTHETIC data only.</em></div>',
                    unsafe_allow_html=True,
                )
                if not bert_results.get("is_fine_tuned"):
                    st.warning(bert_results.get("status", "Model not loaded."))
                else:
                    render_entity_badges(bert_results.get("entities", {}))
                    if bert_results.get("notice"):
                        st.caption(f"ℹ️ {bert_results['notice']}")

        # --- Compare tab ---
        with result_tabs[2]:
            if gemini_results is None and bert_results is None:
                st.info("Run both engines to see a comparison.")
            else:
                g_ents = gemini_results.get("entities") if gemini_results else None
                b_ents = bert_results.get("entities") if bert_results else None

                if g_ents and b_ents:
                    categories = ["PERSON", "SKILL", "DEGREE", "COMPANY"]

                    # Summary comparison table
                    st.markdown("#### 📋 Extraction Count Comparison")
                    comp_data = {
                        "Category": categories,
                        "Gemini Count": [len(g_ents.get(c, [])) for c in categories],
                        "BERT Count": [len(b_ents.get(c, [])) for c in categories],
                    }
                    import pandas as pd
                    st.dataframe(pd.DataFrame(comp_data).set_index("Category"), width="stretch")

                    # Overlap analysis
                    render_metrics_table(b_ents, g_ents)

                    # Full side-by-side entity view
                    st.markdown("#### 📄 Full Entity Comparison")
                    for cat in categories:
                        with st.expander(f"{cat}", expanded=True):
                            col_b, col_g = st.columns(2)
                            with col_b:
                                st.markdown("**🤖 BERT:**")
                                items = b_ents.get(cat, [])
                                st.write(", ".join(items) if items else "_None_")
                            with col_g:
                                st.markdown("**✨ Gemini:**")
                                items = g_ents.get(cat, [])
                                st.write(", ".join(items) if items else "_None_")

                elif g_ents:
                    st.info("Only Gemini results available. Run BERT extraction for comparison.")
                    render_entity_badges(g_ents)
                elif b_ents:
                    st.info("Only BERT results available. Run Gemini extraction for comparison.")
                    render_entity_badges(b_ents)

                # Validation metrics table (from training_history.json)
                st.markdown("---")
                st.markdown("#### 🎓 BERT Validation Metrics (Training History)")
                history_path = Path(__file__).parent / "models" / "bert_ner" / "training_history.json"
                if history_path.exists():
                    with open(history_path) as f:
                        hist = json.load(f)
                    final = hist.get("eval_metrics", {})
                    if final:
                        import pandas as pd
                        rows = [
                            {"Metric": "Validation F1", "Value": final.get("eval_f1", "—")},
                            {"Metric": "Validation Precision", "Value": final.get("eval_precision", "—")},
                            {"Metric": "Validation Recall", "Value": final.get("eval_recall", "—")},
                            {"Metric": "Validation Accuracy", "Value": final.get("eval_accuracy", "—")},
                            {"Metric": "Validation Loss", "Value": final.get("eval_loss", "—")},
                        ]
                        st.dataframe(pd.DataFrame(rows).set_index("Metric"), width="stretch")
                        st.caption(
                            "⚠️ Metrics measured on SYNTHETIC validation data (15 samples from same distribution). "
                            "Not representative of real-world performance."
                        )
                else:
                    st.info("Training history not found.")

        # --- Charts tab ---
        with result_tabs[3]:
            st.markdown("### 📈 DistilBERT Training Curves")
            render_training_chart()
            st.markdown("""
            ---
            **Training Summary:**
            | Parameter | Value |
            |---|---|
            | Base Model | `distilbert-base-uncased` |
            | Epochs | 4 |
            | Training samples | 70 (synthetic) |
            | Validation samples | 15 (synthetic) |
            | Test samples | 15 (synthetic) |
            | Val F1 (epoch 4) | **0.5653** |
            | Val Precision | **0.5116** |
            | Val Recall | **0.6316** |
            | Device | CPU |
            | Training time | ~207 seconds |

            > ⚠️ **Important:** This model was trained entirely on **synthetic resume data**.
            > Performance on real resumes may differ significantly.
            > Gemini is the recommended production extraction engine.
            """)

        # --- Export tab ---
        with result_tabs[4]:
            st.markdown("### 📥 Export Structured Data")
            export_payload: Dict[str, Any] = {
                "source_file": text_data["filename"] if text_data else "resume.pdf",
            }
            if gemini_results:
                export_payload["gemini"] = {
                    "model": gemini_results.get("model_used", selected_model),
                    "entities": gemini_results.get("entities", {}),
                    "privacy": gemini_results.get("privacy", {}),
                }
            if bert_results:
                export_payload["bert"] = {
                    "model": "distilbert-base-uncased (fine-tuned)",
                    "is_fine_tuned": bert_results.get("is_fine_tuned", False),
                    "entities": bert_results.get("entities", {}),
                    "notice": bert_results.get("notice", ""),
                }

            json_str = json.dumps(export_payload, indent=2)
            d1, d2 = st.columns([1, 3])
            with d1:
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name="resume_extracted_entities.json",
                    mime="application/json",
                    width="stretch",
                )
            with st.expander("👁️ Preview JSON"):
                st.json(export_payload)

    elif text_data is None:
        st.info("👈 Upload a PDF resume to get started.")

    # -----------------------------------------------------------------------
    # Responsible AI section (always visible)
    # -----------------------------------------------------------------------
    with st.expander("⚖️ Responsible AI Disclosure", expanded=False):
        st.markdown("""
        **ResumeAI P30 — Responsible AI Notes**

        | Topic | Detail |
        |---|---|
        | Training Data | 100 **synthetic** resumes with fictional names, companies, degrees and skills |
        | Real User Data | No real resumes were used to train BERT |
        | Demographic Inference | The system **never** infers gender, caste, age, ethnicity or religion |
        | Privacy Masking | Emails and phone numbers are redacted before any LLM call |
        | Limitations | BERT (F1=0.5653) may miss entities or produce false positives on real resumes |
        | Production Path | Gemini extraction is recommended for production use |
        | Known Failures | Short resumes, heavily formatted PDFs, non-English text may degrade accuracy |
        | Data Security | No resume data is stored or logged by this application |
        """)


if __name__ == "__main__":
    main()
