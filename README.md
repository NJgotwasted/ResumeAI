# ResumeAI P30 - Intelligent Resume Parser & NER Architecture

A modular, privacy-preserving resume parsing and Named Entity Recognition (NER) platform built with Streamlit, PyMuPDF, Google GenAI SDK, and HuggingFace Transformers.

---

## 🌟 Key Features

1. **High-Fidelity PDF Text Extraction**:
   - Uses **PyMuPDF (`fitz`)** for accurate, structural extraction of text from multi-page PDFs.
   - Comprehensive error handling for corrupted files, encrypted/password-protected PDFs, and scanned documents.

2. **Privacy-Preserving PII Shield**:
   - Automatically detects and redacts personal contact info (**Email addresses** and **Telephone numbers**) before sending text to external LLMs.
   - Maintains an interactive privacy audit log of redacted items.

3. **Structured Entity Extraction with Gemini**:
   - Powered by the official `google-genai` SDK.
   - Strictly enforces structured JSON return containing:
     - `PERSON`: Full candidate and reference names.
     - `SKILL`: Technical skills, tools, programming languages, and competencies.
     - `DEGREE`: Degrees, diplomas, majors, and certifications.
     - `COMPANY`: Employers, organizations, and clients worked for.
   - Model configurable via `GEMINI_MODEL` (default: `gemini-2.5-flash`).

4. **Phase 2 BERT NER Ready**:
   - Modular architecture (`src/ner_model.py`) structured with standard BIO tags (`B-PERSON`, `I-PERSON`, `B-SKILL`, etc.).
   - Pre-configured subword token alignment, HuggingFace dataset mapping, and Trainer callbacks.
   - *Note: In this phase, BERT has not been fine-tuned. Entity extraction is powered by Gemini, keeping the BERT pipeline ready for Phase 2 dataset training without simulating model results.*

5. **Evaluation Suite**:
   - Entity-level set metrics (Precision, Recall, F1-Score, Jaccard Similarity).
   - Token-level `seqeval` evaluation callback ready for BERT training.

6. **Modern Streamlit Dashboard**:
   - PDF drag-and-drop uploader.
   - Raw vs. privacy-masked text inspector.
   - Metric overview cards and interactive entity badges.
   - Real-time Gemini API connection diagnostics.
   - Structured JSON export.

---

## 📁 Project Architecture

```
ResumeAI/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment variables template
├── .gitignore                  # Git ignore rules (secrets, venvs, caches)
├── README.md                   # Documentation & setup guide
│
└── src/
    ├── __init__.py             # Module initialization
    ├── pdf_extractor.py        # PyMuPDF binary & file extraction engine
    ├── privacy.py              # PII detection and regex masking engine
    ├── gemini_extractor.py     # Google GenAI structured JSON extraction client
    ├── ner_model.py            # BERT Token Classification architecture (Phase 2 ready)
    └── evaluator.py            # Set & seqeval metrics for extraction & NER
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Operating System**: Windows (or macOS / Linux)
- **Python**: Version **3.12** recommended

### 2. Virtual Environment Setup

Create and activate a Python 3.12 virtual environment:

**PowerShell (Windows):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Command Prompt (cmd):**
```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```powershell
Copy-Item .env.example .env
```

Edit `.env` to add your Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

*(Note: The application also supports pasting your API key directly into the Streamlit sidebar during runtime if you prefer not to edit `.env`.)*

### 5. Launch the Application

```powershell
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🔒 Privacy & Safety Guarantee

ResumeAI is designed with data privacy in mind:
- Direct candidate phone numbers and email addresses are masked prior to external network calls (`[EMAIL_REDACTED]`, `[PHONE_REDACTED]`).
- Sensitive `.env` and `APIKEY.env` files are excluded by `.gitignore`.

---

## 🧪 Testing Extraction Manually

You can test PDF extraction directly from Python:

```python
from src.pdf_extractor import extract_pdf_text
from src.privacy import mask_resume_text

# 1. Extract text from PDF
result = extract_pdf_text("path/to/resume.pdf")
print(f"Extracted {result['word_count']} words across {result['page_count']} pages.")

# 2. Mask PII
masked_text, meta = mask_resume_text(result["text"])
print(f"Masked {meta['emails_count']} emails and {meta['phones_count']} phone numbers.")
```

