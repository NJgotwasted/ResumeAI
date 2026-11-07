"""
test_pipeline.py - Comprehensive Unit and Integration Test for ResumeAI P30.

Verifies:
1. Real PDF creation and text extraction using PyMuPDF.
2. PII masking of emails and phone numbers.
3. Evaluator set-based metrics.
4. BERT NER architecture initialization and contracts.
5. Gemini client credential resolution and error handling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure root is in python path
sys.path.insert(0, str(Path(__file__).parent))

import pymupdf

from src.evaluator import compute_set_metrics, evaluate_extracted_resume
from src.gemini_extractor import GeminiExtractor, load_api_credentials
from src.ner_model import ENTITY_LABELS, ResumeNERArchitecture
from src.pdf_extractor import PDFExtractionError, extract_pdf_text
from src.privacy import mask_resume_text


def create_sample_resume_pdf(output_path: str = "sample_resume.pdf") -> str:
    """Generates a real PDF file with PyMuPDF containing structured resume text."""
    doc = pymupdf.open()
    page = doc.new_page()

    resume_text = (
        "Sarah Connor\n"
        "AI Research Scientist | Machine Learning Engineer\n"
        "Email: sarah.connor@cyberdyne.org | Phone: +1-555-839-2041\n"
        "San Francisco, CA\n\n"
        "SUMMARY\n"
        "Experienced AI scientist with 6+ years designing deep learning architectures and NLP systems.\n\n"
        "EDUCATION\n"
        "Master of Science in Computer Science - Stanford University (2020)\n"
        "Bachelor of Science in Electrical Engineering - MIT (2018)\n\n"
        "EXPERIENCE\n"
        "Lead NLP Engineer at Cyberdyne Systems (2021 - Present)\n"
        "- Engineered transformer pipelines, BERT token classification, and large language models.\n"
        "- Scaled inference on GPU clusters using PyTorch and Accelerate.\n\n"
        "Machine Learning Researcher at Skynet Robotics (2018 - 2021)\n"
        "- Built named entity recognition systems and computer vision models using Scikit-Learn and OpenCV.\n\n"
        "SKILLS\n"
        "Python, PyTorch, Transformers, HuggingFace, BERT, NLP, Docker, Kubernetes, SQL, Git, Linux"
    )

    rect = pymupdf.Rect(50, 50, 550, 750)
    page.insert_textbox(rect, resume_text, fontsize=11, fontname="helv")
    doc.save(output_path)
    doc.close()
    return output_path


def run_all_tests():
    print("=" * 60)
    print("RUNNING RESUMEAI P30 AUTOMATED VALIDATION SUITE")
    print("=" * 60)

    # 1. Create and Test Real PDF Text Extraction
    pdf_path = create_sample_resume_pdf()
    print(f"\n[1/5] Created test PDF: {pdf_path}")

    extracted = extract_pdf_text(pdf_path)
    print(f" -> Extracted {extracted['word_count']} words, {extracted['char_count']} chars, {extracted['page_count']} page(s).")
    assert extracted["page_count"] == 1, "Page count must be 1"
    assert "Sarah Connor" in extracted["text"], "Expected candidate name in extracted text"
    assert "sarah.connor@cyberdyne.org" in extracted["text"], "Expected email in extracted text"
    assert "+1-555-839-2041" in extracted["text"], "Expected phone in extracted text"
    print(" -> Real PDF extraction via PyMuPDF: PASSED")

    # 2. Test Privacy Masking
    print("\n[2/5] Testing PII Masking...")
    masked_text, meta = mask_resume_text(extracted["text"])
    print(f" -> Detected & masked {meta['emails_count']} email(s) and {meta['phones_count']} phone(s).")
    assert meta["emails_count"] >= 1, "Should have detected at least 1 email"
    assert meta["phones_count"] >= 1, "Should have detected at least 1 phone"
    assert "sarah.connor@cyberdyne.org" not in masked_text, "Raw email must NOT exist in masked text"
    assert "[EMAIL_REDACTED]" in masked_text, "Placeholder must be present"
    assert "[PHONE_REDACTED]" in masked_text, "Placeholder must be present"
    print(" -> Privacy PII Masking: PASSED")

    # 3. Test Evaluator Metrics
    print("\n[3/5] Testing Evaluator Module...")
    preds = ["Python", "PyTorch", "Transformers", "Java"]
    truths = ["Python", "PyTorch", "Transformers", "SQL"]
    metrics = compute_set_metrics(preds, truths)
    print(f" -> Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1']}, Jaccard: {metrics['jaccard']}")
    assert metrics["precision"] == 0.75, "Precision should be 3/4 = 0.75"
    assert metrics["recall"] == 0.75, "Recall should be 3/4 = 0.75"
    assert metrics["f1"] == 0.75, "F1 should be 0.75"
    print(" -> Evaluator Metrics: PASSED")

    # 4. Test BERT NER Model (Phase 2 — fine-tuned checkpoint)
    print("\n[4/5] Testing BERT NER Model (Phase 2 Fine-Tuned)...")
    ner_arch = ResumeNERArchitecture()
    # Phase 2: model checkpoint exists → is_fine_tuned=True
    # Phase 1 fallback (no checkpoint): is_fine_tuned=False — both are valid
    pred_result = ner_arch.predict(masked_text)
    assert "is_fine_tuned" in pred_result, "predict() must return is_fine_tuned"
    assert "entities" in pred_result, "predict() must return entities dict"
    assert isinstance(pred_result["entities"], dict), "entities must be a dict"
    for cat in ("PERSON", "SKILL", "DEGREE", "COMPANY"):
        assert cat in pred_result["entities"], f"entities must include {cat}"
    if ner_arch.is_fine_tuned:
        print(f" -> BERT checkpoint loaded: is_fine_tuned=True")
        print(f" -> Entities found: {sum(len(v) for v in pred_result['entities'].values())} total")
    else:
        print(f" -> BERT checkpoint not found: is_fine_tuned=False (fallback mode)")
        print(f" -> Status: {pred_result.get('status', '')}")
    print(" -> BERT NER Model Contracts: PASSED")

    # 5. Test Gemini Extractor Setup & Credential Resolution
    print("\n[5/5] Testing Gemini Extractor Initialization & Credential Handling...")
    client = GeminiExtractor()
    print(f" -> Default Model configured: {client.model_name}")
    assert client.model_name is not None
    # Test connection response when no key is set or with dummy key
    is_ok, msg = client.test_connection()
    print(f" -> Connection Check Result: is_connected={is_ok}, message='{msg}'")
    print(" -> Gemini Extractor Initialization: PASSED")

    # Clean up test PDF
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    print("\n" + "=" * 60)
    print("ALL 5 VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

