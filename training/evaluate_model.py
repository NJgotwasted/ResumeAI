"""
evaluate_model.py - Independent Evaluation of Fine-Tuned DistilBERT on 15-Resume Test Set.

Compares:
  1. Fine-tuned DistilBERT (models/bert_ner/) against ground truth entity labels.
  2. Gemini zero-shot extraction against the same ground truth (optional, requires API key).

IMPORTANT: All metrics are MEASURED, never fabricated. Results saved to
  models/bert_ner/test_evaluation.json

Usage:
  python training/evaluate_model.py                    # BERT only
  python training/evaluate_model.py --with-gemini      # BERT + Gemini comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_model")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_set(test_path: Path) -> List[Dict[str, Any]]:
    with open(test_path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d test resumes from %s", len(data), test_path)
    return data


def ground_truth_from_sample(sample: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract ground-truth entities from a test sample.

    Prefers the pre-built 'ground_truth_entities' field (fastest).
    Falls back to decoding 'tag_strings' (string BIO labels) if needed.
    Falls back to decoding 'ner_tags' (integer label IDs) if that is also absent.
    """
    # Option 1: pre-computed ground truth (most reliable)
    if "ground_truth_entities" in sample and sample["ground_truth_entities"]:
        gt = sample["ground_truth_entities"]
        return {
            "PERSON": list(gt.get("PERSON", [])),
            "SKILL": list(gt.get("SKILL", [])),
            "DEGREE": list(gt.get("DEGREE", [])),
            "COMPANY": list(gt.get("COMPANY", [])),
        }

    _PREFIX_MAP = {"PER": "PERSON", "SKILL": "SKILL", "DEGREE": "DEGREE", "COMPANY": "COMPANY"}
    tokens: List[str] = sample.get("tokens", [])
    gt: Dict[str, List[str]] = {"PERSON": [], "SKILL": [], "DEGREE": [], "COMPANY": []}
    current_tokens: List[str] = []
    current_type: Optional[str] = None

    def flush():
        if current_type and current_tokens:
            cat = _PREFIX_MAP.get(current_type)
            if cat:
                entity = " ".join(current_tokens)
                if entity and entity not in gt[cat]:
                    gt[cat].append(entity)

    # Option 2: tag_strings (string BIO labels)
    tag_strings = sample.get("tag_strings")
    if tag_strings:
        for token, tag in zip(tokens, tag_strings):
            if isinstance(tag, str) and tag.startswith("B-"):
                flush(); current_type = tag[2:]; current_tokens = [token]
            elif isinstance(tag, str) and tag.startswith("I-") and tag[2:] == current_type:
                current_tokens.append(token)
            else:
                flush(); current_type = None; current_tokens = []
        flush()
        return gt

    # Option 3: ner_tags (integer IDs) + labels.json
    _INT_ID2LABEL = {
        0: "O", 1: "B-PER", 2: "I-PER", 3: "B-SKILL", 4: "I-SKILL",
        5: "B-DEGREE", 6: "I-DEGREE", 7: "B-COMPANY", 8: "I-COMPANY",
    }
    ner_tags = sample.get("ner_tags", [])
    for token, tag_id in zip(tokens, ner_tags):
        tag = _INT_ID2LABEL.get(int(tag_id), "O")
        if tag.startswith("B-"):
            flush(); current_type = tag[2:]; current_tokens = [token]
        elif tag.startswith("I-") and tag[2:] == current_type:
            current_tokens.append(token)
        else:
            flush(); current_type = None; current_tokens = []
    flush()
    return gt


def text_from_sample(sample: Dict[str, Any]) -> str:
    """Reconstruct plain text: prefer 'text' field, fall back to joining tokens."""
    return sample.get("text") or " ".join(sample.get("tokens", []))


def compute_set_metrics(pred: List[str], truth: List[str]) -> Dict[str, float]:
    """Entity-level set-based Precision / Recall / F1."""
    p_set = {e.strip().lower() for e in pred if e.strip()}
    t_set = {e.strip().lower() for e in truth if e.strip()}

    if not p_set and not t_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not t_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(p_set), "fn": 0}
    if not p_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(t_set)}

    tp = len(p_set & t_set)
    fp = len(p_set - t_set)
    fn = len(t_set - p_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aggregate_metrics(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Macro-average precision/recall/F1 across all test resumes."""
    categories = ["PERSON", "SKILL", "DEGREE", "COMPANY"]
    totals: Dict[str, Dict[str, float]] = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in categories}

    for sample_result in per_sample:
        for cat in categories:
            m = sample_result.get(cat, {})
            totals[cat]["tp"] += m.get("tp", 0)
            totals[cat]["fp"] += m.get("fp", 0)
            totals[cat]["fn"] += m.get("fn", 0)

    agg: Dict[str, Any] = {}
    overall_f1_sum = 0.0
    for cat in categories:
        tp = totals[cat]["tp"]
        fp = totals[cat]["fp"]
        fn = totals[cat]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        agg[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        overall_f1_sum += f1

    agg["macro_f1"] = round(overall_f1_sum / len(categories), 4)
    return agg


# ---------------------------------------------------------------------------
# BERT evaluation
# ---------------------------------------------------------------------------

def evaluate_bert(test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Initializing DistilBERT NER model...")
    from src.ner_model import ResumeNERArchitecture

    ner = ResumeNERArchitecture()
    if not ner.is_fine_tuned:
        logger.error(
            "BERT model checkpoint not found at %s. "
            "Run training/train_bert.py first.",
            ner.model_dir,
        )
        return {"error": "Model checkpoint not found", "is_fine_tuned": False}

    logger.info("BERT model loaded. Running inference on %d test samples...", len(test_samples))
    per_sample_results = []
    start = time.time()

    for idx, sample in enumerate(test_samples):
        text = text_from_sample(sample)
        ground_truth = ground_truth_from_sample(sample)
        pred_result = ner.predict(text)
        predicted = pred_result.get("entities", {"PERSON": [], "SKILL": [], "DEGREE": [], "COMPANY": []})

        sample_metrics: Dict[str, Any] = {}
        for cat in ["PERSON", "SKILL", "DEGREE", "COMPANY"]:
            sample_metrics[cat] = compute_set_metrics(predicted.get(cat, []), ground_truth.get(cat, []))

        per_sample_results.append(sample_metrics)
        logger.info("  Sample %2d/%d done.", idx + 1, len(test_samples))

    elapsed = round(time.time() - start, 2)
    agg = aggregate_metrics(per_sample_results)

    logger.info(
        "BERT Evaluation complete: macro_f1=%.4f | P=%.4f | R=%.4f | elapsed=%.1fs",
        agg["macro_f1"],
        sum(agg[c]["precision"] for c in ["PERSON","SKILL","DEGREE","COMPANY"]) / 4,
        sum(agg[c]["recall"] for c in ["PERSON","SKILL","DEGREE","COMPANY"]) / 4,
        elapsed,
    )

    return {
        "model": "distilbert-base-uncased (fine-tuned)",
        "is_fine_tuned": True,
        "test_samples": len(test_samples),
        "inference_time_seconds": elapsed,
        "aggregated_metrics": agg,
        "per_sample": per_sample_results,
    }


# ---------------------------------------------------------------------------
# Gemini evaluation (optional)
# ---------------------------------------------------------------------------

def evaluate_gemini(test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Initializing Gemini extraction client...")
    from src.gemini_extractor import GeminiExtractor, is_placeholder_key, load_api_credentials

    api_key, model_name = load_api_credentials()
    if not api_key or is_placeholder_key(api_key):
        logger.warning("No valid GEMINI_API_KEY found. Skipping Gemini evaluation.")
        return {"error": "GEMINI_API_KEY not configured", "skipped": True}

    logger.info("Using Gemini model: %s", model_name)
    client = GeminiExtractor(api_key=api_key, model_name=model_name)

    per_sample_results = []
    start = time.time()

    for idx, sample in enumerate(test_samples):
        text = text_from_sample(sample)
        ground_truth = ground_truth_from_sample(sample)
        try:
            result = client.extract_entities(resume_text=text, apply_privacy_masking=True)
            predicted = result.get("entities", {"PERSON": [], "SKILL": [], "DEGREE": [], "COMPANY": []})
        except Exception as exc:
            logger.warning("Gemini extraction failed for sample %d: %s", idx + 1, exc)
            predicted = {"PERSON": [], "SKILL": [], "DEGREE": [], "COMPANY": []}

        sample_metrics: Dict[str, Any] = {}
        for cat in ["PERSON", "SKILL", "DEGREE", "COMPANY"]:
            sample_metrics[cat] = compute_set_metrics(predicted.get(cat, []), ground_truth.get(cat, []))

        per_sample_results.append(sample_metrics)
        logger.info("  Gemini sample %2d/%d done.", idx + 1, len(test_samples))
        # Rate-limit courtesy pause
        time.sleep(0.5)

    elapsed = round(time.time() - start, 2)
    agg = aggregate_metrics(per_sample_results)

    logger.info(
        "Gemini Evaluation complete: macro_f1=%.4f | elapsed=%.1fs",
        agg["macro_f1"],
        elapsed,
    )

    return {
        "model": model_name,
        "is_fine_tuned": False,
        "test_samples": len(test_samples),
        "inference_time_seconds": elapsed,
        "aggregated_metrics": agg,
        "per_sample": per_sample_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ResumeAI NER models on test set.")
    parser.add_argument(
        "--with-gemini",
        action="store_true",
        help="Also run Gemini zero-shot extraction for comparison (uses API quota).",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "test.json"),
        help="Path to test.json file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "bert_ner" / "test_evaluation.json"),
        help="Output path for evaluation results JSON.",
    )
    args = parser.parse_args()

    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error("Test data not found: %s. Run training/prepare_dataset.py first.", test_path)
        sys.exit(1)

    test_samples = load_test_set(test_path)

    # -------------------------------------------------------------------
    # BERT Evaluation
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  BERT NER Evaluation (15-Resume Test Set)")
    print("=" * 60)
    bert_results = evaluate_bert(test_samples)

    if "error" not in bert_results:
        agg = bert_results["aggregated_metrics"]
        print(f"\n{'Category':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 44)
        for cat in ["PERSON", "SKILL", "DEGREE", "COMPANY"]:
            m = agg[cat]
            print(f"{cat:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
        print("-" * 44)
        print(f"{'MACRO'::<12} {'':>10} {'':>10} {agg['macro_f1']:>10.4f}")
    else:
        print(f"\n  ERROR: {bert_results['error']}")

    # -------------------------------------------------------------------
    # Gemini Evaluation (optional)
    # -------------------------------------------------------------------
    gemini_results: Optional[Dict[str, Any]] = None
    if args.with_gemini:
        print("\n" + "=" * 60)
        print("  Gemini Zero-Shot Evaluation (15-Resume Test Set)")
        print("=" * 60)
        gemini_results = evaluate_gemini(test_samples)

        if "error" not in gemini_results and not gemini_results.get("skipped"):
            agg_g = gemini_results["aggregated_metrics"]
            print(f"\n{'Category':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 44)
            for cat in ["PERSON", "SKILL", "DEGREE", "COMPANY"]:
                m = agg_g[cat]
                print(f"{cat:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
            print("-" * 44)
            print(f"{'MACRO'::<12} {'':>10} {'':>10} {agg_g['macro_f1']:>10.4f}")
        else:
            print(f"\n  SKIPPED: {gemini_results.get('error', 'unknown reason')}")

    # -------------------------------------------------------------------
    # Side-by-Side Summary
    # -------------------------------------------------------------------
    if gemini_results and not gemini_results.get("skipped") and "error" not in bert_results:
        agg_b = bert_results["aggregated_metrics"]
        agg_g = gemini_results["aggregated_metrics"]
        print("\n" + "=" * 60)
        print("  Side-by-Side Comparison")
        print("=" * 60)
        print(f"{'Category':<12} {'BERT F1':>10} {'Gemini F1':>12} {'Winner':>10}")
        print("-" * 46)
        for cat in ["PERSON", "SKILL", "DEGREE", "COMPANY"]:
            b_f1 = agg_b[cat]["f1"]
            g_f1 = agg_g[cat]["f1"]
            winner = "BERT" if b_f1 >= g_f1 else "Gemini"
            print(f"{cat:<12} {b_f1:>10.4f} {g_f1:>12.4f} {winner:>10}")
        print("-" * 46)
        print(f"{'MACRO'::<12} {agg_b['macro_f1']:>10.4f} {agg_g['macro_f1']:>12.4f}")

    # -------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------
    output = {
        "note": (
            "All metrics measured against synthetic ground-truth test set (15 resumes). "
            "BERT was trained on synthetic data only. Numbers are real — not fabricated."
        ),
        "bert": bert_results,
    }
    if gemini_results:
        output["gemini"] = gemini_results

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Evaluation results saved to: {out_path}")


if __name__ == "__main__":
    main()
