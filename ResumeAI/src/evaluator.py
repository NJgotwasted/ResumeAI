"""
evaluator.py - Evaluation Metrics for Resume Entity Extraction and NER.

Supports:
1. Exact and soft entity-level set evaluation (Precision, Recall, F1, Jaccard).
2. Token-level seqeval metrics for HuggingFace BERT TokenClassification training.
3. Comparative evaluation between model extractions and ground truth benchmarks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


def compute_set_metrics(
    predicted_entities: List[str],
    ground_truth_entities: List[str],
    case_sensitive: bool = False
) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1-Score, and Jaccard Similarity between two entity sets.
    """
    if not case_sensitive:
        preds: Set[str] = {p.strip().lower() for p in predicted_entities if p.strip()}
        truths: Set[str] = {g.strip().lower() for g in ground_truth_entities if g.strip()}
    else:
        preds = {p.strip() for p in predicted_entities if p.strip()}
        truths = {g.strip() for g in ground_truth_entities if g.strip()}

    if not truths and not preds:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}
    if not truths or not preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}

    true_positives = len(preds.intersection(truths))
    false_positives = len(preds - truths)
    false_negatives = len(truths - preds)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    union_len = len(preds.union(truths))
    jaccard = true_positives / union_len if union_len > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(jaccard, 4),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }


def evaluate_extracted_resume(
    predicted: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Evaluates entity extraction performance across all 4 categories:
    PERSON, SKILL, DEGREE, COMPANY.
    """
    categories = ["PERSON", "SKILL", "DEGREE", "COMPANY"]
    results = {}
    total_f1 = 0.0

    for cat in categories:
        p_list = predicted.get(cat, [])
        g_list = ground_truth.get(cat, [])
        cat_metrics = compute_set_metrics(p_list, g_list)
        results[cat] = cat_metrics
        total_f1 += cat_metrics["f1"]

    results["macro_f1"] = round(total_f1 / len(categories), 4)
    return results


def build_seqeval_compute_metrics(id2label: Dict[int, str]):
    """
    Creates a `compute_metrics` callback function for HuggingFace Trainer
    using `seqeval` for token-level BIO evaluation.
    """
    import numpy as np
    from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics
