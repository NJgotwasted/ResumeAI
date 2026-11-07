
"""
evaluator.py
ResumeAI P30 - Evaluation and Gemini vs BERT comparison.
"""

from typing import Dict, List, Any


ENTITY_CATEGORIES = ["PERSON", "SKILL", "DEGREE", "COMPANY"]


def _normalize_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Normalize entity values for comparison."""
    normalized = {}

    for category in ENTITY_CATEGORIES:
        values = entities.get(category, []) or []
        cleaned = {
            str(value).strip().casefold()
            for value in values
            if value and str(value).strip()
        }
        normalized[category] = sorted(cleaned)

    return normalized


def compare_bert_vs_gemini(
    bert_entities: Dict[str, List[str]],
    gemini_entities: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Compare extracted entities from BERT and Gemini."""

    bert = _normalize_entities(bert_entities)
    gemini = _normalize_entities(gemini_entities)

    overlap = {}
    bert_only = {}
    gemini_only = {}

    for category in ENTITY_CATEGORIES:
        bert_set = set(bert[category])
        gemini_set = set(gemini[category])

        common = bert_set.intersection(gemini_set)

        overlap[category] = {
            "count": len(common),
            "entities": sorted(common)
        }

        bert_only[category] = sorted(bert_set - gemini_set)
        gemini_only[category] = sorted(gemini_set - bert_set)

    return {
        "overlap": overlap,
        "bert_only": bert_only,
        "gemini_only": gemini_only
    }


def compute_set_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, float]:
    """Calculate precision, recall and F1 for entity sets."""

    true_set = set(y_true)
    pred_set = set(y_pred)

    correct = len(true_set.intersection(pred_set))

    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(true_set) if true_set else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
