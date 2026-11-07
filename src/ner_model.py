"""
ner_model.py - Fine-Tuned DistilBERT Resume Named Entity Recognition.

STATUS: PHASE 2 - PRODUCTION INFERENCE MODULE.
Model: distilbert-base-uncased fine-tuned on 85 synthetic resume samples.
Training completed: 4 epochs, eval_f1=0.5653, eval_precision=0.5116, eval_recall=0.6316
Trained on SYNTHETIC data only. Gemini remains the production extraction path.

Label schema (from models/bert_ner/config.json):
  0: O
  1: B-PER  2: I-PER
  3: B-SKILL 4: I-SKILL
  5: B-DEGREE 6: I-DEGREE
  7: B-COMPANY 8: I-COMPANY
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label constants — must match models/bert_ner/config.json exactly
# ---------------------------------------------------------------------------
ENTITY_LABELS: List[str] = [
    "O",
    "B-PER", "I-PER",
    "B-SKILL", "I-SKILL",
    "B-DEGREE", "I-DEGREE",
    "B-COMPANY", "I-COMPANY",
]

ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(ENTITY_LABELS)}
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(ENTITY_LABELS)}

# Map BIO prefixes to output category keys
_PREFIX_TO_CATEGORY: Dict[str, str] = {
    "PER": "PERSON",
    "SKILL": "SKILL",
    "DEGREE": "DEGREE",
    "COMPANY": "COMPANY",
}

# Default checkpoint path relative to project root
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bert_ner"


def _empty_entities() -> Dict[str, List[str]]:
    return {"PERSON": [], "SKILL": [], "DEGREE": [], "COMPANY": []}


def _aggregate_bio_spans(
    tokens: List[str],
    tag_ids: List[int],
    id2label: Dict[int, str],
) -> Dict[str, List[str]]:
    """
    Converts a flat list of BIO tag IDs back to grouped entity strings.

    Strategy:
    - Walk tokens/tags left-to-right.
    - On B-X: start a new span of type X.
    - On I-X (matching current open span): extend span.
    - On anything else: flush the current span if one is open.
    - Subword tokens that begin with '##' are merged without space.
    - Skip [CLS] / [SEP] / [PAD] special tokens.
    """
    entities = _empty_entities()
    current_span: List[str] = []
    current_type: Optional[str] = None

    def flush_span():
        if current_type and current_span:
            # Join subwords: tokens starting with ## are continuations
            merged = ""
            for tok in current_span:
                if tok.startswith("##"):
                    merged += tok[2:]
                else:
                    merged = (merged + " " + tok).strip() if merged else tok
            # Clean up artifacts and filter noise
            merged = merged.strip()
            if merged and len(merged) > 1 and not re.fullmatch(r"[^a-zA-Z0-9]+", merged):
                category = _PREFIX_TO_CATEGORY.get(current_type)
                if category and merged not in entities[category]:
                    entities[category].append(merged)

    for token, tag_id in zip(tokens, tag_ids):
        # Skip special tokens
        if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"):
            flush_span()
            current_span = []
            current_type = None
            continue

        label = id2label.get(tag_id, "O")

        if label.startswith("B-"):
            flush_span()
            current_type = label[2:]  # e.g. "PER", "SKILL"
            current_span = [token]

        elif label.startswith("I-"):
            bio_type = label[2:]
            if bio_type == current_type:
                current_span.append(token)
            else:
                # Mismatched I- tag: treat as new B-
                flush_span()
                current_type = bio_type
                current_span = [token]

        else:  # "O"
            flush_span()
            current_span = []
            current_type = None

    flush_span()
    return entities


class ResumeNERArchitecture:
    """
    Fine-tuned DistilBERT Named Entity Recognition for Resume Parsing.

    Loads the model checkpoint from models/bert_ner/ and performs real
    inference. If the checkpoint is missing, falls back gracefully with
    is_fine_tuned=False and returns empty entity lists.

    Training details (CPU, synthetic data):
      - Base model: distilbert-base-uncased
      - Epochs: 4
      - Final eval F1:  0.5653
      - Precision:      0.5116
      - Recall:         0.6316
      - Accuracy:       0.9262
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        base_model_name: str = "distilbert-base-uncased",
    ):
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self.base_model_name = base_model_name
        self.is_fine_tuned: bool = False
        self.model_path: str = str(self.model_dir)
        self._tokenizer = None
        self._model = None
        self._id2label: Dict[int, str] = ID2LABEL.copy()
        self._loaded: bool = False

        # Training metadata for UI display
        self.training_metadata: Dict[str, Any] = {
            "epochs": 4,
            "eval_f1": 0.5653,
            "eval_precision": 0.5116,
            "eval_recall": 0.6316,
            "eval_accuracy": 0.9262,
            "eval_loss": 0.2156,
            "training_data": "synthetic (100 annotated resumes)",
            "device": "CPU",
            "base_model": "distilbert-base-uncased",
        }

        self._try_load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load(self) -> None:
        """Attempt to load tokenizer and model from checkpoint. Silent on failure."""
        config_file = self.model_dir / "config.json"
        tokenizer_file = self.model_dir / "tokenizer.json"
        weights_file = self.model_dir / "model.safetensors"

        if not all(f.exists() for f in [config_file, tokenizer_file, weights_file]):
            logger.info(
                "BERT checkpoint not found at %s. "
                "Run training/train_bert.py to generate the model.",
                self.model_dir,
            )
            return

        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            import torch

            logger.info("Loading DistilBERT tokenizer from %s", self.model_dir)
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

            logger.info("Loading DistilBERT model from %s", self.model_dir)
            self._model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
            self._model.eval()

            # Prefer id2label from loaded config
            if hasattr(self._model.config, "id2label"):
                loaded = self._model.config.id2label
                self._id2label = {int(k): v for k, v in loaded.items()}

            self.is_fine_tuned = True
            self._loaded = True
            logger.info(
                "DistilBERT NER loaded successfully. Labels: %s",
                list(self._id2label.values()),
            )

        except Exception as exc:
            logger.warning("Failed to load BERT model: %s", exc)
            self.is_fine_tuned = False
            self._loaded = False

    # ------------------------------------------------------------------
    # Tokenization alignment (used externally by training scripts too)
    # ------------------------------------------------------------------

    def tokenize_and_align_labels(
        self,
        examples: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """
        HuggingFace Dataset mapping function for subword BIO label alignment.
        Aligns word-level BIO annotations with DistilBERT WordPiece tokens.
        Special tokens and continuation subwords receive label -100.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Call _try_load() or ensure the model checkpoint exists."
            )

        tokenized_inputs = self._tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=512,
        )

        labels = []
        for i, label_seq in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            prev_word_idx = None
            label_ids: List[int] = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # [CLS] / [SEP]
                elif word_idx != prev_word_idx:
                    label_ids.append(label_seq[word_idx])  # first subword
                else:
                    label_ids.append(-100)  # continuation subword
                prev_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run DistilBERT NER inference on resume text.

        Returns a dict with:
          - is_fine_tuned (bool)
          - entities: {PERSON, SKILL, DEGREE, COMPANY} → List[str]
          - model_type, status, notice
        """
        if not self.is_fine_tuned or not self._loaded:
            return {
                "model_type": "DistilBERT Token Classification",
                "is_fine_tuned": False,
                "status": (
                    "Fine-tuned checkpoint not loaded. "
                    "Run training/train_bert.py to generate models/bert_ner/."
                ),
                "entities": _empty_entities(),
                "notice": (
                    "BERT model requires a trained checkpoint. "
                    "Gemini is the active production extraction engine."
                ),
            }

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Shape: [1, seq_len, num_labels] → [seq_len]
            tag_ids: List[int] = torch.argmax(outputs.logits, dim=2)[0].tolist()
            tokens: List[str] = self._tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0]
            )

            entities = _aggregate_bio_spans(tokens, tag_ids, self._id2label)
            total = sum(len(v) for v in entities.values())

            return {
                "model_type": "DistilBERT Token Classification",
                "is_fine_tuned": True,
                "status": f"Fine-tuned DistilBERT inference completed. {total} entities found.",
                "entities": entities,
                "notice": (
                    "⚠️ Trained on SYNTHETIC data only (100 samples). "
                    "F1=0.5653 on synthetic validation set. "
                    "Gemini extraction is recommended for production use."
                ),
            }

        except Exception as exc:
            logger.error("BERT inference failed: %s", exc)
            return {
                "model_type": "DistilBERT Token Classification",
                "is_fine_tuned": True,
                "status": f"Inference error: {exc}",
                "entities": _empty_entities(),
                "notice": "An error occurred during BERT inference.",
            }

    def get_training_config(
        self,
        output_dir: str = "./models/bert_ner",
        num_train_epochs: int = 4,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
    ) -> Dict[str, Any]:
        """Returns TrainingArguments dict for HuggingFace Trainer."""
        return {
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": num_train_epochs,
            "weight_decay": 0.01,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "logging_dir": f"{output_dir}/logs",
        }
