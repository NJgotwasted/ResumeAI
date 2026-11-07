"""
train_bert.py - Fine-Tuning DistilBERT for Resume Named Entity Recognition.

Performs:
1. Loads synthetic annotated resumes from data/processed/train.json and data/processed/val.json.
2. Tokenizes and aligns BIO labels with subword tokens using DistilBERT tokenizer.
3. Sets -100 for special tokens and non-initial subwords.
4. Fine-tunes AutoModelForTokenClassification with PyTorch and HuggingFace Trainer.
5. Computes token-level seqeval metrics (Precision, Recall, F1).
6. Saves trained model, tokenizer, and training history logs locally to models/bert_ner/.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainBERT")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "bert_ner"

BASE_MODEL_NAME = "distilbert-base-uncased"
RANDOM_SEED = 42
set_seed(RANDOM_SEED)


class LossHistoryCallback(TrainerCallback):
    """Custom callback to collect step-by-step training loss and validation metrics."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step, "epoch": logs.get("epoch", 0.0)}
            if "loss" in logs:
                entry["train_loss"] = round(logs["loss"], 4)
            if "eval_loss" in logs:
                entry["eval_loss"] = round(logs["eval_loss"], 4)
            if "eval_f1" in logs:
                entry["eval_f1"] = round(logs["eval_f1"], 4)
            if "eval_precision" in logs:
                entry["eval_precision"] = round(logs["eval_precision"], 4)
            if "eval_recall" in logs:
                entry["eval_recall"] = round(logs["eval_recall"], 4)
            self.history.append(entry)


def load_dataset_splits() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Loads train/val json files and label mappings."""
    train_file = DATA_DIR / "train.json"
    val_file = DATA_DIR / "val.json"
    labels_file = DATA_DIR / "labels.json"

    if not train_file.exists() or not val_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {DATA_DIR}. Please run training/prepare_dataset.py first."
        )

    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    with open(labels_file, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)

    return train_data, val_data, labels_meta


def align_labels_with_tokens(tokenizer, examples):
    """
    Subword token alignment using DistilBERT tokenizer.
    Assigns -100 to special tokens ([CLS], [SEP]) and subsequent subwords within the same word.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword token of the word gets the original label
                label_ids.append(label[word_idx])
            else:
                # Subsequent subword tokens get -100
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def build_metrics_fn(id2label: Dict[int, str]):
    """Constructs seqeval evaluation function for token classification."""
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
            "precision": round(float(precision_score(true_labels, true_predictions)), 4),
            "recall": round(float(recall_score(true_labels, true_predictions)), 4),
            "f1": round(float(f1_score(true_labels, true_predictions)), 4),
            "accuracy": round(float(accuracy_score(true_labels, true_predictions)), 4),
        }

    return compute_metrics


def train_distilbert_ner():
    start_time = time.time()
    logger.info("=" * 65)
    logger.info("PHASE 2: DISTILBERT RESUME NER MODEL TRAINING")
    logger.info("=" * 65)

    # 1. Load Data
    train_raw, val_raw, labels_meta = load_dataset_splits()
    label_list = labels_meta["label_list"]
    id2label = {int(k): v for k, v in labels_meta["id2label"].items()}
    label2id = labels_meta["label2id"]

    logger.info(f"Loaded {len(train_raw)} training resumes and {len(val_raw)} validation resumes.")
    logger.info(f"BIO Labels ({len(label_list)}): {label_list}")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list([{"tokens": r["tokens"], "ner_tags": r["ner_tags"]} for r in train_raw])
    val_dataset = Dataset.from_list([{"tokens": r["tokens"], "ner_tags": r["ner_tags"]} for r in val_raw])

    # 2. Tokenizer & Subword Alignment
    logger.info(f"Initializing tokenizer from '{BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    logger.info("Aligning BIO labels with subword tokens...")
    train_encoded = train_dataset.map(lambda ex: align_labels_with_tokens(tokenizer, ex), batched=True)
    val_encoded = val_dataset.map(lambda ex: align_labels_with_tokens(tokenizer, ex), batched=True)

    # 3. Model Architecture
    logger.info(f"Initializing {BASE_MODEL_NAME} for Token Classification with {len(label_list)} labels...")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = build_metrics_fn(id2label)
    history_callback = LossHistoryCallback()

    # 4. Training Arguments
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=4e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=5,
        seed=RANDOM_SEED,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[history_callback],
    )

    # 5. Train
    logger.info("Starting model training...")
    train_result = trainer.train()

    # 6. Final Evaluation on Validation Set
    eval_metrics = trainer.evaluate()
    logger.info(f"Final Validation Metrics: {eval_metrics}")

    # 7. Save Model & Tokenizer
    logger.info(f"Saving fine-tuned DistilBERT model and tokenizer to {MODEL_OUTPUT_DIR}...")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    # Save training history and loss curves
    elapsed_time = round(time.time() - start_time, 2)
    history_file = MODEL_OUTPUT_DIR / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_loss_history": history_callback.history,
                "eval_metrics": eval_metrics,
                "elapsed_time_seconds": elapsed_time,
                "epochs": 4,
            },
            f,
            indent=2,
        )

    # Save metadata
    metadata = {
        "model_architecture": "DistilBertForTokenClassification",
        "base_model": BASE_MODEL_NAME,
        "num_labels": len(label_list),
        "labels": label_list,
        "id2label": id2label,
        "label2id": label2id,
        "training_samples": len(train_raw),
        "validation_samples": len(val_raw),
        "validation_f1": eval_metrics.get("eval_f1", 0.0),
        "validation_precision": eval_metrics.get("eval_precision", 0.0),
        "validation_recall": eval_metrics.get("eval_recall", 0.0),
        "training_time_seconds": elapsed_time,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    with open(MODEL_OUTPUT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"TRAINING COMPLETE IN {elapsed_time}s! Model saved to {MODEL_OUTPUT_DIR}")
    logger.info(f"Validation F1: {eval_metrics.get('eval_f1', 0.0):.4f}")
    logger.info("=" * 65)


if __name__ == "__main__":
    train_distilbert_ner()

