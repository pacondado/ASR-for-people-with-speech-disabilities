# Copyright (c) 2026 Paulo Condado
# License: MIT
"""WER / CER computation for the Trainer and for per-phrase evaluation."""

from __future__ import annotations

from typing import Callable

import jiwer
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerBase

from .collator import IGNORE_INDEX


def word_error_rate(references: list[str], predictions: list[str]) -> float:
    return float(jiwer.wer(references, predictions))


def char_error_rate(references: list[str], predictions: list[str]) -> float:
    return float(jiwer.cer(references, predictions))


def build_compute_metrics(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Return a `compute_metrics` function for `Seq2SeqTrainer` (corpus-level WER and CER)."""

    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.where(np.asarray(predictions) == IGNORE_INDEX, pad_id, predictions)
        labels = np.where(np.asarray(labels) == IGNORE_INDEX, pad_id, labels)

        decoded_preds = [t.strip() for t in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        decoded_labels = [t.strip() for t in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        return {
            "wer": word_error_rate(decoded_labels, decoded_preds),
            "cer": char_error_rate(decoded_labels, decoded_preds),
        }

    return compute_metrics
