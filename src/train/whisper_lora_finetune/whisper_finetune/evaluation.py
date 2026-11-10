# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Phrase-by-phrase evaluation of the fine-tuned model on the validation split."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import WhisperProcessor

from .config import ModelConfig
from .metrics import char_error_rate, word_error_rate
from .utils import log_section, save_json

logger = logging.getLogger(__name__)


@torch.no_grad()
def transcribe(
    model,
    processor: WhisperProcessor,
    dataset: Dataset,
    model_cfg: ModelConfig,
    batch_size: int,
) -> list[str]:
    """Generate transcriptions for a pre-processed dataset (with `input_features`)."""
    model.eval()
    param = next(model.parameters())
    predictions: list[str] = []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        generate_kwargs = {
            "input_features": torch.tensor(
                np.asarray(batch["input_features"]), dtype=param.dtype, device=param.device
            ),
            "language": model_cfg.language,
            "task": model_cfg.task,
            "do_sample": False,
        }
        if "attention_mask" in batch:
            generate_kwargs["attention_mask"] = torch.tensor(
                np.asarray(batch["attention_mask"]), dtype=torch.long, device=param.device
            )

        generated_ids = model.generate(**generate_kwargs)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(text.strip() for text in decoded)

    return predictions


def evaluate_per_phrase(
    model,
    processor: WhisperProcessor,
    dataset: Dataset,
    references: list[str],
    model_cfg: ModelConfig,
    batch_size: int,
    output_dir: Path,
) -> dict[str, float]:
    """Compute WER/CER for every validation phrase, log them and save to JSON."""
    log_section("PHRASE-BY-PHRASE VALIDATION")

    predictions = transcribe(model, processor, dataset, model_cfg, batch_size)

    results = []
    for index, (reference, hypothesis) in enumerate(zip(references, predictions), start=1):
        wer = word_error_rate([reference], [hypothesis])
        cer = char_error_rate([reference], [hypothesis])
        results.append(
            {"index": index, "reference": reference, "hypothesis": hypothesis, "wer": wer, "cer": cer}
        )
        logger.info("[%03d] REF: %s", index, reference)
        logger.info("[%03d] HYP: %s", index, hypothesis)
        logger.info("[%03d] WER: %.3f | CER: %.3f", index, wer, cer)

    results_file = save_json(results, output_dir / "validation_results.json")

    summary = {
        "mean_phrase_wer": float(np.mean([r["wer"] for r in results])),
        "mean_phrase_cer": float(np.mean([r["cer"] for r in results])),
        "corpus_wer": word_error_rate(references, predictions),
        "corpus_cer": char_error_rate(references, predictions),
    }
    save_json(summary, output_dir / "validation_summary.json")

    log_section("PHRASE-BY-PHRASE SUMMARY", {**summary, "Detailed results": results_file})
    return summary
