# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Generic helpers: logging, device detection, output paths and JSON I/O."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

SEPARATOR = "=" * 70


class _DropMessages(logging.Filter):
    """Logging filter that drops records containing any of the given texts."""

    def __init__(self, *texts: str) -> None:
        super().__init__()
        self.texts = texts

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(text in message for text in self.texts)


def silence_whisper_attention_mask_warning() -> None:
    """Hide a misleading `generate()` warning that Whisper always triggers.

    Whisper uses the same token for padding and end-of-text. When the
    validation step calls `generate()`, older Transformers versions warn that
    "the attention mask is not set" even though the audio attention mask *is*
    passed: the check runs on the decoder prompt (<|startoftranscript|><|pt|>
    <|transcribe|><|notimestamps|>), which is identical for every utterance
    and has no padding, so there is nothing to mask. The warning does not
    affect the results.
    """
    logging.getLogger("transformers.generation.utils").addFilter(
        _DropMessages("The attention mask is not set")
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    silence_whisper_attention_mask_warning()


def log_section(title: str, items: Mapping[str, Any] | None = None) -> None:
    """Log a titled block of `key: value` lines."""
    logger.info(SEPARATOR)
    logger.info(title)
    logger.info(SEPARATOR)
    if items:
        width = max(len(k) for k in items)
        for key, value in items.items():
            logger.info("%s : %s", key.ljust(width), value)


def detect_device() -> str:
    """Return the best available accelerator (informational; Trainer selects it itself)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_output_dir(
    root: str | Path,
    model_size: str,
    n_train: int,
    n_val: int,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    timestamp: datetime | None = None,
) -> Path:
    """Build `output_lora_[MODEL]_r[R]_a[ALPHA]_lr[LR]_[N_TRAIN]_[N_VAL]_[YYYYMMDD-HHMMSS]`.

    The learning rate uses its shortest form (e.g. 6e-05, 0.0001).
    """
    timestamp = timestamp or datetime.now()
    name = (
        f"output_lora_{model_size}_r{lora_r}_a{lora_alpha}_lr{learning_rate:g}_"
        f"{n_train}_{n_val}_{timestamp:%Y%m%d-%H%M%S}"
    )
    return Path(root) / name


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path
