# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Batch collation for Whisper sequence-to-sequence training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import MAX_TARGET_LENGTH

IGNORE_INDEX = -100


@dataclass
class WhisperDataCollator:
    """Stack audio features and pad label sequences with -100.

    The tokenizer prepends `<|startoftranscript|>` to every label sequence, and
    the model prepends `decoder_start_token_id` again when shifting labels to
    build the decoder input. When `strip_bos` is True the leading token is
    removed (Hugging Face reference recipe); when False the labels are kept
    exactly as in the original script.
    """

    decoder_start_token_id: int
    strip_bos: bool = False
    max_target_length: int = MAX_TARGET_LENGTH

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = torch.tensor(
            np.asarray([f["input_features"] for f in features]), dtype=torch.float32
        )

        if "attention_mask" in features[0]:
            attention_mask = torch.tensor(
                np.asarray([f["attention_mask"] for f in features]), dtype=torch.long
            )
        else:
            # Whisper inputs are always padded to 30 s, so every frame is valid.
            attention_mask = torch.ones(
                (input_features.shape[0], input_features.shape[-1]), dtype=torch.long
            )

        sequences = [list(f["labels"])[: self.max_target_length] for f in features]
        if self.strip_bos and all(
            seq and seq[0] == self.decoder_start_token_id for seq in sequences
        ):
            sequences = [seq[1:] for seq in sequences]

        max_len = max(len(seq) for seq in sequences)
        labels = torch.full((len(sequences), max_len), IGNORE_INDEX, dtype=torch.long)
        for i, seq in enumerate(sequences):
            labels[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "labels": labels,
        }
