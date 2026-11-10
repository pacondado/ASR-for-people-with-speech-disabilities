# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Dataset loading, train/validation split and feature extraction."""

from __future__ import annotations

import logging
import os
import random
import re
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict, Value
from transformers import WhisperProcessor

from .config import MAX_TARGET_LENGTH, SAMPLING_RATE, DataConfig
from .long_audio import LONG_COLUMN, SOURCE_COLUMN, LongAudioHandler

logger = logging.getLogger(__name__)

TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def _resolve_audio_path(example: dict, audio_column: str, audio_dir: str) -> dict:
    """Prefix the audio file name with `audio_dir` unless it already points to a file."""
    filename = str(example[audio_column])
    path = Path(filename)
    if path.is_absolute() or path.parts[: len(Path(audio_dir).parts)] == Path(audio_dir).parts:
        example[audio_column] = str(path)
    else:
        example[audio_column] = os.path.join(audio_dir, filename)
    return example


def _has_text(example: dict, text_column: str) -> bool:
    text = example[text_column]
    return text is not None and str(text).strip() != ""


def _check_audio_files(dataset: Dataset, audio_column: str, max_listed: int = 20) -> None:
    missing = [p for p in dataset[audio_column] if not os.path.exists(p)]
    if missing:
        for path in missing[:max_listed]:
            logger.error("Missing audio file: %s", path)
        raise FileNotFoundError(f"{len(missing)} audio files are missing.")
    logger.info("All %d audio files found.", len(dataset))


_WORD_RE = re.compile(r"\w", re.UNICODE)


def _normalise_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def count_words(text: str) -> int:
    """Number of whitespace-separated tokens containing at least one letter or digit."""
    return sum(1 for token in str(text).split() if _WORD_RE.search(token))


def _val_exclusion_reasons(dataset: Dataset, cfg: DataConfig) -> list[str | None]:
    """For each row, why it may not be used for validation (None = allowed)."""
    texts = dataset[cfg.text_column]
    from_long = (
        dataset[LONG_COLUMN]
        if cfg.long_audio_train_only and LONG_COLUMN in dataset.column_names
        else [False] * len(dataset)
    )
    reasons: list[str | None] = []
    for text, is_long in zip(texts, from_long):
        if is_long:
            reasons.append("long recording")
        elif count_words(text) < cfg.min_val_words:
            reasons.append(f"fewer than {cfg.min_val_words} words")
        else:
            reasons.append(None)
    return reasons


def _log_exclusions(reasons: list[str | None], action: str) -> None:
    counts = Counter(r for r in reasons if r is not None)
    for reason, n in counts.items():
        logger.info("%d rows (%s) %s.", n, reason, action)


def _build_groups(dataset: Dataset, cfg: DataConfig) -> list[list[int]]:
    """Group row indices that must stay together in the same subset.

    Rows are linked when they come from the same original recording (segments
    of a long file) and, with `group_by_text`, when they share the same
    transcription. Without links every row is its own group.
    """
    n = len(dataset)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_seen: dict[tuple[str, str], int] = {}
    texts = dataset[cfg.text_column]
    sources = dataset[SOURCE_COLUMN] if SOURCE_COLUMN in dataset.column_names else [None] * n
    for i in range(n):
        keys = []
        if cfg.group_by_text:
            keys.append(("text", _normalise_text(texts[i])))
        if sources[i] is not None:
            keys.append(("source", str(sources[i])))
        for key in keys:
            if key in first_seen:
                parent[find(i)] = find(first_seen[key])
            else:
                first_seen[key] = i

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda idx: idx[0])


def _limit_samples(dataset: Dataset, limit: int, cfg: DataConfig, name: str) -> Dataset:
    """Randomly keep at most `limit` rows, never splitting a group (reproducible via seed)."""
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    groups = _build_groups(dataset, cfg)
    random.Random(cfg.seed).shuffle(groups)
    keep: list[int] = []
    for idx in groups:
        if len(keep) + len(idx) <= limit:
            keep.extend(idx)
        if len(keep) == limit:
            break
    logger.warning("--max-samples: %s reduced from %d to %d rows.", name, len(dataset), len(keep))
    return dataset.select(sorted(keep))


def _split(dataset: Dataset, cfg: DataConfig) -> tuple[Dataset, Dataset]:
    """Shuffle groups and fill the validation subset up to `val_ratio`.

    A group is only eligible for validation if none of its rows is excluded
    (fewer than `min_val_words` words, or derived from a long recording when
    `long_audio_train_only` is set). Excluded groups always go to training.
    """
    reasons = _val_exclusion_reasons(dataset, cfg)
    groups = _build_groups(dataset, cfg)
    random.Random(cfg.seed).shuffle(groups)

    eligible = [idx for idx in groups if all(reasons[i] is None for i in idx)]
    n_eligible = sum(len(idx) for idx in eligible)
    target = round(len(dataset) * cfg.val_ratio)
    if n_eligible < target:
        logger.warning(
            "Only %d rows are eligible for validation (target %d with val_ratio=%.2f).",
            n_eligible, target, cfg.val_ratio,
        )

    val_idx: list[int] = []
    for idx in eligible:
        if len(val_idx) >= target:
            break
        val_idx.extend(idx)

    _log_exclusions(reasons, "kept out of validation (used for training)")
    val_set = set(val_idx)
    train_idx = [i for i in range(len(dataset)) if i not in val_set]
    return dataset.select(train_idx), dataset.select(sorted(val_idx))


def _filter_validation(dataset: Dataset, cfg: DataConfig) -> Dataset:
    """Pre-defined validation set: remove rows that may not be used for validation."""
    reasons = _val_exclusion_reasons(dataset, cfg)
    _log_exclusions(reasons, "removed from the validation CSV")
    return dataset.select([i for i, r in enumerate(reasons) if r is None])


def load_csv(path: str, cfg: DataConfig) -> Dataset:
    """Load one CSV, validate its columns, drop empty transcriptions and resolve audio paths."""
    logger.info("Loading dataset from %s", path)
    # Read with pandas instead of `load_dataset("csv", ...)`: some `datasets`
    # versions contact the Hugging Face Hub even for local CSV files (usage
    # statistics), which hangs or fails without a working connection.
    # All columns are read as text so file names and transcriptions are
    # never converted to numbers or NaN.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    for column in (cfg.audio_column, cfg.text_column):
        if column not in dataset.column_names:
            raise ValueError(
                f"Column '{column}' not found in {path}. "
                f"Available columns: {dataset.column_names}"
            )

    # Rows without a transcription would break WER/CER computation.
    n_before = len(dataset)
    dataset = dataset.filter(partial(_has_text, text_column=cfg.text_column))
    if len(dataset) < n_before:
        logger.warning("Dropped %d rows with empty transcription in %s.", n_before - len(dataset), path)

    dataset = dataset.map(
        partial(_resolve_audio_path, audio_column=cfg.audio_column, audio_dir=cfg.audio_dir)
    )
    _check_audio_files(dataset, cfg.audio_column)
    return dataset


def load_and_split(cfg: DataConfig, long_audio: LongAudioHandler | None = None) -> DatasetDict:
    """Return the training and validation subsets.

    * `train_csv` + `val_csv`: the pre-defined split is used; validation rows
      with fewer than `min_val_words` words are removed.
    * `dataset_csv`: a single CSV is split with `val_ratio` (reproducible
      through `cfg.seed`). Segments of the same long recording (and, with
      `group_by_text`, identical sentences) stay in the same subset, and rows
      with fewer than `min_val_words` words only go to training.

    Recordings longer than `cfg.max_audio_seconds` are handled by `long_audio`
    (segmented, dropped or kept) first. `max_samples` then limits the total
    number of rows (train + validation).
    """

    def load(path: str) -> Dataset:
        dataset = load_csv(path, cfg)
        return long_audio.process(dataset, path) if long_audio is not None else dataset

    if cfg.train_csv and cfg.val_csv:
        train, val = load(cfg.train_csv), load(cfg.val_csv)
        val = _filter_validation(val, cfg)
        if cfg.max_samples and len(train) + len(val) > cfg.max_samples:
            n_val = max(1, round(cfg.max_samples * len(val) / (len(train) + len(val))))
            train = _limit_samples(train, cfg.max_samples - n_val, cfg, "train")
            val = _limit_samples(val, n_val, cfg, "validation")
        splits = DatasetDict({TRAIN_SPLIT: train, VAL_SPLIT: val})
        logger.info(
            "Pre-defined split -> train: %d | validation: %d",
            len(splits[TRAIN_SPLIT]),
            len(splits[VAL_SPLIT]),
        )
    elif cfg.dataset_csv:
        dataset = load(cfg.dataset_csv)
        if cfg.max_samples:
            dataset = _limit_samples(dataset, cfg.max_samples, cfg, cfg.dataset_csv)
        train, val = _split(dataset, cfg)
        splits = DatasetDict({TRAIN_SPLIT: train, VAL_SPLIT: val})
        logger.info(
            "Split %d examples -> train: %d | validation: %d "
            "(val_ratio=%.2f, seed=%d, group_by_text=%s, min_val_words=%d)",
            len(dataset),
            len(splits[TRAIN_SPLIT]),
            len(splits[VAL_SPLIT]),
            cfg.val_ratio,
            cfg.seed,
            cfg.group_by_text,
            cfg.min_val_words,
        )
    else:
        raise ValueError("Provide either dataset_csv or both train_csv and val_csv.")

    if len(splits[TRAIN_SPLIT]) == 0 or len(splits[VAL_SPLIT]) == 0:
        raise ValueError("Dataset too small: one of the splits is empty.")
    return splits


def export_splits(splits: DatasetDict, output_dir: str | Path) -> None:
    """Save the exact train/validation rows to CSV so the run can be reproduced."""
    output_dir = Path(output_dir)
    for name, subset in splits.items():
        subset.to_csv(str(output_dir / f"{name}_split.csv"), index=False)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _audio_to_array(audio: Any) -> tuple[np.ndarray, int]:
    """Return (samples, sampling_rate) for both `datasets` < 4 (dict) and >= 4 (AudioDecoder)."""
    if isinstance(audio, dict):
        return np.asarray(audio["array"]), int(audio["sampling_rate"])
    samples = audio.get_all_samples()
    return samples.data.mean(dim=0).numpy(), int(samples.sample_rate)


def _prepare_example(
    example: dict,
    processor: WhisperProcessor,
    audio_column: str,
    text_column: str,
) -> dict:
    array, sampling_rate = _audio_to_array(example[audio_column])

    features = processor.feature_extractor(
        array,
        sampling_rate=sampling_rate,
        return_attention_mask=True,
    )
    labels = processor.tokenizer(
        str(example[text_column]).strip(),
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    ).input_ids

    result = {"input_features": features.input_features[0], "labels": labels}
    if "attention_mask" in features:
        result["attention_mask"] = features.attention_mask[0]
    return result


def prepare_dataset(dataset: Dataset, processor: WhisperProcessor, cfg: DataConfig) -> Dataset:
    """Decode audio at 16 kHz, compute log-Mel features and tokenize transcriptions."""
    # Recent `datasets` versions read CSV text as `large_string`, which cannot be
    # cast directly to `Audio`; normalise to `string` first.
    dataset = dataset.cast_column(cfg.audio_column, Value("string"))
    dataset = dataset.cast_column(cfg.audio_column, Audio(sampling_rate=SAMPLING_RATE))
    return dataset.map(
        partial(
            _prepare_example,
            processor=processor,
            audio_column=cfg.audio_column,
            text_column=cfg.text_column,
        ),
        remove_columns=dataset.column_names,
        num_proc=cfg.num_proc,
    )
