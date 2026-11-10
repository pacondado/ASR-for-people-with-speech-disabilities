# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Configuration objects and constants for Whisper LoRA fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Mapping from the short model name accepted on the command line
# to the Hugging Face Hub checkpoint.
SUPPORTED_MODELS: dict[str, str] = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}

# Whisper expects 16 kHz mono audio.
SAMPLING_RATE = 16_000

# Maximum number of decoder positions supported by Whisper.
MAX_TARGET_LENGTH = 448


@dataclass
class DataConfig:
    """Where the data lives and how it is split.

    Either `dataset_csv` (automatically split using `val_ratio`) or the pair
    `train_csv` + `val_csv` (pre-defined split, as in the original script) is used.
    """

    dataset_csv: str | None = None
    train_csv: str | None = None
    val_csv: str | None = None
    audio_dir: str = "audio"
    audio_column: str = "audio"
    text_column: str = "text"
    val_ratio: float = 0.2
    seed: int = 42
    # Whisper only sees the first 30 s of audio; longer files get a transcription
    # that does not match the audio and teach the model to hallucinate.
    # long_audio: "segment" (cut into aligned segments), "drop" or "keep".
    long_audio: str = "segment"
    max_audio_seconds: float = 30.0
    segment_max_seconds: float = 28.0
    segment_min_seconds: float = 3.0
    segment_min_match: float = 0.5
    # Character similarity for a reference word to count as recognised (fuzzy match).
    segment_word_similarity: float = 0.6
    segment_model: str = "large-v3"
    segment_device: str | None = None
    segment_batch_size: int = 4
    segments_subdir: str = "segments"
    # Never put rows derived from long recordings (segments or whole files) in validation.
    long_audio_train_only: bool = False
    # Keep every occurrence of the same transcription in the same split.
    group_by_text: bool = False
    # Validation rows need at least this many words (shorter ones go to training).
    min_val_words: int = 2
    # Limit the total number of rows (train + validation); None = use everything.
    max_samples: int | None = None
    num_proc: int | None = None


@dataclass
class ModelConfig:
    """Which Whisper model to fine-tune and for which language/task."""

    size: str = "medium"
    language: str = "pt"
    task: str = "transcribe"

    @property
    def checkpoint(self) -> str:
        return SUPPORTED_MODELS[self.size]


@dataclass
class LoraSettings:
    """LoRA adapter hyper-parameters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    """Optimisation and checkpointing hyper-parameters."""

    num_epochs: int = 13
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 6e-5
    warmup_steps: int = 5
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    save_total_limit: int = 3
    logging_steps: int = 5
    bf16: bool = False
    fp16: bool = False
    # Remove the leading <|startoftranscript|> from the labels (Hugging Face
    # reference recipe). Disabled by default to match the original script.
    strip_bos_label: bool = False
    output_root: str = "."

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


@dataclass
class ExperimentConfig:
    """Top-level configuration grouping every section."""

    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"]["checkpoint"] = self.model.checkpoint
        return result
