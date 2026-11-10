# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Command-line interface: turns CLI arguments into an ExperimentConfig."""

from __future__ import annotations

import argparse
from typing import Sequence

from .config import (
    SUPPORTED_MODELS,
    DataConfig,
    ExperimentConfig,
    LoraSettings,
    ModelConfig,
    TrainingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    defaults_t = TrainingConfig()
    defaults_l = LoraSettings()

    parser = argparse.ArgumentParser(
        description="Fine-tune OpenAI Whisper with LoRA on a single CSV dataset "
        "that is automatically split into training and validation sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data -----------------------------------------------------------
    data = parser.add_argument_group("data")
    source = data.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        help="Single CSV (audio-path + transcription columns), split automatically with --val-ratio.",
    )
    source.add_argument(
        "--train-csv",
        help="Pre-split training CSV (requires --val-csv; --val-ratio is ignored).",
    )
    data.add_argument("--val-csv", help="Pre-split validation CSV (used with --train-csv).")
    data.add_argument("--audio-dir", default="audio", help="Directory containing the audio files.")
    data.add_argument("--audio-column", default="audio", help="CSV column with the audio file name.")
    data.add_argument("--text-column", default="text", help="CSV column with the transcription.")
    data.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of the dataset used for validation.",
    )
    data.add_argument("--seed", type=int, default=42, help="Random seed (split and training).")
    data.add_argument(
        "--min-val-words",
        type=int,
        default=2,
        help="Minimum number of words for a validation example. Shorter examples are used "
        "only for training (--dataset) or removed from the validation CSV (--train-csv).",
    )
    data.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the total number of examples (train + validation) to this value, "
        "chosen at random with --seed. Applied after long-audio segmentation.",
    )
    long_audio = parser.add_argument_group("long audio (> 30 s)")
    long_audio.add_argument(
        "--long-audio",
        choices=["segment", "drop", "keep"],
        default="segment",
        help="segment: cut long recordings into aligned segments with Whisper word timestamps "
        "(cached); drop: discard them; keep: use them as is (Whisper truncates at 30 s).",
    )
    long_audio.add_argument(
        "--long-audio-train-only",
        action="store_true",
        help="Use long recordings (their segments with 'segment', the whole files with 'keep') "
        "only for training: they never go to validation.",
    )
    long_audio.add_argument(
        "--max-audio-seconds", type=float, default=30.0, help="Recordings longer than this are 'long'."
    )
    long_audio.add_argument("--segment-max-seconds", type=float, default=28.0, help="Maximum segment length.")
    long_audio.add_argument("--segment-min-seconds", type=float, default=3.0, help="Preferred minimum segment length.")
    long_audio.add_argument(
        "--segment-min-match",
        type=float,
        default=0.5,
        help="Drop segments where a smaller fraction of reference words was found by the ASR.",
    )
    long_audio.add_argument(
        "--segment-word-similarity",
        type=float,
        default=0.6,
        help="Character similarity (0-1) for a reference word to count as recognised by the ASR "
        "(lower it for atypical / dysarthric speech).",
    )
    long_audio.add_argument(
        "--segment-model", choices=sorted(SUPPORTED_MODELS), default="large-v3", help="Whisper model used for timing."
    )
    long_audio.add_argument("--segment-device", default=None, help="cuda / mps / cpu (auto by default).")
    long_audio.add_argument("--segment-batch-size", type=int, default=4)
    long_audio.add_argument(
        "--segments-subdir", default="segments", help="Sub-folder of --audio-dir where segments are cached."
    )
    data.add_argument(
        "--group-by-text",
        action="store_true",
        help="With --dataset: keep all recordings of the same sentence in the same split "
        "(validation only contains sentences never seen in training).",
    )
    data.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes used for dataset pre-processing.",
    )

    # ---- Model ----------------------------------------------------------
    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS),
        default="medium",
        help="Whisper model size to fine-tune.",
    )
    model.add_argument("--language", default="pt", help="Transcription language code.")
    model.add_argument(
        "--task", choices=["transcribe", "translate"], default="transcribe", help="Whisper task."
    )

    # ---- Training -------------------------------------------------------
    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=defaults_t.num_epochs)
    train.add_argument("--batch-size", type=int, default=defaults_t.batch_size)
    train.add_argument("--grad-accum", type=int, default=defaults_t.grad_accum)
    train.add_argument("--learning-rate", type=float, default=defaults_t.learning_rate)
    train.add_argument("--warmup-steps", type=int, default=defaults_t.warmup_steps)
    train.add_argument("--weight-decay", type=float, default=defaults_t.weight_decay)
    train.add_argument(
        "--early-stopping-patience", type=int, default=defaults_t.early_stopping_patience
    )
    train.add_argument("--save-total-limit", type=int, default=defaults_t.save_total_limit)
    train.add_argument("--logging-steps", type=int, default=defaults_t.logging_steps)
    precision = train.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision.")
    precision.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision (CUDA only).")
    train.add_argument(
        "--strip-bos-label",
        action="store_true",
        help="Remove the leading <|startoftranscript|> token from the labels "
        "(Hugging Face reference recipe). Off by default, as in the original script.",
    )
    train.add_argument(
        "--output-root",
        default=".",
        help="Parent directory where the run output directory is created.",
    )

    # ---- LoRA -----------------------------------------------------------
    lora = parser.add_argument_group("lora")
    lora.add_argument("--lora-r", type=int, default=defaults_l.r)
    lora.add_argument("--lora-alpha", type=int, default=defaults_l.alpha)
    lora.add_argument("--lora-dropout", type=float, default=defaults_l.dropout)
    lora.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=defaults_l.target_modules,
        help="Names of the modules to which LoRA adapters are attached.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be strictly between 0 and 1.")
    if args.max_samples is not None and args.max_samples < 2:
        parser.error("--max-samples must be at least 2.")
    if args.min_val_words < 1:
        parser.error("--min-val-words must be at least 1.")
    if not 0.0 < args.segment_max_seconds < 30.0:
        parser.error("--segment-max-seconds must be between 0 and 30.")
    if args.train_csv and not args.val_csv:
        parser.error("--train-csv requires --val-csv.")
    if args.val_csv and not args.train_csv:
        parser.error("--val-csv can only be used together with --train-csv.")

    return ExperimentConfig(
        data=DataConfig(
            dataset_csv=args.dataset,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            audio_dir=args.audio_dir,
            audio_column=args.audio_column,
            text_column=args.text_column,
            val_ratio=args.val_ratio,
            seed=args.seed,
            long_audio=args.long_audio,
            max_audio_seconds=args.max_audio_seconds,
            segment_max_seconds=args.segment_max_seconds,
            segment_min_seconds=args.segment_min_seconds,
            segment_min_match=args.segment_min_match,
            segment_word_similarity=args.segment_word_similarity,
            segment_model=args.segment_model,
            segment_device=args.segment_device,
            segment_batch_size=args.segment_batch_size,
            segments_subdir=args.segments_subdir,
            long_audio_train_only=args.long_audio_train_only,
            group_by_text=args.group_by_text,
            min_val_words=args.min_val_words,
            max_samples=args.max_samples,
            num_proc=args.num_proc,
        ),
        model=ModelConfig(size=args.model, language=args.language, task=args.task),
        lora=LoraSettings(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
        ),
        training=TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
            save_total_limit=args.save_total_limit,
            logging_steps=args.logging_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            strip_bos_label=args.strip_bos_label,
            output_root=args.output_root,
        ),
    )
