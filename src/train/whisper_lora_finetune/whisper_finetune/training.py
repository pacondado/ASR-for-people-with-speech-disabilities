# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Construction of the Hugging Face training arguments and trainer."""

from __future__ import annotations

import inspect
import logging
import math
from pathlib import Path

from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
)

from .collator import WhisperDataCollator
from .config import TrainingConfig
from .metrics import build_compute_metrics
from .utils import log_section

logger = logging.getLogger(__name__)


def _accepts(callable_obj, name: str) -> bool:
    return name in inspect.signature(callable_obj).parameters


def log_step_estimation(num_examples: int, cfg: TrainingConfig) -> None:
    steps_per_epoch = math.ceil(num_examples / cfg.effective_batch_size)
    log_section(
        "TRAINING STEP ESTIMATION",
        {
            "Training examples": num_examples,
            "Effective batch size": cfg.effective_batch_size,
            "Steps / epoch": steps_per_epoch,
            "Total optimizer steps": steps_per_epoch * cfg.num_epochs,
        },
    )


def build_training_args(cfg: TrainingConfig, output_dir: Path) -> Seq2SeqTrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=cfg.logging_steps,
        # Evaluation
        predict_with_generate=True,
        # Checkpoints
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # Precision
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        # Misc
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        label_names=["labels"],  # required because PeftModel hides the label signature
    )

    # `evaluation_strategy` was renamed to `eval_strategy` in Transformers 4.41.
    strategy_key = (
        "eval_strategy"
        if _accepts(Seq2SeqTrainingArguments.__init__, "eval_strategy")
        else "evaluation_strategy"
    )
    kwargs[strategy_key] = "epoch"

    # Drop options removed in newer Transformers releases (e.g. `logging_dir` in v5).
    supported = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    for key in [k for k in kwargs if k not in supported]:
        logger.warning("Ignoring '%s': not supported by this Transformers version.", key)
        kwargs.pop(key)

    return Seq2SeqTrainingArguments(**kwargs)


def build_trainer(
    model,
    processor: WhisperProcessor,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: TrainingConfig,
    output_dir: Path,
) -> Seq2SeqTrainer:
    collator = WhisperDataCollator(
        decoder_start_token_id=model.config.decoder_start_token_id,
        strip_bos=cfg.strip_bos_label,
    )

    kwargs = dict(
        model=model,
        args=build_training_args(cfg, output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor.tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    # `tokenizer` was replaced by `processing_class` in Transformers 4.46.
    if _accepts(Seq2SeqTrainer.__init__, "processing_class"):
        kwargs["processing_class"] = processor.tokenizer
    else:
        kwargs["tokenizer"] = processor.tokenizer

    return Seq2SeqTrainer(**kwargs)
