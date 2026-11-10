# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Entry point: fine-tune Whisper with LoRA on a single CSV split 80/20.

Example:
    python train.py --dataset data.csv --audio-dir audio --model medium
"""

from __future__ import annotations

import logging
import sys
from typing import Sequence

import torch
from transformers import set_seed

from whisper_finetune.cli import parse_args
from whisper_finetune.data import TRAIN_SPLIT, VAL_SPLIT, export_splits, load_and_split, prepare_dataset
from whisper_finetune.evaluation import evaluate_per_phrase
from whisper_finetune.long_audio import LongAudioHandler
from whisper_finetune.modeling import apply_lora, load_model, load_processor
from whisper_finetune.training import build_trainer, log_step_estimation
from whisper_finetune.utils import build_output_dir, detect_device, log_section, save_json, setup_logging

logger = logging.getLogger("train")


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    cfg = parse_args(argv)
    set_seed(cfg.data.seed)

    log_section(
        "WHISPER FINE-TUNING WITH LoRA",
        {
            "PyTorch": torch.__version__,
            "Device": detect_device(),
            "Model": cfg.model.checkpoint,
            "Language": cfg.model.language,
            "Task": cfg.model.task,
            "Dataset": cfg.data.dataset_csv or f"{cfg.data.train_csv} + {cfg.data.val_csv}",
            "Validation ratio": cfg.data.val_ratio if cfg.data.dataset_csv else "pre-defined",
            "Strip BOS label": cfg.training.strip_bos_label,
            "Long audio (> %.0f s)" % cfg.data.max_audio_seconds: cfg.data.long_audio,
            "Epochs": cfg.training.num_epochs,
            "Effective batch": cfg.training.effective_batch_size,
            "Learning rate": cfg.training.learning_rate,
            "LoRA r / alpha": f"{cfg.lora.r} / {cfg.lora.alpha}",
        },
    )

    # ---- Data -----------------------------------------------------------
    long_audio = LongAudioHandler(cfg.data, cfg.model.language)
    splits = load_and_split(cfg.data, long_audio)
    long_audio.close()  # free the segmentation ASR model before training
    n_train, n_val = len(splits[TRAIN_SPLIT]), len(splits[VAL_SPLIT])

    output_dir = build_output_dir(
        cfg.training.output_root, cfg.model.size, n_train, n_val,
        cfg.lora.r, cfg.lora.alpha, cfg.training.learning_rate,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Output directory: %s", output_dir)

    save_json(cfg.to_dict(), output_dir / "run_config.json")
    export_splits(splits, output_dir)
    segments_report = long_audio.report()
    if len(segments_report):
        segments_report.to_csv(output_dir / "segments_report.csv", index=False)

    processor = load_processor(cfg.model)
    references = [str(t).strip() for t in splits[VAL_SPLIT][cfg.data.text_column]]
    train_dataset = prepare_dataset(splits[TRAIN_SPLIT], processor, cfg.data)
    val_dataset = prepare_dataset(splits[VAL_SPLIT], processor, cfg.data)

    # ---- Model ----------------------------------------------------------
    model = apply_lora(load_model(cfg.model), cfg.lora)

    # ---- Training -------------------------------------------------------
    log_step_estimation(n_train, cfg.training)
    trainer = build_trainer(model, processor, train_dataset, val_dataset, cfg.training, output_dir)

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()

    # ---- Save artefacts (best model, thanks to load_best_model_at_end) ---
    lora_dir = output_dir / "lora_weights"
    processor_dir = output_dir / "processor"
    trainer.model.save_pretrained(lora_dir)
    processor.save_pretrained(processor_dir)
    log_section("SAVED", {"LoRA weights": lora_dir, "Processor": processor_dir})

    # ---- Detailed validation --------------------------------------------
    evaluate_per_phrase(
        trainer.model,
        processor,
        val_dataset,
        references,
        cfg.model,
        cfg.training.batch_size,
        output_dir,
    )

    log_section("DONE", {"Output": output_dir})
    return 0


if __name__ == "__main__":
    sys.exit(main())
