# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Model and processor loading, generation setup and LoRA wrapping."""

from __future__ import annotations

import logging

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .config import LoraSettings, ModelConfig

logger = logging.getLogger(__name__)


def load_processor(cfg: ModelConfig) -> WhisperProcessor:
    logger.info("Loading processor: %s", cfg.checkpoint)
    return WhisperProcessor.from_pretrained(cfg.checkpoint, language=cfg.language, task=cfg.task)


def load_model(cfg: ModelConfig) -> WhisperForConditionalGeneration:
    logger.info("Loading model: %s", cfg.checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.checkpoint)
    configure_generation(model, cfg)
    return model


def configure_generation(model: WhisperForConditionalGeneration, cfg: ModelConfig) -> None:
    """Set language/task and deterministic decoding.

    `forced_decoder_ids` is cleared on purpose: recent Transformers versions
    derive the prompt from `language`/`task` and warn (or fail) when both are set.
    """
    gen = model.generation_config
    gen.language = cfg.language
    gen.task = cfg.task
    gen.do_sample = False
    gen.return_timestamps = False
    gen.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    logger.info(
        "Generation config -> language=%s task=%s do_sample=%s forced_decoder_ids=%s",
        gen.language,
        gen.task,
        gen.do_sample,
        gen.forced_decoder_ids,
    )


def apply_lora(model: WhisperForConditionalGeneration, cfg: LoraSettings) -> PeftModel:
    logger.info(
        "Applying LoRA (r=%d, alpha=%d, dropout=%.3f, targets=%s)",
        cfg.r,
        cfg.alpha,
        cfg.dropout,
        cfg.target_modules,
    )
    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
