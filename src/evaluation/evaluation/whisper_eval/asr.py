# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Whisper model loading and transcription.

Every function takes the `Settings` object rather than reading module
constants, so a command line override reaches the decoder.
"""

import gc
import os

import torch
import librosa

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from peft import PeftModel

from .settings import resolve_model_name


# ============================================================
# DEVICE
# ============================================================

def pick_device(preferred=None):
    """cuda > mps > cpu, unless `preferred` forces one."""

    if preferred:
        return preferred

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


# ============================================================
# MEMORY
# ============================================================

def cleanup_memory():
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ============================================================
# AUDIO
# ============================================================

def load_wav(path, sample_rate):
    audio, actual_sr = librosa.load(
        path,
        sr=sample_rate,
        mono=True,
    )

    return audio


# ============================================================
# GENERATION CONFIGURATION
# ============================================================

def configure_generation(model, settings):
    """
    Prepares a model's generation config for deterministic decoding.

    Whisper ships `max_length` in its generation config and passes it
    down to the generic `generate()`, so also supplying
    `max_new_tokens` makes transformers emit this on every call:

        Both `max_new_tokens` (=225) and `max_length`(=448) seem to
        have been set. `max_new_tokens` will take precedence.

    Clearing `max_length` does NOT help: Whisper rebuilds its
    generation config internally and the field comes back filled with
    the GenerationConfig class default, so the warning returns as
    `max_length`(=20).

    The reliable fix is to drop `max_new_tokens` entirely and bound the
    output with `max_length` alone. The warning is guarded by
    `if generation_config.max_new_tokens is not None`, so with it set
    to None the warning is unreachable regardless of what Whisper does
    to `max_length` afterwards.

    Note that `max_length` counts the whole decoder sequence, prompt
    tokens included, whereas `max_new_tokens` counted only generated
    ones. With 448 positions against utterances of a few seconds, the
    cap is never the binding constraint either way.
    """

    model.config.forced_decoder_ids = None

    model.generation_config.forced_decoder_ids = None

    model.generation_config.pad_token_id = model.config.pad_token_id

    model.generation_config.eos_token_id = model.config.eos_token_id

    # max_length alone bounds the output; max_new_tokens must stay None
    # or transformers warns about the two of them clashing.
    model.generation_config.max_length = settings.MAX_LENGTH

    model.generation_config.max_new_tokens = None


# ============================================================
# LOADING
# ============================================================

def load_processor(settings):
    processor = WhisperProcessor.from_pretrained(
        resolve_model_name(settings.MODEL_NAME),
        language=settings.LANGUAGE,
        task=settings.TASK,
    )

    # Whisper's feature extractor omits the attention mask by default
    processor.feature_extractor.return_attention_mask = True

    return processor


def load_base_model(settings, device):
    model = WhisperForConditionalGeneration.from_pretrained(
        resolve_model_name(settings.MODEL_NAME)
    )

    configure_generation(model, settings)

    model.to(device)
    model.eval()

    return model


def load_lora_model(settings, device):
    """Loads a second base model and applies the LoRA adapter to it."""

    if not os.path.exists(settings.LORA_PATH):
        raise FileNotFoundError(
            f"LoRA path not found:\n{settings.LORA_PATH}"
        )

    base = WhisperForConditionalGeneration.from_pretrained(
        resolve_model_name(settings.MODEL_NAME)
    )

    model = PeftModel.from_pretrained(
        base,
        settings.LORA_PATH,
        is_trainable=False,
    )

    configure_generation(model, settings)

    model.to(device)
    model.eval()

    return model


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_input(processor, audio, settings, device):
    """Log-mel features plus a validated attention mask, on `device`."""

    features = processor(
        audio,
        sampling_rate=settings.SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = features.input_features

    attention_mask = getattr(features, "attention_mask", None)

    expected_shape = (input_features.shape[0], input_features.shape[-1])

    if attention_mask is None:
        print("WARNING: processor did not return attention_mask.")

        attention_mask = torch.ones(expected_shape, dtype=torch.long)

    else:
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)

        if (
            attention_mask.ndim != 2
            or attention_mask.shape[0] != expected_shape[0]
            or attention_mask.shape[1] != expected_shape[1]
        ):
            print("WARNING: processor returned unexpected attention_mask shape.")
            print("Received:", attention_mask.shape)
            print("Expected:", expected_shape)

            attention_mask = torch.ones(expected_shape, dtype=torch.long)

    input_features = input_features.to(device)

    attention_mask = attention_mask.to(device=device, dtype=torch.long)

    return input_features, attention_mask


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe(model, processor, audio, settings, device):
    """Greedy transcription of a single utterance."""

    input_features = None
    attention_mask = None
    generated_ids = None

    try:
        input_features, attention_mask = prepare_input(
            processor,
            audio,
            settings,
            device,
        )

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                language=settings.LANGUAGE,
                task=settings.TASK,
                do_sample=False,
                no_repeat_ngram_size=settings.NO_REPEAT_NGRAM_SIZE,
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return text

    finally:
        if generated_ids is not None:
            del generated_ids

        if input_features is not None:
            del input_features

        if attention_mask is not None:
            del attention_mask

        cleanup_memory()
