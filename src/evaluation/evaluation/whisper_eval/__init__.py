# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Whisper BASE vs LoRA evaluation toolkit.

Modules:

    config      defaults for every tunable
    cli         argparse front end (--model, --data, ...)
    settings    runtime Settings object handed to the other modules
    dataset     test manifest loading and validation
    asr         model loading, generation config, transcription
    text        text normalisation shared by the metrics
    g2p         grapheme-to-phoneme backends (epitran / rules)
    acoustic    TTS + MFCC + DTW similarity
    metrics     per-sample scoring and aggregation
    display     console report
    results     results JSON assembly and writing

The entry point is evaluate.py, next to this package.
"""

__author__ = "Paulo Condado"
__license__ = "MIT"

__all__ = [
    "acoustic",
    "asr",
    "cli",
    "config",
    "dataset",
    "display",
    "g2p",
    "metrics",
    "results",
    "settings",
    "text",
]
