# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Command line interface.

    python src/evaluation/evaluate.py \\
        --model output_whisper/lora_weights \\
        --data data/test
"""

import argparse
import os

from . import config
from .settings import AUDIO_SUBDIR, MANIFEST_NAME, Settings, default_results_file


def build_parser():
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Evaluate a LoRA-adapted Whisper model against its base "
            "model, per speaker, reporting WER, CER, PER and an "
            "optional acoustic similarity score."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The data directory is expected to contain:\n"
            f"  {MANIFEST_NAME}          manifest with 'audio' and 'text' columns\n"
            f"  {AUDIO_SUBDIR}/<speaker>/  one audio subdirectory per speaker\n"
            "\n"
            "Example:\n"
            "  python src/evaluation/evaluate.py \\\n"
            "      --model output_whisper/lora_weights \\\n"
            "      --data data/test\n"
        ),
    )

    # --------------------------------------------------------
    # Required-ish paths
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        dest="LORA_PATH",
        default=config.LORA_PATH,
        metavar="DIR",
        help=(
            "Directory of the trained LoRA adapter "
            f"(default: {config.LORA_PATH})"
        ),
    )

    parser.add_argument(
        "--data",
        dest="DATA_DIR",
        default=config.DATA_DIR,
        metavar="DIR",
        help=(
            f"Directory holding {MANIFEST_NAME} and {AUDIO_SUBDIR}/ "
            f"(default: {config.DATA_DIR})"
        ),
    )

    parser.add_argument(
        "--output",
        dest="RESULTS_FILE",
        default=None,
        metavar="FILE",
        help=(
            "Where to write the results JSON (default: "
            f"{os.path.dirname(config.RESULTS_FILE) or '.'}/<model folder name>.json)"
        ),
    )

    # --------------------------------------------------------
    # Overrides for the layout, when it differs
    # --------------------------------------------------------

    parser.add_argument(
        "--test-csv",
        dest="TEST_CSV",
        default=None,
        metavar="FILE",
        help=f"Manifest path, if not <data>/{MANIFEST_NAME}",
    )

    parser.add_argument(
        "--audio-dir",
        dest="AUDIO_DIR",
        default=None,
        metavar="DIR",
        help=f"Audio root, if not <data>/{AUDIO_SUBDIR}",
    )

    # --------------------------------------------------------
    # Model / decoding
    # --------------------------------------------------------

    parser.add_argument(
        "--base-model",
        "--model-base",
        dest="MODEL_NAME",
        default=None,
        metavar="NAME",
        help=(
            "Base Whisper model: "
            + ", ".join(config.SUPPORTED_MODELS)
            + ", or any Hugging Face id / local path. Must match the "
            f"model the LoRA adapter was trained on (default: {config.MODEL_NAME})"
        ),
    )

    parser.add_argument(
        "--language",
        dest="LANGUAGE",
        default=None,
        help=f"Decoding language (default: {config.LANGUAGE})",
    )

    parser.add_argument(
        "--device",
        dest="DEVICE",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force a device instead of auto-detecting",
    )

    parser.add_argument(
        "--max-length",
        dest="MAX_LENGTH",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Total decoder length, prompt included "
            f"(default: {config.MAX_LENGTH})"
        ),
    )

    # --------------------------------------------------------
    # Speakers
    # --------------------------------------------------------

    parser.add_argument(
        "--speakers",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Restrict evaluation to these speakers. Defaults to every "
            "speaker configured, plus any extra subdirectory found "
            "under the audio root."
        ),
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    parser.add_argument(
        "--normalize",
        dest="NORMALIZE_TEXT",
        action="store_true",
        default=None,
        help=(
            "Compute WER and CER on normalised text (lowercase, no "
            "punctuation). PER always uses normalised text."
        ),
    )

    parser.add_argument(
        "--no-per",
        dest="ENABLE_PHONETIC",
        action="store_false",
        default=None,
        help="Skip the phoneme error rate",
    )

    parser.add_argument(
        "--g2p",
        dest="G2P_PREFERENCE",
        nargs="+",
        default=None,
        choices=["epitran", "rules"],
        metavar="BACKEND",
        help=(
            "G2P backends in order of preference "
            f"(default: {' '.join(config.G2P_PREFERENCE)})"
        ),
    )

    acoustic = parser.add_mutually_exclusive_group()

    acoustic.add_argument(
        "--acoustic",
        dest="ENABLE_ACOUSTIC",
        action="store_true",
        default=None,
        help="Enable TTS + DTW acoustic similarity (needs network access)",
    )

    acoustic.add_argument(
        "--no-acoustic",
        dest="ENABLE_ACOUSTIC",
        action="store_false",
        default=None,
        help="Disable acoustic similarity",
    )

    parser.add_argument(
        "--tts-voice",
        dest="TTS_VOICE",
        default=None,
        metavar="VOICE",
        help=f"edge-tts voice (default: {config.TTS_VOICE})",
    )

    return parser


# ============================================================
# SPEAKER RESOLUTION
# ============================================================

def resolve_participants(settings, requested):
    """
    Decides which speakers to evaluate.

    Without --speakers, this is the configured set plus any extra
    subdirectory found under the audio root, so a new speaker is picked
    up by dropping their audio in place. With --speakers, only the
    named ones are kept, and an unknown name is an error rather than a
    silent skip.
    """

    configured = dict(settings.PARTICIPANTS)

    discovered = settings.discovered_speakers()

    if requested:
        unknown = [
            name
            for name in requested
            if name not in configured and name not in discovered
        ]

        if unknown:
            raise ValueError(
                "Unknown speaker(s): "
                + ", ".join(unknown)
                + "\nAvailable under "
                + settings.AUDIO_DIR
                + ": "
                + (", ".join(discovered) if discovered else "(none)")
            )

        return {
            name: configured.get(name, {"description": ""})
            for name in requested
        }

    participants = dict(configured)

    for name in discovered:
        participants.setdefault(name, {"description": ""})

    return participants


# ============================================================
# ENTRY POINT HELPER
# ============================================================

def parse_settings(argv=None):
    """Parses argv into a validated Settings object."""

    parser = build_parser()

    args = parser.parse_args(argv)

    requested_speakers = args.speakers

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "speakers" and value is not None
    }

    settings = Settings.from_defaults(**overrides)

    # Without --output, the JSON is named after the model folder.
    if args.RESULTS_FILE is None:
        settings = settings.with_overrides(
            RESULTS_FILE=default_results_file(settings.LORA_PATH)
        )

    settings = settings.with_overrides(
        PARTICIPANTS=resolve_participants(settings, requested_speakers)
    )

    return settings.validate()
