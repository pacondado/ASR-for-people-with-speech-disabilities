# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Runtime settings.

`config.py` holds the defaults; this module turns them into an
immutable object that every other module receives, so the same code
path works whether a value came from the defaults or from the command
line.

Field names are upper case to match the constants they replace, which
keeps the reporting and serialisation code unchanged.
"""

import os

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from . import config


# ============================================================
# LAYOUT OF A DATA DIRECTORY
# ============================================================
#
# A data directory passed with --data is expected to look like this:
#
#     data/test/
#       ├── test.csv          manifest with the 'audio' and 'text' columns
#       └── audio/
#           ├── Condado/      one subdirectory per speaker
#           └── Lobo/
#
# ============================================================

MANIFEST_NAME = "test.csv"

AUDIO_SUBDIR = "audio"


def resolve_model_name(name):
    """
    Maps a short model name to its Hugging Face Hub checkpoint.

    The short names in config.SUPPORTED_MODELS are matched leniently:
    case, '_' instead of '-', and an 'openai/' or 'whisper-' prefix are
    ignored, so "large-v3", "Large-V3", "large_v3", "whisper-large-v3"
    and "openai/whisper-large-v3" all give the same checkpoint.

    Any other Hub id ("org/model") or an existing local directory is
    returned unchanged. Anything else is an error that lists the
    accepted short names.
    """

    if name is None:
        return None

    name = str(name).strip()

    key = name.lower().replace("_", "-")

    for prefix in ("openai/", "whisper-"):
        if key.startswith(prefix):
            key = key[len(prefix):]

    if key in config.SUPPORTED_MODELS:
        return config.SUPPORTED_MODELS[key]

    if "/" in name or os.path.isdir(name):
        return name

    raise ValueError(
        f"Unknown base model '{name}'. Use one of: "
        + ", ".join(config.SUPPORTED_MODELS)
        + ", a Hugging Face id (e.g. openai/whisper-medium) or a local "
        "model directory."
    )



def default_results_file(lora_path):
    """
    Results JSON named after the evaluated model, used when --output is
    omitted. It goes into the directory of config.RESULTS_FILE.

    The name is the training run folder, so the JSON can be matched to
    the model that produced it:

        .../output_lora_medium_r16_a32_lr6e-05_715_50_20261029-190352/lora_weights
            -> output_lora_medium_r16_a32_lr6e-05_715_50_20261029-190352.json

        .../output_lora_medium_..._20261029-190352/checkpoint-358
            -> output_lora_medium_..._20261029-190352_checkpoint-358.json

    Any other adapter directory gives <directory name>.json.
    """

    path = os.path.normpath(os.path.abspath(lora_path))

    name = os.path.basename(path)

    parent = os.path.basename(os.path.dirname(path))

    if name == "lora_weights" and parent:
        name = parent

    elif name.startswith("checkpoint-") and parent:
        name = f"{parent}_{name}"

    results_dir = os.path.dirname(config.RESULTS_FILE)

    return os.path.join(results_dir, f"{name}.json")

@dataclass(frozen=True)
class Settings:

    # ---- models ----
    MODEL_NAME: str
    LORA_PATH: str

    # ---- data ----
    DATA_DIR: str
    TEST_CSV: str
    AUDIO_DIR: str
    RESULTS_FILE: str

    # ---- decoding ----
    LANGUAGE: str
    TASK: str
    SAMPLE_RATE: int
    MAX_LENGTH: int
    NO_REPEAT_NGRAM_SIZE: int

    # ---- phonetic metrics ----
    ENABLE_PHONETIC: bool
    NORMALIZE_TEXT: bool
    G2P_PREFERENCE: List[str]
    EPITRAN_CODE: str

    # ---- acoustic metrics ----
    ENABLE_ACOUSTIC: bool
    TTS_VOICE: str
    TTS_CACHE_DIR: str
    N_MFCC: int

    # ---- speakers ----
    PARTICIPANTS: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # ---- runtime ----
    DEVICE: Optional[str] = None

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------

    @classmethod
    def from_defaults(cls, **overrides):
        """Settings built from config.py, with optional overrides."""

        data_dir = overrides.pop("DATA_DIR", config.DATA_DIR)

        settings = cls(
            MODEL_NAME=resolve_model_name(config.MODEL_NAME),
            LORA_PATH=config.LORA_PATH,
            DATA_DIR=data_dir,
            TEST_CSV=os.path.join(data_dir, MANIFEST_NAME),
            AUDIO_DIR=os.path.join(data_dir, AUDIO_SUBDIR),
            RESULTS_FILE=config.RESULTS_FILE,
            LANGUAGE=config.LANGUAGE,
            TASK=config.TASK,
            SAMPLE_RATE=config.SAMPLE_RATE,
            MAX_LENGTH=config.MAX_LENGTH,
            NO_REPEAT_NGRAM_SIZE=config.NO_REPEAT_NGRAM_SIZE,
            ENABLE_PHONETIC=config.ENABLE_PHONETIC,
            NORMALIZE_TEXT=config.NORMALIZE_TEXT,
            G2P_PREFERENCE=list(config.G2P_PREFERENCE),
            EPITRAN_CODE=config.EPITRAN_CODE,
            ENABLE_ACOUSTIC=config.ENABLE_ACOUSTIC,
            TTS_VOICE=config.TTS_VOICE,
            TTS_CACHE_DIR=config.TTS_CACHE_DIR,
            N_MFCC=config.N_MFCC,
            PARTICIPANTS=dict(config.PARTICIPANTS),
        )

        return settings.with_overrides(**overrides)

    def with_overrides(self, **overrides):
        """A copy with some fields replaced, keeping derived paths in sync."""

        overrides = {
            key: value
            for key, value in overrides.items()
            if value is not None
        }

        if "MODEL_NAME" in overrides:
            overrides["MODEL_NAME"] = resolve_model_name(
                overrides["MODEL_NAME"]
            )

        # Changing the data directory moves the manifest and the audio
        # with it, unless either was given explicitly.
        if "DATA_DIR" in overrides:
            data_dir = overrides["DATA_DIR"]

            overrides.setdefault(
                "TEST_CSV", os.path.join(data_dir, MANIFEST_NAME)
            )

            overrides.setdefault(
                "AUDIO_DIR", os.path.join(data_dir, AUDIO_SUBDIR)
            )

        return replace(self, **overrides)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate(self):
        """Fails early, with a message that says which path is wrong."""

        if not os.path.isdir(self.DATA_DIR):
            raise NotADirectoryError(
                f"Data directory not found: {self.DATA_DIR}"
            )

        if not os.path.isfile(self.TEST_CSV):
            raise FileNotFoundError(
                f"Manifest not found: {self.TEST_CSV}\n"
                f"Expected '{MANIFEST_NAME}' inside {self.DATA_DIR}"
            )

        if not os.path.isdir(self.AUDIO_DIR):
            raise NotADirectoryError(
                f"Audio directory not found: {self.AUDIO_DIR}\n"
                f"Expected an '{AUDIO_SUBDIR}' subdirectory inside "
                f"{self.DATA_DIR}"
            )

        if not os.path.isdir(self.LORA_PATH):
            raise NotADirectoryError(
                f"Model directory not found: {self.LORA_PATH}"
            )

        if not self.PARTICIPANTS:
            raise ValueError("No speakers selected.")

        return self

    # --------------------------------------------------------
    # Speaker discovery
    # --------------------------------------------------------

    def discovered_speakers(self):
        """Speaker subdirectories actually present under AUDIO_DIR."""

        if not os.path.isdir(self.AUDIO_DIR):
            return []

        return sorted(
            name
            for name in os.listdir(self.AUDIO_DIR)
            if os.path.isdir(os.path.join(self.AUDIO_DIR, name))
        )
