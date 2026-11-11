# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Default configuration.

Every value here can be overridden on the command line; see cli.py or
run `python src/evaluation/evaluate.py --help`. The defaults are what
you get when a flag is omitted.
"""

# ============================================================
# MODELS
# ============================================================

# Short names accepted by --base-model, mapped to their Hugging Face Hub
# checkpoints (same names as in the fine-tuning script). Any other value
# is used as given: a full Hub id or a local model directory.
SUPPORTED_MODELS = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}

# Overridden by --base-model. Must be the model the LoRA adapter was
# trained on.
MODEL_NAME = "large-v3"

# Overridden by --model. This is the directory holding the LoRA
# adapter (adapter_config.json + adapter_model.safetensors).
LORA_PATH = "output_whisper/lora_weights"


# ============================================================
# DATA
# ============================================================

# Overridden by --data. Expected to contain test.csv and audio/.
DATA_DIR = "data/test"

# Overridden by --output.
RESULTS_FILE = "output_whisper/base_vs_lora_by_person.json"


# ============================================================
# DECODING
# ============================================================

LANGUAGE = "pt"

TASK = "transcribe"

SAMPLE_RATE = 16000

# Total decoder length, prompt tokens included. Whisper's decoder can
# hold 448 positions (config.max_target_positions), which is the value
# the model itself ships. Using max_length instead of max_new_tokens is
# deliberate — see asr.configure_generation().
MAX_LENGTH = 448

NO_REPEAT_NGRAM_SIZE = 3


# ============================================================
# PHONETIC METRICS
# ============================================================
#
# PER (Phoneme Error Rate)
#   Converts reference and hypothesis into phonemes and computes the
#   error rate over that sequence. Captures the cases where the text
#   comes out different but the sounds are the same ("casa" / "caza").
#   Requires: pip install epitran — or nothing, with the rule-based G2P.
#
# Acoustic similarity
#   Synthesises both texts with the SAME voice (edge-tts), extracts
#   MFCCs and aligns the sequences with DTW. Requires network access.
#   Requires: pip install edge-tts
#
# Either one can be turned off; the script degrades with a warning if
# the dependency is unavailable.
# ============================================================

ENABLE_PHONETIC = True

# Overridden by --normalize. When True, WER and CER are computed on the
# normalised texts (lowercase, no punctuation), so "Irão"/"irão" or an
# extra comma do not count as errors. PER always uses normalised text.
NORMALIZE_TEXT = False

# G2P backends in order of preference; the first one that works is
# used. Both are pure Python — neither needs external programs
# installed on the system:
#
#   "epitran"  pip install epitran   (curated maps, recommended)
#   "rules"    embedded in g2p.py    (nothing to install)
#
# To install nothing at all, set G2P_PREFERENCE = ["rules"].
G2P_PREFERENCE = ["epitran", "rules"]

EPITRAN_CODE = "por-Latn"

ENABLE_ACOUSTIC = False

TTS_VOICE = "pt-PT-RaquelNeural"

TTS_CACHE_DIR = "output_whisper/tts_cache"

N_MFCC = 13


# ============================================================
# PARTICIPANTS
# ============================================================
#
# Maps a speaker to the description used in the report. The key must
# match a subdirectory name under <data>/audio/.
#
# Speakers present in the data but missing here are still evaluated,
# with an empty description, unless --speakers restricts the set.
# ============================================================

PARTICIPANTS = {}
