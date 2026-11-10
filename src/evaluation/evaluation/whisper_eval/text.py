# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Text normalisation shared by every metric.

Texts are normalised before the G2P step and before TTS, so that
punctuation invented by the model does not contaminate the phoneme
sequence or the synthesised audio.
"""

import re
import string

PUNCTUATION = str.maketrans("", "", string.punctuation + "«»“”…—–¡¿")


def normalize_text(text):
    """Lowercase, punctuation stripped, whitespace collapsed."""

    if not text:
        return ""

    text = text.lower().strip()

    text = text.translate(PUNCTUATION)

    text = re.sub(r"\s+", " ", text)

    return text.strip()
