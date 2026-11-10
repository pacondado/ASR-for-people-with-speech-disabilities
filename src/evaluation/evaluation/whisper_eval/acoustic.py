# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Acoustic similarity between the TTS of the reference and of the
hypothesis.

Both texts are synthesised with the SAME voice, so any difference in
the audio reflects a difference in the phone sequence rather than in
speaker or prosody. MFCCs are aligned with DTW and the average
alignment cost is turned into a similarity score in [0, 1].

The score has no absolute meaning — it is only useful for comparing
BASE against LoRA under identical conditions.
"""

import asyncio
import hashlib
import os
import sys

import numpy as np
import librosa

try:
    import edge_tts
except ImportError:
    edge_tts = None


class AcousticSimilarity:
    """TTS-backed acoustic similarity with an on-disk synthesis cache."""

    @classmethod
    def from_settings(cls, settings):
        """Builds the scorer straight from a Settings object."""

        return cls(
            enabled=settings.ENABLE_ACOUSTIC,
            voice=settings.TTS_VOICE,
            cache_dir=settings.TTS_CACHE_DIR,
            sample_rate=settings.SAMPLE_RATE,
            n_mfcc=settings.N_MFCC,
        )

    def __init__(
        self,
        enabled,
        voice,
        cache_dir,
        sample_rate,
        n_mfcc,
    ):
        self.voice = voice

        self.cache_dir = cache_dir

        self.sample_rate = sample_rate

        self.n_mfcc = n_mfcc

        if enabled and edge_tts is None:
            print(
                "WARNING: edge-tts is not installed — acoustic similarity "
                "disabled.\n         pip install edge-tts",
                file=sys.stderr,
            )

        self.available = enabled and edge_tts is not None

        if self.available:
            os.makedirs(self.cache_dir, exist_ok=True)

    # --------------------------------------------------------
    # TTS
    # --------------------------------------------------------

    def _cache_path(self, text):
        """Deterministic cache path for a (voice, text) pair."""

        key = hashlib.sha1(
            f"{self.voice}||{text}".encode("utf-8")
        ).hexdigest()

        return os.path.join(self.cache_dir, f"{key}.mp3")

    async def _synthesize(self, text, path):
        communicate = edge_tts.Communicate(text, self.voice)

        await communicate.save(path)

    def _ensure_tts(self, text):
        """Synthesises the text unless already cached. Returns the path."""

        if not self.available or not text:
            return None

        path = self._cache_path(text)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path

        try:
            asyncio.run(self._synthesize(text, path))

        except Exception as error:
            print(f"  WARNING: TTS failed ({error})", file=sys.stderr)

            if os.path.exists(path):
                os.remove(path)

            return None

        return path

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    def _mfcc(self, path):
        """Per-coefficient normalised MFCCs (CMVN), shape (n_mfcc, frames)."""

        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)

        audio, _ = librosa.effects.trim(audio, top_db=30)

        if audio.size < self.sample_rate // 20:
            return None

        features = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
        )

        mean = features.mean(axis=1, keepdims=True)

        std = features.std(axis=1, keepdims=True) + 1e-8

        return (features - mean) / std

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    def similarity(self, reference, hypothesis):
        """
        Similarity in [0, 1] between the TTS of the reference and the TTS
        of the hypothesis. Returns None when it cannot be computed.

        Both texts are expected to be normalised already.
        """

        if not self.available:
            return None

        if not hypothesis:
            return 0.0

        if hypothesis == reference:
            return 1.0

        reference_audio = self._ensure_tts(reference)

        hypothesis_audio = self._ensure_tts(hypothesis)

        if reference_audio is None or hypothesis_audio is None:
            return None

        try:
            reference_features = self._mfcc(reference_audio)

            hypothesis_features = self._mfcc(hypothesis_audio)

        except Exception as error:
            print(
                f"  WARNING: failed to read TTS audio ({error})",
                file=sys.stderr,
            )
            return None

        if reference_features is None or hypothesis_features is None:
            return None

        cost_matrix, path = librosa.sequence.dtw(
            X=reference_features,
            Y=hypothesis_features,
            metric="cosine",
        )

        # Accumulated cost normalised by the length of the warping path.
        # With cosine distance, the per-frame cost lives in [0, 2].
        normalized_cost = float(cost_matrix[-1, -1] / len(path))

        return float(np.clip(1.0 - normalized_cost / 2.0, 0.0, 1.0))
