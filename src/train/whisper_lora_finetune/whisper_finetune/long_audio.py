# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Handling of recordings longer than Whisper's 30-second input window.

Whisper only sees the first 30 s of each file. If a longer file is used for
training, the label contains text that is not in the (truncated) audio and the
model learns to hallucinate. Three strategies are available:

* ``segment`` (default): cut the recording into aligned segments (see
  `segmentation.py`) using Whisper word timestamps. The Whisper output is cached
  in ``<audio_dir>/<segments_subdir>/asr_cache.json`` so the slow ASR pass runs
  once per file; alignment and cutting are recomputed on every run, so the
  segmentation options can be changed without re-transcribing.
* ``drop``: discard long recordings.
* ``keep``: leave them untouched (original behaviour, not recommended).
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from datasets import Dataset

from .config import SAMPLING_RATE, SUPPORTED_MODELS, DataConfig
from .segmentation import Word, align_reference, choose_segments
from .utils import detect_device

logger = logging.getLogger(__name__)

LONG_AUDIO_MODES = ("segment", "drop", "keep")
SOURCE_COLUMN = "source_audio"
# True for rows that come from a recording longer than `max_audio_seconds`.
LONG_COLUMN = "from_long_audio"
WHISPER_WINDOW_SECONDS = 30.0

Transcriber = Callable[[np.ndarray], list[Word]]


# ---------------------------------------------------------------------------
# Audio and ASR helpers
# ---------------------------------------------------------------------------

def audio_duration(path: str) -> float:
    """Duration in seconds without decoding the whole file."""
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:  # noqa: BLE001 - formats not supported by libsndfile (e.g. some mp3)
        import librosa

        return float(librosa.get_duration(path=path))


def load_audio(path: str) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True)
    return audio


def build_transcriber(model_size: str, language: str, device: str | None, batch_size: int) -> tuple[Transcriber, object]:
    """Return a Whisper ASR function producing word-level timestamps, and the pipeline object."""
    import torch
    from transformers import pipeline

    device = device or detect_device()
    asr = pipeline(
        "automatic-speech-recognition",
        model=SUPPORTED_MODELS[model_size],
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    logger.info("Segmentation ASR model %s on %s", SUPPORTED_MODELS[model_size], device)

    def transcribe(audio: np.ndarray) -> list[Word]:
        result = asr(
            {"raw": audio, "sampling_rate": SAMPLING_RATE},
            chunk_length_s=30,
            batch_size=batch_size,
            return_timestamps="word",
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        return [
            Word(chunk["text"], chunk["timestamp"][0], chunk["timestamp"][1])
            for chunk in result.get("chunks", [])
        ]

    return transcribe, asr


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class LongAudioHandler:
    """Apply the configured long-audio strategy to a dataset of resolved audio paths."""

    def __init__(self, cfg: DataConfig, language: str, transcriber: Transcriber | None = None):
        if cfg.long_audio not in LONG_AUDIO_MODES:
            raise ValueError(f"long_audio must be one of {LONG_AUDIO_MODES}")
        self.cfg = cfg
        self.language = language
        self._transcriber = transcriber
        self._pipeline = None
        self._report: list[dict] = []
        self.segments_dir = Path(cfg.audio_dir) / cfg.segments_subdir
        self._cache_path = self.segments_dir / "asr_cache.json"

    # -- public API -----------------------------------------------------------

    def process(self, dataset: Dataset, source_name: str) -> Dataset:
        """Apply the long-audio strategy.

        Every row derived from a long recording (a segment, or the whole file
        with ``keep``) gets ``LONG_COLUMN = True`` so the split can keep it out
        of validation when ``long_audio_train_only`` is set.
        """
        cfg = self.cfg
        durations = [audio_duration(p) for p in dataset[cfg.audio_column]]
        long_idx = {i for i, d in enumerate(durations) if d > cfg.max_audio_seconds}
        if not long_idx:
            logger.info("No recordings longer than %.1f s in %s.", cfg.max_audio_seconds, source_name)
            return dataset

        if cfg.long_audio == "keep":
            logger.warning(
                "Keeping %d recordings longer than %.1f s from %s as is (Whisper truncates them at 30 s)%s.",
                len(long_idx), cfg.max_audio_seconds, source_name,
                "; they will only be used for training" if cfg.long_audio_train_only else "",
            )
            return dataset.add_column(LONG_COLUMN, [i in long_idx for i in range(len(dataset))])

        if cfg.long_audio == "drop":
            for i in sorted(long_idx):
                logger.info("  long audio dropped: %s (%.1f s)", dataset[i][cfg.audio_column], durations[i])
            logger.warning("Dropped %d recordings longer than %.1f s from %s.", len(long_idx), cfg.max_audio_seconds, source_name)
            return dataset.select([i for i in range(len(dataset)) if i not in long_idx])

        logger.info("Segmenting %d recordings longer than %.1f s from %s ...", len(long_idx), cfg.max_audio_seconds, source_name)
        rows, n_kept, n_total = [], 0, 0
        for i, row in enumerate(dataset.to_list()):
            source = row[cfg.audio_column]
            if i not in long_idx:
                rows.append({**row, SOURCE_COLUMN: source, LONG_COLUMN: False})
                continue
            for seg in self._segments_for(source, str(row[cfg.text_column]).strip()):
                n_total += 1
                if seg["match_ratio"] >= cfg.segment_min_match and seg["duration"] <= WHISPER_WINDOW_SECONDS:
                    n_kept += 1
                    rows.append(
                        {
                            **row,
                            cfg.audio_column: seg["path"],
                            cfg.text_column: seg["text"],
                            SOURCE_COLUMN: source,
                            LONG_COLUMN: True,
                        }
                    )
                else:
                    logger.warning(
                        "  segment skipped: match=%.2f < %.2f (reference words recognised by %s) | %.1f s | %s | %s",
                        seg["match_ratio"], cfg.segment_min_match, cfg.segment_model,
                        seg["duration"], seg["path"], seg["text"],
                    )

        logger.info(
            "%s: %d long recordings -> %d segments kept (of %d); %d rows in total.",
            source_name, len(long_idx), n_kept, n_total, len(rows),
        )
        return Dataset.from_list(rows)

    def report(self) -> pd.DataFrame:
        report = pd.DataFrame(self._report)
        if len(report):
            report["kept"] = (report["match_ratio"] >= self.cfg.segment_min_match) & (
                report["duration"] <= WHISPER_WINDOW_SECONDS
            )
        return report

    def close(self) -> None:
        """Release the ASR model before training starts."""
        if self._pipeline is None:
            return
        self._pipeline = None
        self._transcriber = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # -- internals ------------------------------------------------------------

    def _get_transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber, self._pipeline = build_transcriber(
                self.cfg.segment_model, self.language, self.cfg.segment_device, self.cfg.segment_batch_size
            )
        return self._transcriber

    def _cache_key(self, path: str) -> str:
        """The ASR output only depends on the audio file, the model and the language."""
        stat = os.stat(path)
        return "|".join(
            map(str, (os.path.abspath(path), stat.st_size, int(stat.st_mtime), self.cfg.segment_model, self.language))
        )

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        return {}

    def _asr_words(self, path: str, audio: np.ndarray) -> list[Word]:
        key = self._cache_key(path)
        cache = self._load_cache()
        if key in cache:
            logger.info("  %s: Whisper timestamps loaded from cache", path)
            return [Word(*w) for w in cache[key]]
        logger.info("  %s: transcribing with %s for word timestamps ...", path, self.cfg.segment_model)
        words = self._get_transcriber()(audio)
        cache[key] = [[w.text, w.start, w.end] for w in words]
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return words

    def _segments_for(self, path: str, reference: str) -> list[dict]:
        import soundfile as sf

        audio = load_audio(path)
        total = len(audio) / SAMPLING_RATE
        words = align_reference(reference, self._asr_words(path, audio), total, self.cfg.segment_word_similarity)
        segments = choose_segments(words, total, self.cfg.segment_max_seconds, self.cfg.segment_min_seconds)

        self.segments_dir.mkdir(parents=True, exist_ok=True)
        result = []
        for k, seg in enumerate(segments, start=1):
            seg_path = self.segments_dir / f"{Path(path).stem}_seg{k:02d}.wav"
            sf.write(seg_path, audio[int(seg.start * SAMPLING_RATE) : int(seg.end * SAMPLING_RATE)], SAMPLING_RATE)
            result.append(
                {
                    SOURCE_COLUMN: path,
                    "path": str(seg_path),
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "duration": round(seg.duration, 2),
                    "match_ratio": round(seg.match_ratio, 3),
                    "text": seg.text,
                }
            )
            logger.info("  %s  %.1f-%.1f s  match=%.2f  %s", seg_path.name, seg.start, seg.end, seg.match_ratio, seg.text)

        self._report.extend(result)
        return result
