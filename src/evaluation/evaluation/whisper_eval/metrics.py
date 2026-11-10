# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Per-sample scoring and aggregation of WER, CER, PER and acoustic
similarity.

Two flavours of average are reported throughout:

  mean_phrase_*  the mean of the per-utterance scores, so every
                 utterance weighs the same;
  corpus_*       total errors divided by total reference units, so
                 long utterances weigh more. This is the conventional
                 way to report a single figure.
"""

import numpy as np

from jiwer import wer as calc_wer
from jiwer import cer as calc_cer

from .text import normalize_text


# ============================================================
# HELPERS
# ============================================================

def mean_of(values):
    """Mean over non-None values. Returns None if there are none."""

    values = [value for value in values if value is not None]

    if not values:
        return None

    return float(np.mean(values))


def relative_improvement(base_value, lora_value):
    """Relative reduction of an error metric. None if not applicable."""

    if base_value is None or lora_value is None:
        return None

    if base_value <= 0:
        return 0.0

    return (base_value - lora_value) / base_value


def absolute_change(base_value, lora_value):
    """base - lora, or None if either side is missing."""

    if base_value is None or lora_value is None:
        return None

    return base_value - lora_value


# ============================================================
# PER
# ============================================================

def phoneme_error_rate(reference_phones, hypothesis_phones):
    """PER = error rate computed over the phoneme sequence."""

    if not reference_phones:
        return None

    if not hypothesis_phones:
        return 1.0

    return float(calc_wer(reference_phones, hypothesis_phones))


# ============================================================
# PER-SAMPLE SCORING
# ============================================================

def score_reference(reference, phonemize):
    """Normalised form and phonemes of a reference transcript."""

    normalized = normalize_text(reference)

    phonemes = phonemize(normalized) if phonemize else None

    return normalized, phonemes


def score_hypothesis(
    reference,
    reference_normalized,
    reference_phonemes,
    hypothesis,
    phonemize,
    acoustic,
    elapsed,
    normalize=False,
):
    """
    Scores one hypothesis against the reference.

    WER and CER are computed on the raw texts by default, or on the
    normalised texts (lowercase, no punctuation) when `normalize` is
    True. PER and acoustic similarity always use the normalised forms,
    since punctuation would otherwise leak into the phoneme sequence.
    """

    normalized = normalize_text(hypothesis)

    phonemes = phonemize(normalized) if phonemize else None

    return {
        "hypothesis": hypothesis,
        "hypothesis_normalized": normalized,
        "phonemes": phonemes,
        "wer": (
            calc_wer(reference_normalized, normalized)
            if normalize
            else calc_wer(reference, hypothesis)
        ),
        "cer": (
            calc_cer(reference_normalized, normalized)
            if normalize
            else calc_cer(reference, hypothesis)
        ),
        "per": (
            phoneme_error_rate(reference_phonemes, phonemes)
            if phonemize
            else None
        ),
        "acoustic_similarity": (
            acoustic.similarity(reference_normalized, normalized)
            if acoustic is not None
            else None
        ),
        "time_seconds": elapsed,
    }


def compare(base, lora):
    """BASE vs LoRA comparison block for a single sample."""

    return {
        "wer_improvement": relative_improvement(base["wer"], lora["wer"]),
        "cer_improvement": relative_improvement(base["cer"], lora["cer"]),
        "per_improvement": relative_improvement(base["per"], lora["per"]),
        "wer_absolute_change": absolute_change(base["wer"], lora["wer"]),
        "cer_absolute_change": absolute_change(base["cer"], lora["cer"]),
        "per_absolute_change": absolute_change(base["per"], lora["per"]),
        "acoustic_similarity_change": absolute_change(
            lora["acoustic_similarity"],
            base["acoustic_similarity"],
        ),
        "faster_or_slower": (
            "faster"
            if lora["time_seconds"] < base["time_seconds"]
            else "slower"
        ),
    }


# ============================================================
# AGGREGATION
# ============================================================

def _corpus_scores(results, model, normalize=False):
    """Corpus-level WER, CER and PER for one model over a result list."""

    if not results:
        return 0.0, 0.0, None

    if normalize:
        references = [item["reference_normalized"] for item in results]
        hypotheses = [item[model]["hypothesis_normalized"] for item in results]
    else:
        references = [item["reference"] for item in results]
        hypotheses = [item[model]["hypothesis"] for item in results]

    corpus_wer = float(calc_wer(references, hypotheses))

    corpus_cer = float(calc_cer(references, hypotheses))

    # Corpus PER runs over the phoneme sequences, skipping any sample
    # whose reference could not be phonemised.
    phoneme_pairs = [
        (item["reference_phonemes"], item[model]["phonemes"] or "")
        for item in results
        if item["reference_phonemes"]
    ]

    if phoneme_pairs:
        corpus_per = float(
            calc_wer(
                [reference for reference, _ in phoneme_pairs],
                [hypothesis for _, hypothesis in phoneme_pairs],
            )
        )
    else:
        corpus_per = None

    return corpus_wer, corpus_cer, corpus_per


def _model_summary(results, model, normalize=False):
    """Mean and corpus figures for one model over a result list."""

    corpus_wer, corpus_cer, corpus_per = _corpus_scores(results, model, normalize)

    return {
        "mean_phrase_wer": mean_of([r[model]["wer"] for r in results]) or 0.0,
        "mean_phrase_cer": mean_of([r[model]["cer"] for r in results]) or 0.0,
        "mean_phrase_per": mean_of([r[model]["per"] for r in results]),
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "corpus_per": corpus_per,
        "mean_acoustic_similarity": mean_of(
            [r[model]["acoustic_similarity"] for r in results]
        ),
        "mean_time_seconds": mean_of(
            [r[model]["time_seconds"] for r in results]
        ) or 0.0,
    }


def summarise(results, description=None, normalize=False):
    """
    Full BASE vs LoRA summary for a list of samples.

    Used both per speaker and for the whole corpus.
    """

    base = _model_summary(results, "base", normalize)

    lora = _model_summary(results, "lora", normalize)

    summary = {
        "samples": len(results),
        "base": base,
        "lora": lora,
        "comparison": {
            "mean_phrase_wer_improvement": relative_improvement(
                base["mean_phrase_wer"], lora["mean_phrase_wer"]
            ),
            "mean_phrase_cer_improvement": relative_improvement(
                base["mean_phrase_cer"], lora["mean_phrase_cer"]
            ),
            "mean_phrase_per_improvement": relative_improvement(
                base["mean_phrase_per"], lora["mean_phrase_per"]
            ),
            "corpus_wer_improvement": relative_improvement(
                base["corpus_wer"], lora["corpus_wer"]
            ),
            "corpus_cer_improvement": relative_improvement(
                base["corpus_cer"], lora["corpus_cer"]
            ),
            "corpus_per_improvement": relative_improvement(
                base["corpus_per"], lora["corpus_per"]
            ),
            "mean_acoustic_similarity_change": absolute_change(
                lora["mean_acoustic_similarity"],
                base["mean_acoustic_similarity"],
            ),
            "mean_time_change": (
                lora["mean_time_seconds"] - base["mean_time_seconds"]
            ),
            "lora_faster": (
                lora["mean_time_seconds"] < base["mean_time_seconds"]
            ),
        },
    }

    if description is not None:
        summary["description"] = description

    return summary
