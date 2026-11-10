# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Console formatting for the evaluation report.

Everything that writes to stdout lives here, so the metric code stays
free of presentation concerns.
"""

RULE = "=" * 100

THIN_RULE = "-" * 100


# ============================================================
# STATUS AND FORMATTING
# ============================================================

def wer_status(value):
    """Green/yellow/red for error metrics (lower is better)."""

    if value is None:
        return "⚪"

    if value < 0.20:
        return "🟢"
    elif value < 0.50:
        return "🟡"
    else:
        return "🔴"


def similarity_status(value):
    """Green/yellow/red for similarity scores (higher is better)."""

    if value is None:
        return "⚪"

    if value > 0.80:
        return "🟢"
    elif value > 0.50:
        return "🟡"
    else:
        return "🔴"


def fmt(value, decimals=6):
    """Formats a float or None, keeping the column width."""

    if value is None:
        return "n/a".rjust(decimals + 2)

    return f"{value:.{decimals}f}"


def _percent_change(label, value):
    """One line describing a relative change in an error metric."""

    if value is None:
        return

    if value > 0:
        print(f"LoRA {label} improvement: ⬇️ {value * 100:.2f}%")
    elif value < 0:
        print(f"LoRA {label} change: ⬆️ {-value * 100:.2f}% worse")
    else:
        print(f"LoRA {label} improvement: 0.00%")


# ============================================================
# HEADER
# ============================================================

def print_header(settings, device, g2p_backend, acoustic_on):
    print()
    print(RULE)
    print("WHISPER — BASE vs LoRA")
    print("EVALUATION BY SPEAKER")
    print(RULE)
    print()

    print(f"Base model         : {settings.MODEL_NAME}")
    print(f"LoRA (--model)     : {settings.LORA_PATH}")
    print(f"Data (--data)      : {settings.DATA_DIR}")
    print(f"Test CSV           : {settings.TEST_CSV}")
    print(f"Audio directory    : {settings.AUDIO_DIR}")
    print(f"Results (--output) : {settings.RESULTS_FILE}")
    print()

    print(f"Language           : {settings.LANGUAGE}")
    print(f"Task               : {settings.TASK}")
    print(f"Sample rate        : {settings.SAMPLE_RATE}")
    print(f"Max length         : {settings.MAX_LENGTH}")
    print(f"No-repeat ngram    : {settings.NO_REPEAT_NGRAM_SIZE}")
    print(
        "WER / CER text     : "
        + ("normalised (lowercase, no punctuation)" if settings.NORMALIZE_TEXT else "raw")
    )
    print(f"Device             : {device}")
    print()

    print(f"PER (phonemes)     : {'on' if g2p_backend else 'off'}")
    print(f"Acoustic (TTS+DTW) : {'on' if acoustic_on else 'off'}")

    if g2p_backend:
        print(f"G2P backend        : {g2p_backend}")

        if g2p_backend == "epitran":
            print(f"G2P language       : {settings.EPITRAN_CODE}")

    if acoustic_on:
        print(f"TTS voice          : {settings.TTS_VOICE}")
        print(f"TTS cache          : {settings.TTS_CACHE_DIR}")

    print()
    print("Participants:")

    for person, info in settings.PARTICIPANTS.items():
        description = info.get("description") or "(no description)"

        print(f"  - {person}: {description}")

    print()
    print(RULE)


def print_model_info(base_model, lora_model):
    print()
    print(RULE)
    print("MODEL CONFIGURATION")
    print(RULE)
    print()

    for label, model in (("BASE", base_model), ("LoRA", lora_model)):
        generation_config = model.generation_config

        print(f"{label} forced_decoder_ids : {generation_config.forced_decoder_ids}")
        print(f"{label} pad_token_id       : {generation_config.pad_token_id}")
        print(f"{label} eos_token_id       : {generation_config.eos_token_id}")

        # max_new_tokens must be None so that no warning can be triggered
        print(
            f"{label} max_length         : {generation_config.max_length}"
            f" (max_new_tokens={generation_config.max_new_tokens})"
        )
        print()

    print(RULE)


# ============================================================
# PER-SAMPLE
# ============================================================

def print_sample(sample, total, phonetic_on, acoustic_on, normalize=False):
    """
    One utterance: reference, both hypotheses and their scores.

    With `normalize`, the normalised texts (lowercase, no punctuation),
    which are what WER and CER were computed on, are shown below the
    original ones, prefixed with "norm:".
    """

    base = sample["base"]

    lora = sample["lora"]

    comparison = sample["comparison"]

    print()
    print(RULE)
    print(f"[{sample['speaker']}] [{sample['index']:02d}/{total}]")
    print(f"Audio    : {sample['audio']}")
    print(f"Duration : {sample['duration_seconds']:.2f}s")
    print()
    print("REF:")
    print(f"  {sample['reference']}")

    if normalize:
        print(f"  norm: {sample['reference_normalized']}")

    if phonetic_on:
        print(f"  /{sample['reference_phonemes']}/")

    for label, scores in (("BASE", base), ("LoRA", lora)):
        print()
        print(f"{wer_status(scores['wer'])} {label}:")
        print(f"  {scores['hypothesis']}")

        if normalize:
            print(f"  norm: {scores['hypothesis_normalized']}")

        if phonetic_on:
            print(f"  /{scores['phonemes']}/")

        print(
            f"  WER={scores['wer']:.3f} "
            f"CER={scores['cer']:.3f} "
            f"TIME={scores['time_seconds']:.2f}s"
        )

        if phonetic_on:
            print(f"  {wer_status(scores['per'])} PER={fmt(scores['per'], 3)}")

        if acoustic_on:
            similarity = scores["acoustic_similarity"]

            print(
                f"  {similarity_status(similarity)} "
                f"SIM={fmt(similarity, 3)}"
            )

    print()

    _percent_change("WER", comparison["wer_improvement"])
    _percent_change("CER", comparison["cer_improvement"])
    _percent_change("PER", comparison["per_improvement"])

    change = comparison["acoustic_similarity_change"]

    if change is not None:
        arrow = "⬆️" if change > 0 else ("⬇️" if change < 0 else "")
        print(f"LoRA sound similarity: {arrow} {change:+.4f}")

    print()
    print(THIN_RULE)


# ============================================================
# SUMMARIES
# ============================================================

def _summary_table(summary):
    base = summary["base"]

    lora = summary["lora"]

    print("                         BASE                  LoRA")
    print()

    rows = [
        ("Mean phrase WER", "mean_phrase_wer", wer_status),
        ("Mean phrase CER", "mean_phrase_cer", wer_status),
        ("Mean phrase PER", "mean_phrase_per", wer_status),
        (None, None, None),
        ("Corpus WER", "corpus_wer", wer_status),
        ("Corpus CER", "corpus_cer", wer_status),
        ("Corpus PER", "corpus_per", wer_status),
        (None, None, None),
        ("Sound similarity", "mean_acoustic_similarity", similarity_status),
    ]

    for label, key, status in rows:

        if label is None:
            print()
            continue

        print(
            f"{label:<20}"
            f"{status(base[key])} {fmt(base[key])}          "
            f"{status(lora[key])} {fmt(lora[key])}"
        )

    print()
    print(
        f"{'Mean inference time':<20}"
        f"{base['mean_time_seconds']:.3f}s"
        f"{'':<14}"
        f"{lora['mean_time_seconds']:.3f}s"
    )


def print_person_summary(person, summary):
    comparison = summary["comparison"]

    print()
    print()
    print(RULE)
    print(f"FINAL RESULTS — {person}")
    print(f"{summary.get('description') or '(no description)'}")
    print(RULE)
    print()

    _summary_table(summary)

    print()

    for label, key in (
        ("WER", "corpus_wer_improvement"),
        ("CER", "corpus_cer_improvement"),
        ("PER", "corpus_per_improvement"),
    ):
        if comparison[key] is not None:
            print(f"Corpus {label} improvement: {comparison[key] * 100:.2f}%")

    change = comparison["mean_acoustic_similarity_change"]

    if change is not None:
        print(f"Sound similarity change: {change:+.4f}")

    print()
    print(f"Samples: {summary['samples']}")
    print()
    print(RULE)


def print_cross_speaker(summaries):
    print()
    print()
    print(RULE)
    print("CROSS-SPEAKER COMPARISON")
    print(RULE)
    print()

    print(
        "The following comparison shows how the LoRA model behaves for "
        "the speakers with and without speech disorders."
    )

    print()

    for person, summary in summaries.items():

        base = summary["base"]

        lora = summary["lora"]

        comparison = summary["comparison"]

        print(f"{person} ({summary.get('description') or 'no description'})")

        blocks = [
            ("WER", "corpus_wer", "corpus_wer_improvement"),
            ("CER", "corpus_cer", "corpus_cer_improvement"),
            ("PER", "corpus_per", "corpus_per_improvement"),
        ]

        for label, key, improvement_key in blocks:

            if base[key] is None:
                continue

            print()
            print(f"  BASE corpus {label} : {fmt(base[key])}")
            print(f"  LoRA corpus {label} : {fmt(lora[key])}")

            if comparison[improvement_key] is not None:
                print(
                    f"  {label} improvement : "
                    f"{comparison[improvement_key] * 100:.2f}%"
                )

        if base["mean_acoustic_similarity"] is not None:
            print()
            print(f"  BASE sound sim. : {fmt(base['mean_acoustic_similarity'])}")
            print(f"  LoRA sound sim. : {fmt(lora['mean_acoustic_similarity'])}")

            change = comparison["mean_acoustic_similarity_change"]

            if change is not None:
                print(f"  Change          : {change:+.4f}")

        print()
        print(THIN_RULE)


def print_overall(summary):
    comparison = summary["comparison"]

    print()
    print()
    print(RULE)
    print("OVERALL RESULTS — ALL SPEAKERS")
    print(RULE)
    print()

    _summary_table(summary)

    print()

    for label, key in (
        ("WER", "corpus_wer_improvement"),
        ("CER", "corpus_cer_improvement"),
        ("PER", "corpus_per_improvement"),
    ):
        if comparison[key] is not None:
            print(f"Overall {label} improvement: {comparison[key] * 100:.2f}%")

    change = comparison["mean_acoustic_similarity_change"]

    if change is not None:
        print(f"Overall sound similarity change: {change:+.4f}")

    print()
    print(f"Total samples: {summary['samples']}")
    print()
    print(RULE)


def print_saved(path):
    print()
    print("Results saved to:")
    print(path)
    print()
    print(RULE)
    print("DONE")
    print(RULE)
