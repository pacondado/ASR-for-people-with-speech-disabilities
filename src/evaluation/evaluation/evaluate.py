#!/usr/bin/env python3
# Copyright (c) 2026 Paulo Condado
# License: MIT
"""
Whisper — BASE vs LoRA evaluation, per speaker.

Reports WER, CER, PER (phoneme error rate) and an optional acoustic
similarity score, per utterance, per speaker and over the whole
corpus, then writes everything to JSON.

Usage:

    python src/evaluation/evaluate.py \\
        --model output_whisper/lora_weights \\
        --data data/test

where --model is the directory of the trained LoRA adapter and --data
holds test.csv plus an audio/ subdirectory with one folder per
speaker. Run with --help for the full list of options; defaults live
in whisper_eval/config.py.
"""

import os
import sys
import time

# Running this file directly puts its own directory on sys.path, so the
# package next to it is importable. The insert also covers `python -m`
# and imports from other working directories.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_eval import asr, cli, dataset, display, g2p, metrics, results
from whisper_eval.acoustic import AcousticSimilarity


# ============================================================
# SETUP
# ============================================================

def build_scorers(settings):
    """Sets up the optional G2P and acoustic backends."""

    if settings.ENABLE_PHONETIC:
        phonemize, g2p_backend = g2p.build_phonemizer(
            settings.G2P_PREFERENCE,
            settings.EPITRAN_CODE,
        )
    else:
        phonemize, g2p_backend = None, None

    acoustic = AcousticSimilarity.from_settings(settings)

    return phonemize, g2p_backend, acoustic


def load_models(settings, device):
    """Processor plus both models, with progress on stdout."""

    print()
    print("Loading processor...")

    processor = asr.load_processor(settings)

    print("Processor loaded.")

    print()
    print(display.RULE)
    print("LOADING BASE MODEL")
    print(display.RULE)

    base_model = asr.load_base_model(settings, device)

    print("BASE model loaded.")

    print()
    print(display.RULE)
    print("LOADING LoRA MODEL")
    print(display.RULE)

    lora_model = asr.load_lora_model(settings, device)

    print("LoRA model loaded.")

    display.print_model_info(base_model, lora_model)

    return base_model, lora_model, processor


# ============================================================
# ONE SAMPLE
# ============================================================

def evaluate_sample(
    row,
    index,
    total,
    person,
    person_info,
    audio_path,
    models,
    phonemize,
    acoustic,
    settings,
    device,
):
    """Transcribes one utterance with both models and scores it."""

    base_model, lora_model, processor = models

    reference = row["text"].strip()

    audio = asr.load_wav(audio_path, settings.SAMPLE_RATE)

    duration = len(audio) / settings.SAMPLE_RATE

    # ----- BASE -----

    print()
    print(f"[{person}] Processing BASE [{index}/{total}]...")

    start = time.perf_counter()

    base_hypothesis = asr.transcribe(
        base_model,
        processor,
        audio,
        settings,
        device,
    )

    base_time = time.perf_counter() - start

    # ----- LoRA -----

    print()
    print(f"[{person}] Processing LoRA [{index}/{total}]...")

    start = time.perf_counter()

    lora_hypothesis = asr.transcribe(
        lora_model,
        processor,
        audio,
        settings,
        device,
    )

    lora_time = time.perf_counter() - start

    del audio

    asr.cleanup_memory()

    # ----- scoring -----

    if acoustic.available:
        print()
        print(f"[{person}] TTS + DTW [{index}/{total}]...")

    reference_normalized, reference_phonemes = metrics.score_reference(
        reference,
        phonemize,
    )

    scorer = acoustic if acoustic.available else None

    base = metrics.score_hypothesis(
        reference,
        reference_normalized,
        reference_phonemes,
        base_hypothesis,
        phonemize,
        scorer,
        base_time,
        normalize=settings.NORMALIZE_TEXT,
    )

    lora = metrics.score_hypothesis(
        reference,
        reference_normalized,
        reference_phonemes,
        lora_hypothesis,
        phonemize,
        scorer,
        lora_time,
        normalize=settings.NORMALIZE_TEXT,
    )

    return {
        "index": index,
        "speaker": person,
        "speaker_description": person_info.get("description", ""),
        "audio": row["audio"].strip(),
        "audio_path": audio_path,
        "duration_seconds": duration,
        "reference": reference,
        "reference_normalized": reference_normalized,
        "reference_phonemes": reference_phonemes,
        "base": base,
        "lora": lora,
        "comparison": metrics.compare(base, lora),
    }


# ============================================================
# ONE PARTICIPANT
# ============================================================

def evaluate_participant(
    person,
    person_info,
    rows,
    models,
    phonemize,
    acoustic,
    settings,
    device,
):
    """Runs every test utterance for one speaker. Returns their samples."""

    print()
    print()
    print(display.RULE)
    print(f"PARTICIPANT: {person}")
    print(f"Description: {person_info.get('description') or '(none)'}")
    print(display.RULE)

    person_audio_dir = os.path.join(settings.AUDIO_DIR, person)

    if not os.path.exists(person_audio_dir):
        print()
        print("⚠️ WARNING:")
        print(f"Audio directory not found: {person_audio_dir}")
        print("Skipping participant.")
        return []

    samples = []

    for i, row in enumerate(rows):

        audio_path = os.path.join(person_audio_dir, row["audio"].strip())

        if not os.path.exists(audio_path):
            print()
            print("⚠️ WARNING:")
            print(f"Audio not found: {audio_path}")
            continue

        sample = evaluate_sample(
            row=row,
            index=i + 1,
            total=len(rows),
            person=person,
            person_info=person_info,
            audio_path=audio_path,
            models=models,
            phonemize=phonemize,
            acoustic=acoustic,
            settings=settings,
            device=device,
        )

        display.print_sample(
            sample,
            len(rows),
            phonemize is not None,
            acoustic.available,
            normalize=settings.NORMALIZE_TEXT,
        )

        samples.append(sample)

    return samples


# ============================================================
# MAIN
# ============================================================

def run(settings):
    device = asr.pick_device(settings.DEVICE)

    phonemize, g2p_backend, acoustic = build_scorers(settings)

    display.print_header(settings, device, g2p_backend, acoustic.available)

    models = load_models(settings, device)

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    print()
    print(display.RULE)
    print("LOADING TEST DATA")
    print(display.RULE)

    rows = dataset.load_test_rows(settings.TEST_CSV)

    print()
    print(f"Test samples: {len(rows)}")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results_by_person = {}

    global_results = []

    for person, person_info in settings.PARTICIPANTS.items():

        results_by_person[person] = {
            "description": person_info.get("description", ""),
            "results": [],
        }

        samples = evaluate_participant(
            person=person,
            person_info=person_info,
            rows=rows,
            models=models,
            phonemize=phonemize,
            acoustic=acoustic,
            settings=settings,
            device=device,
        )

        if not samples:
            continue

        summary = metrics.summarise(
            samples,
            person_info.get("description", ""),
            normalize=settings.NORMALIZE_TEXT,
        )

        results_by_person[person]["results"] = samples

        results_by_person[person]["summary"] = summary

        global_results.extend(samples)

        display.print_person_summary(person, summary)

    if not global_results:
        raise RuntimeError(
            "No sample was evaluated. Check that the audio filenames in "
            f"{settings.TEST_CSV} exist under {settings.AUDIO_DIR}/<speaker>/."
        )

    # --------------------------------------------------------
    # Cross-speaker and overall
    # --------------------------------------------------------

    summaries = {
        person: data["summary"]
        for person, data in results_by_person.items()
        if "summary" in data
    }

    display.print_cross_speaker(summaries)

    overall = metrics.summarise(
        global_results,
        normalize=settings.NORMALIZE_TEXT,
    )

    display.print_overall(overall)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    document = results.build_output(
        settings=settings,
        device=device,
        g2p_backend=g2p_backend,
        acoustic_on=acoustic.available,
        results_by_person=results_by_person,
        overall=overall,
    )

    results.save(settings.RESULTS_FILE, document)

    display.print_saved(settings.RESULTS_FILE)


def main(argv=None):
    try:
        settings = cli.parse_settings(argv)

    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)

        return 2

    run(settings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
