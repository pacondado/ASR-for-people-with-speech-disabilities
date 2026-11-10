#!/usr/bin/env python3
# Copyright (c) 2026 Paulo Condado
# License: MIT
"""
Compare two evaluate.py result files on the same test set.

Answers the question "is the WER difference between model A and model B
real, or could it be chance?" with a paired bootstrap over utterances:
the test set is resampled thousands of times and the WER difference is
recomputed each time. If the 95 % interval contains 0, the two models
cannot be told apart on this test set.

Usage:

    python compare_results.py output_whisper/run_A.json output_whisper/run_B.json

Options:
    --model lora|base   which hypothesis to compare (default: lora)
    --top N             list only the N utterances with the largest difference
                        (default: every utterance where A and B differ)
    --samples N         bootstrap resamples (default: 10000)
    --normalize         lowercase and strip punctuation before scoring, so
                        "Irão"/"irão" or an extra comma do not count as errors
    --speakers NAME...  only these speakers (e.g. --speakers Condado)
    --description TEXT  only speakers with this description, as configured
                        in whisper_eval/config.py (e.g. "with speech disorders")
"""

import argparse
import json
import sys

import os

import numpy as np
from jiwer import cer as calc_cer
from jiwer import process_words
from jiwer import wer as calc_wer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_eval.text import normalize_text


def load_samples(path, model, speakers=None, description=None):
    """
    {(speaker, audio): sample} from a results JSON, where sample holds the
    reference, the hypothesis and the phoneme sequences used for PER.

    `speakers` keeps only the named speakers; `description` keeps only the
    speakers whose description matches exactly (case-insensitive).
    """

    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)

    samples = {}

    wanted_description = description.strip().lower() if description else None

    for speaker, data in document["results_by_person"].items():
        if speakers and speaker not in speakers:
            continue

        speaker_description = (data.get("description") or "").strip().lower()

        if wanted_description is not None and speaker_description != wanted_description:
            continue

        for item in data.get("results", []):
            key = (speaker, item["audio"])
            samples[key] = {
                "reference": item["reference"],
                "hypothesis": item[model]["hypothesis"],
                "reference_phonemes": item.get("reference_phonemes"),
                "phonemes": item[model].get("phonemes"),
            }

    return samples


def word_errors(reference, hypothesis):
    """(errors, reference words) for one utterance."""

    if not reference.strip():
        return 0, 0

    out = process_words(reference, hypothesis)

    errors = out.substitutions + out.deletions + out.insertions

    reference_words = out.hits + out.substitutions + out.deletions

    return errors, reference_words


def utterance_rate(metric, reference, hypothesis):
    """WER or CER of one utterance, None for an empty reference."""

    if not reference.strip():
        return None

    return float(metric(reference, hypothesis))


def phoneme_rate(sample):
    """PER of one utterance from the phonemes stored by evaluate.py."""

    reference = sample.get("reference_phonemes")

    if not reference:
        return None

    return float(calc_wer(reference, sample.get("phonemes") or ""))


def corpus_per(samples, keys):
    """Corpus PER over the utterances that have phonemes, or None."""

    pairs = [
        (samples[k]["reference_phonemes"], samples[k].get("phonemes") or "")
        for k in keys
        if samples[k].get("reference_phonemes")
    ]

    if not pairs:
        return None

    return float(calc_wer([r for r, _ in pairs], [h for _, h in pairs]))


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("result_a", help="Results JSON of model A")
    parser.add_argument("result_b", help="Results JSON of model B")
    parser.add_argument("--model", choices=["lora", "base"], default="lora")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="List only the N largest differences (default: all of them)",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Lowercase and strip punctuation before computing WER",
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        metavar="NAME",
        help="Compare only these speakers",
    )
    parser.add_argument(
        "--description",
        metavar="TEXT",
        help='Compare only speakers with this description, e.g. "with speech disorders"',
    )
    args = parser.parse_args(argv)

    a = load_samples(args.result_a, args.model, args.speakers, args.description)
    b = load_samples(args.result_b, args.model, args.speakers, args.description)

    keys = sorted(set(a) & set(b))

    if not keys:
        print(
            "error: no utterance in common for the selected speakers. "
            "Check --speakers / --description against the JSON files.",
            file=sys.stderr,
        )
        return 2

    if len(keys) < max(len(a), len(b)):
        print(
            f"WARNING: comparing only the {len(keys)} utterances present in both "
            f"files (A has {len(a)}, B has {len(b)}).",
            file=sys.stderr,
        )

    rows = []

    per_utterance = {}

    prepare = normalize_text if args.normalize else (lambda text: text)

    for key in keys:
        reference = a[key]["reference"]
        hyp_a, hyp_b = a[key]["hypothesis"], b[key]["hypothesis"]
        errors_a, words = word_errors(prepare(reference), prepare(hyp_a))
        errors_b, _ = word_errors(prepare(reference), prepare(hyp_b))
        rows.append((key, reference, hyp_a, hyp_b, errors_a, errors_b, words))

        per_utterance[key] = {
            "wer": (utterance_rate(calc_wer, prepare(reference), prepare(hyp_a)),
                    utterance_rate(calc_wer, prepare(reference), prepare(hyp_b))),
            "cer": (utterance_rate(calc_cer, prepare(reference), prepare(hyp_a)),
                    utterance_rate(calc_cer, prepare(reference), prepare(hyp_b))),
            "per": (phoneme_rate(a[key]), phoneme_rate(b[key])),
        }

    errors_a = np.array([r[4] for r in rows], dtype=float)
    errors_b = np.array([r[5] for r in rows], dtype=float)
    words = np.array([r[6] for r in rows], dtype=float)

    wer_a = errors_a.sum() / words.sum()
    wer_b = errors_b.sum() / words.sum()
    difference = wer_b - wer_a

    # ---- paired bootstrap over utterances ----
    rng = np.random.default_rng(args.seed)
    n = len(rows)
    index = rng.integers(0, n, size=(args.samples, n))
    total_words = words[index].sum(axis=1)
    boot = (errors_b[index].sum(axis=1) - errors_a[index].sum(axis=1)) / total_words
    low, high = np.percentile(boot, [2.5, 97.5])
    share_b_worse = float((boot > 0).mean())

    rule = "=" * 70

    print(rule)
    print(
        f"PAIRED COMPARISON ({args.model.upper()} hypotheses, "
        f"{'normalised' if args.normalize else 'raw'} text)"
    )
    print(rule)
    print(f"A: {args.result_a}")
    print(f"B: {args.result_b}")
    selected = sorted({speaker for (speaker, _audio), *_ in rows})
    print(f"Speakers: {', '.join(selected)}"
          + (f'   (description: "{args.description}")' if args.description else ""))
    print(f"Utterances: {n}   Reference words: {int(words.sum())}")
    print()
    print(f"Corpus WER  A: {wer_a:.4f}   B: {wer_b:.4f}")
    refs = [prepare(a[k]["reference"]) for k in keys]
    print(f"Corpus CER  A: {calc_cer(refs, [prepare(a[k]['hypothesis']) for k in keys]):.4f}   "
          f"B: {calc_cer(refs, [prepare(b[k]['hypothesis']) for k in keys]):.4f}")
    per_a, per_b = corpus_per(a, keys), corpus_per(b, keys)
    print(f"Corpus PER  A: {fmt(per_a)}   B: {fmt(per_b)}"
          + ("   (no phonemes in the JSON: run evaluate.py with PER enabled)"
             if per_a is None and per_b is None else ""))
    print(f"Difference (B - A): {difference:+.4f}   "
          f"= {int(errors_b.sum() - errors_a.sum()):+d} word errors")
    print(f"95 % bootstrap interval: [{low:+.4f}, {high:+.4f}]")
    print(f"B worse than A in {share_b_worse:.1%} of resamples")
    print()

    if low <= 0 <= high:
        print("=> The interval contains 0: the difference is NOT significant on")
        print("   this test set. It may be chance (training randomness, a few")
        print("   utterances). More test data or several seeds are needed.")
    else:
        better = "A" if difference > 0 else "B"
        print(f"=> The interval excludes 0: model {better} is better on this test set.")

    # ---- where the difference comes from ----
    diffs = errors_b - errors_a
    order = np.argsort(-np.abs(diffs))

    concentrated = np.abs(diffs)[order[: min(5, n)]].sum()
    total_abs = np.abs(diffs).sum()

    print()
    print(rule)
    n_different = int((diffs != 0).sum())
    limit = n_different if args.top is None else min(args.top, n_different)
    print(f"UTTERANCES WHERE A AND B DIFFER ({limit} of {n_different}, largest first)")
    print(rule)

    if total_abs:
        print(f"The top 5 utterances account for {concentrated / total_abs:.0%} "
              "of all per-utterance differences.")

    print(f"Utterances where B is worse: {int((diffs > 0).sum())}, "
          f"better: {int((diffs < 0).sum())}, equal: {int((diffs == 0).sum())}")

    for i in order[:limit]:
        if diffs[i] == 0:
            break
        (speaker, audio), reference, hyp_a, hyp_b, ea, eb, w = rows[i]
        print()
        m = per_utterance[(speaker, audio)]
        print(f"[{speaker}] {audio}   errors A={int(ea)} B={int(eb)} (of {int(w)} words)")
        print(f"  WER A={fmt(m['wer'][0])} B={fmt(m['wer'][1])}   "
              f"CER A={fmt(m['cer'][0])} B={fmt(m['cer'][1])}   "
              f"PER A={fmt(m['per'][0])} B={fmt(m['per'][1])}")
        print(f"  REF: {reference}")
        print(f"  A  : {hyp_a}")
        print(f"  B  : {hyp_b}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
