# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Optional: run only the long-audio segmentation step, without training.

`train.py` already segments long recordings automatically (`--long-audio segment`,
the default) and caches the result. This script lets you run that step on its
own to inspect the segments (and listen to them) before training. The cache is
shared, so a later `train.py` run with the same settings reuses the segments.

Example:
    python segment_long_audio.py --dataset data/data.csv --audio-dir data/audio \
        --output-csv data/data_segmented.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from whisper_finetune.config import SUPPORTED_MODELS, DataConfig
from whisper_finetune.data import load_csv
from whisper_finetune.long_audio import LongAudioHandler
from whisper_finetune.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Input CSV.")
    parser.add_argument("--audio-dir", default="audio")
    parser.add_argument("--output-csv", required=True, help="Output CSV (short rows + segments).")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--language", default="pt")
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--segment-max-seconds", type=float, default=28.0)
    parser.add_argument("--segment-min-seconds", type=float, default=3.0)
    parser.add_argument("--segment-min-match", type=float, default=0.5)
    parser.add_argument("--segment-word-similarity", type=float, default=0.6)
    parser.add_argument("--segment-model", choices=sorted(SUPPORTED_MODELS), default="large-v3")
    parser.add_argument("--segment-device", default=None)
    parser.add_argument("--segment-batch-size", type=int, default=4)
    parser.add_argument("--segments-subdir", default="segments")
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    cfg = DataConfig(
        dataset_csv=args.dataset,
        audio_dir=args.audio_dir,
        audio_column=args.audio_column,
        text_column=args.text_column,
        long_audio="segment",
        max_audio_seconds=args.max_audio_seconds,
        segment_max_seconds=args.segment_max_seconds,
        segment_min_seconds=args.segment_min_seconds,
        segment_min_match=args.segment_min_match,
        segment_word_similarity=args.segment_word_similarity,
        segment_model=args.segment_model,
        segment_device=args.segment_device,
        segment_batch_size=args.segment_batch_size,
        segments_subdir=args.segments_subdir,
    )
    handler = LongAudioHandler(cfg, args.language)
    dataset = handler.process(load_csv(args.dataset, cfg), args.dataset)
    handler.close()

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(str(out_csv), index=False)
    report = handler.report()
    if len(report):
        report.to_csv(out_csv.with_name(f"{out_csv.stem}_segments_report.csv"), index=False)
    print(f"Saved {len(dataset)} rows to {out_csv}")


if __name__ == "__main__":
    main()
