# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Compare training runs to understand why validation metrics differ.

Usage:
    python diagnose_runs.py OUTPUT_DIR_1 [OUTPUT_DIR_2 ...] [--top 10]

Run it from the same working directory used for training, so the audio
paths stored in the split CSVs can be resolved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import pandas as pd

LONG_AUDIO_SECONDS = 30.0  # Whisper silently truncates anything longer.


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _duration(path: str) -> float | None:
    try:
        import soundfile as sf

        return sf.info(path).duration
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def _epoch_table(run: Path) -> pd.DataFrame:
    """Train loss / eval metrics per epoch from trainer_state.json."""
    state_file = run / "trainer_state.json"
    if not state_file.exists():
        return pd.DataFrame()
    history = json.loads(state_file.read_text())["log_history"]
    rows, last_train_loss = [], None
    for entry in history:
        if "loss" in entry:
            last_train_loss = entry["loss"]  # most recent training loss before this eval
        if "eval_loss" in entry:
            rows.append(
                {
                    "epoch": round(entry.get("epoch", 0), 2),
                    "train_loss": last_train_loss,
                    "eval_loss": entry["eval_loss"],
                    "eval_wer": entry.get("eval_wer"),
                    "eval_cer": entry.get("eval_cer"),
                }
            )
    best = json.loads(state_file.read_text()).get("best_model_checkpoint")
    if best:
        print(f"  best checkpoint: {best}")
    return pd.DataFrame(rows)


def diagnose(run: Path, top: int, text_col: str, audio_col: str) -> pd.DataFrame:
    print("=" * 80)
    print(f"RUN: {run}")
    print("=" * 80)

    train = pd.read_csv(run / "train_split.csv")
    val = pd.read_csv(run / "validation_split.csv")
    results = pd.DataFrame(json.loads((run / "validation_results.json").read_text()))
    val = val.reset_index(drop=True).join(results[["hypothesis", "wer", "cer"]])

    # 1. Learning curve -------------------------------------------------------
    epochs = _epoch_table(run)
    print("\n[1] Learning curve (did eval WER/CER improve? did early stopping kick in?)")
    print(epochs.to_string(index=False) if not epochs.empty else "  trainer_state.json not found")

    # 2. Overlap between train and validation ---------------------------------
    same_text = _norm(val[text_col]).isin(set(_norm(train[text_col])))
    same_audio = val[audio_col].isin(set(train[audio_col]))
    print("\n[2] Train/validation overlap")
    print(f"  validation rows whose TEXT also appears in train : {same_text.sum()} / {len(val)}")
    print(f"  validation rows whose AUDIO also appears in train: {same_audio.sum()} / {len(val)}")
    if same_text.any() and (~same_text).any():
        print(f"  mean CER  text seen in train : {val.loc[same_text, 'cer'].mean():.4f}")
        print(f"  mean CER  text NOT in train  : {val.loc[~same_text, 'cer'].mean():.4f}")

    # 3. Audio duration --------------------------------------------------------
    val["duration"] = [_duration(p) for p in val[audio_col]]
    if val["duration"].notna().any():
        long_mask = val["duration"] > LONG_AUDIO_SECONDS
        print("\n[3] Audio duration (validation)")
        print(f"  mean / max duration: {val['duration'].mean():.1f}s / {val['duration'].max():.1f}s")
        print(f"  files > {LONG_AUDIO_SECONDS:.0f}s (truncated by Whisper): {long_mask.sum()}")
        if long_mask.any():
            print(f"  mean CER on files > {LONG_AUDIO_SECONDS:.0f}s: {val.loc[long_mask, 'cer'].mean():.4f}")
    else:
        print("\n[3] Audio duration: files not found from this directory (skipped)")

    # 4. Worst phrases ---------------------------------------------------------
    worst = val.sort_values("cer", ascending=False).head(top)
    print(f"\n[4] {top} worst phrases (repetition loops, truncation or wrong labels show up here)")
    for _, row in worst.iterrows():
        dur = f"{row['duration']:.1f}s" if pd.notna(row.get("duration")) else "?"
        print(f"  CER={row['cer']:.3f}  dur={dur}  {row[audio_col]}")
        print(f"    REF: {row[text_col]}")
        print(f"    HYP: {row['hypothesis']}")

    # 5. How much do the worst phrases weigh? ----------------------------------
    rest = val.drop(worst.index)
    refs, hyps = val[text_col].astype(str).str.strip().tolist(), val["hypothesis"].fillna("").tolist()
    print("\n[5] Corpus metrics")
    print(f"  corpus CER (all)              : {jiwer.cer(refs, hyps):.4f}")
    if len(rest):
        print(
            f"  corpus CER without worst {top:<3}: "
            f"{jiwer.cer(rest[text_col].astype(str).str.strip().tolist(), rest['hypothesis'].fillna('').tolist()):.4f}"
        )
    print()
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Output directories created by train.py")
    parser.add_argument("--top", type=int, default=10, help="Number of worst phrases to list.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--audio-column", default="audio")
    args = parser.parse_args()

    vals = {run: diagnose(run, args.top, args.text_column, args.audio_column) for run in args.runs}

    # Cross-run comparison: same validation phrases, different results?
    if len(vals) > 1:
        print("=" * 80)
        print("CROSS-RUN COMPARISON")
        print("=" * 80)
        runs = list(vals)
        for i, a in enumerate(runs):
            for b in runs[i + 1 :]:
                va, vb = vals[a], vals[b]
                tb = pd.read_csv(b / "train_split.csv")
                shared = va.merge(vb, on=args.audio_column, suffixes=("_a", "_b"))
                print(f"\n{a.name}  vs  {b.name}")
                print(f"  validation rows in common: {len(shared)}")
                if len(shared):
                    print(f"  mean CER on common rows -> A: {shared['cer_a'].mean():.4f} | B: {shared['cer_b'].mean():.4f}")
                in_b_train = va[args.audio_column].isin(set(tb[args.audio_column])).sum()
                print(f"  validation rows of A used for TRAINING in B: {in_b_train} / {len(va)}")


if __name__ == "__main__":
    main()
