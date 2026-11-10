# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Split long recordings (> 30 s) into shorter segments with matching transcriptions.

Pipeline for one recording:
1. Whisper transcribes the audio with word-level timestamps (ASR words).
2. The *reference* transcription is aligned to the ASR words with a fuzzy,
   character-level similarity, so every reference word gets a start/end time
   (words without a counterpart are interpolated).
3. Cut points are chosen between reference words, preferring sentence
   punctuation and pauses, so that every segment is at most `max_seconds`.

The reference text is always kept as the label; Whisper's own output is only
used for timing. `match_ratio` (fraction of reference words paired with a
sufficiently similar ASR word) measures how reliable the timing of a segment is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Bonus for cutting after a word ending with these characters.
_PUNCT_SCORES = ((".!?…", 3.0), (";:", 2.0), (",—–-", 1.0))


@dataclass
class Word:
    """A word with (possibly unknown) start/end time in seconds."""

    text: str
    start: float | None = None
    end: float | None = None


@dataclass
class TimedWord:
    text: str
    start: float
    end: float
    matched: bool


@dataclass
class Segment:
    start: float
    end: float
    text: str
    match_ratio: float
    n_words: int

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def normalise_word(word: str) -> str:
    return _PUNCT_RE.sub("", word.lower())


def tokenize_reference(text: str) -> list[str]:
    """Split on whitespace, gluing punctuation-only tokens (e.g. '—') to the previous word."""
    tokens: list[str] = []
    for token in str(text).split():
        if not normalise_word(token) and tokens:
            tokens[-1] = f"{tokens[-1]} {token}"
        else:
            tokens.append(token)
    return tokens


def _clean_asr_words(words: Sequence[Word], total: float) -> list[Word]:
    """Drop empty words and fill missing timestamps with neighbouring values."""
    cleaned, last_end = [], 0.0
    for w in words:
        if not normalise_word(w.text):
            continue
        start = w.start if w.start is not None else last_end
        end = w.end if w.end is not None else start
        start, end = min(max(start, last_end), total), min(max(end, start), total)
        cleaned.append(Word(w.text.strip(), start, end))
        last_end = end
    return cleaned


def _spread(texts: list[str], start: float, end: float) -> list[tuple[float, float]]:
    """Distribute [start, end] over `texts` proportionally to their length."""
    lengths = [max(len(t), 1) for t in texts]
    total_len, spans, cursor = sum(lengths), [], start
    for length in lengths:
        step = (end - start) * length / total_len
        spans.append((cursor, cursor + step))
        cursor += step
    return spans


def word_similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1] between two normalised words."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _dp_align(ref: list[str], asr: list[str], gap_cost: float = 0.6) -> list[tuple[int, int, float]]:
    """Monotonic word alignment (Needleman-Wunsch) with fuzzy substitution cost.

    Returns (ref_index, asr_index, similarity) for every paired word. Words that
    are skipped (missing in the ASR output, or extra ASR words) are not returned.
    """
    n, m = len(ref), len(asr)
    sim = [[word_similarity(r, a) for a in asr] for r in ref]
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]  # 0 = pair, 1 = skip ref, 2 = skip asr
    for i in range(1, n + 1):
        cost[i][0], move[i][0] = i * gap_cost, 1
    for j in range(1, m + 1):
        cost[0][j], move[0][j] = j * gap_cost, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                cost[i - 1][j - 1] + 1.0 - sim[i - 1][j - 1],
                cost[i - 1][j] + gap_cost,
                cost[i][j - 1] + gap_cost,
            )
            best = min(range(3), key=options.__getitem__)
            cost[i][j], move[i][j] = options[best], best

    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        step = move[i][j]
        if step == 0:
            pairs.append((i - 1, j - 1, sim[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def align_reference(
    reference: str,
    asr_words: Sequence[Word],
    total_duration: float,
    min_similarity: float = 0.6,
) -> list[TimedWord]:
    """Give every reference word a time span using the ASR word timestamps.

    Words are aligned with a fuzzy (character-level) similarity, so a word that
    Whisper recognised imperfectly (e.g. atypical or dysarthric speech) still
    provides timing. A reference word counts as `matched` when its similarity
    with the paired ASR word is at least `min_similarity`.
    """
    ref_tokens = tokenize_reference(reference)
    asr = _clean_asr_words(asr_words, total_duration)
    n = len(ref_tokens)
    starts: list[float | None] = [None] * n
    ends: list[float | None] = [None] * n
    matched = [False] * n

    pairs = _dp_align([normalise_word(t) for t in ref_tokens], [normalise_word(w.text) for w in asr])
    for i, j, similarity in pairs:
        starts[i], ends[i] = asr[j].start, asr[j].end
        matched[i] = similarity >= min_similarity

    # Interpolate runs of words without timing between their known neighbours.
    i = 0
    while i < n:
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < n and starts[j] is None:
            j += 1
        left = ends[i - 1] if i > 0 else 0.0
        right = starts[j] if j < n else total_duration
        for k, (s, e) in enumerate(_spread(ref_tokens[i:j], left, max(left, right))):
            starts[i + k], ends[i + k] = s, e
        i = j

    # Enforce monotonic, in-range times.
    timed, cursor = [], 0.0
    for tok, s, e, m in zip(ref_tokens, starts, ends, matched):
        s = min(max(s, cursor), total_duration)
        e = min(max(e, s), total_duration)
        timed.append(TimedWord(tok, s, e, m))
        cursor = s
    return timed


# ---------------------------------------------------------------------------
# Choosing cut points
# ---------------------------------------------------------------------------

def _punct_score(token: str) -> float:
    last = token.rstrip()[-1:] if token.strip() else ""
    for chars, score in _PUNCT_SCORES:
        if last and last in chars:
            return score
    return 0.0


def _make_segment(words: list[TimedWord], start: float, end: float) -> Segment:
    return Segment(
        start=start,
        end=end,
        text=" ".join(w.text for w in words),
        match_ratio=sum(w.matched for w in words) / len(words),
        n_words=len(words),
    )


def choose_segments(
    words: list[TimedWord],
    total_duration: float,
    max_seconds: float = 28.0,
    min_seconds: float = 3.0,
) -> list[Segment]:
    """Greedily cut the word sequence into segments no longer than `max_seconds`.

    Among the admissible cut points (between two words, giving a segment of
    `min_seconds`..`max_seconds`), the one after strong punctuation and/or a
    long pause wins; ties prefer longer segments.
    """
    if not words:
        return []

    segments: list[Segment] = []
    first, seg_start, n = 0, 0.0, len(words)

    while first < n:
        if total_duration - seg_start <= max_seconds:
            segments.append(_make_segment(words[first:], seg_start, total_duration))
            break

        best: tuple[float, int, float] | None = None  # (score, index, cut_time)
        fallback: tuple[int, float] | None = None
        for i in range(first, n - 1):
            cut = (words[i].end + words[i + 1].start) / 2
            length = cut - seg_start
            if length > max_seconds:
                break
            fallback = (i, cut)
            if length < min_seconds:
                continue
            gap = max(0.0, words[i + 1].start - words[i].end)
            score = _punct_score(words[i].text) + 2.0 * min(gap, 1.0) + 0.02 * length
            if best is None or score >= best[0]:
                best = (score, i, cut)

        if best is not None:
            _, idx, cut = best
        elif fallback is not None:
            idx, cut = fallback
        else:
            # A single word straddles the limit: cut hard after it.
            idx, cut = first, min(words[first].end, seg_start + max_seconds)

        segments.append(_make_segment(words[first : idx + 1], seg_start, cut))
        first, seg_start = idx + 1, cut

    return segments
