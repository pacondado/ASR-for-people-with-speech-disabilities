# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Grapheme-to-phoneme conversion for the PER metric.

Two paths, in order of preference. Neither requires external
programs — both are pure Python:

  1. epitran — G2P maps curated per language. Only English needs
     Flite; for Portuguese (por-Latn) pip is enough.

  2. Embedded rule-based G2P (fallback) — an approximation of
     European Portuguese with no dependencies at all. Less accurate,
     but since reference and hypothesis go through the SAME rules,
     PER remains valid as a BASE vs LoRA comparative measure.
"""

import re
import sys


# ============================================================
# G2P VIA EPITRAN — pip only, no external programs
# ============================================================

def build_epitran_phonemizer(code):
    """G2P via epitran. Returns a function, or None if unavailable."""

    try:
        import epitran

    except ImportError:
        print(
            "WARNING: epitran is not installed.\n"
            "         pip install epitran",
            file=sys.stderr,
        )
        return None

    try:
        transliterator = epitran.Epitran(code)

    except Exception as error:
        print(
            f"WARNING: epitran unavailable for '{code}' ({error}).",
            file=sys.stderr,
        )
        return None

    def phonemize(text):
        phones = []

        for word in text.split():

            # trans_list returns the IPA segments already split
            try:
                phones.extend(transliterator.trans_list(word))

            except Exception:
                phones.extend(list(transliterator.transliterate(word)))

        return " ".join(phone for phone in phones if phone.strip())

    # Sanity check
    try:
        if not phonemize("teste"):
            raise RuntimeError("returned an empty string")

    except Exception as error:
        print(
            f"WARNING: epitran produced no phonemes ({error}).",
            file=sys.stderr,
        )
        return None

    return phonemize


# ============================================================
# RULE-BASED G2P — European Portuguese (fallback)
# ============================================================

VOWELS = set("aeiouáàâãéêíóôõúü")

ACCENT_MAP = {
    "á": "a", "à": "a", "â": "ɐ",
    "é": "ɛ", "ê": "e",
    "í": "i",
    "ó": "ɔ", "ô": "o",
    "ú": "u", "ü": "u",
}


def rule_based_phonemize(text):
    """
    Approximate phonemic transcription for European Portuguese.

    Collapses the spelling ambiguities that matter for PER: s/z,
    c/ç/k, g/j, final m/n, nasals, and the reduction of unstressed
    final vowels.
    """

    if not text:
        return ""

    output = []

    for word in text.split():

        phones = []

        i = 0

        n = len(word)

        while i < n:

            ch = word[i]

            nxt = word[i + 1] if i + 1 < n else ""

            nxt2 = word[i + 2] if i + 2 < n else ""

            last = i == n - 1

            # ---- digraphs ----

            if ch == "n" and nxt == "h":
                phones.append("ɲ"); i += 2; continue

            if ch == "l" and nxt == "h":
                phones.append("ʎ"); i += 2; continue

            if ch == "c" and nxt == "h":
                phones.append("ʃ"); i += 2; continue

            if ch == "r" and nxt == "r":
                phones.append("ʁ"); i += 2; continue

            if ch == "s" and nxt == "s":
                phones.append("s"); i += 2; continue

            if ch == "q" and nxt == "u" and nxt2 in ("e", "i"):
                phones.append("k"); i += 2; continue

            if ch == "g" and nxt == "u" and nxt2 in ("e", "i"):
                phones.append("g"); i += 2; continue

            # ---- orthographic nasal vowels ----

            if ch in "ãõ":
                phones.append("ɐ̃" if ch == "ã" else "õ")
                i += 1
                continue

            # vowel + m/n in coda -> nasal vowel
            if (
                ch in VOWELS
                and nxt in ("m", "n")
                and (i + 2 >= n or word[i + 2] not in VOWELS)
            ):
                base = ACCENT_MAP.get(ch, ch)

                phones.append(base + "̃")

                i += 2

                continue

            # ---- context-dependent consonants ----

            if ch == "c":
                phones.append("s" if nxt in ("e", "i") else "k")
                i += 1
                continue

            if ch == "ç":
                phones.append("s"); i += 1; continue

            if ch == "g":
                phones.append("ʒ" if nxt in ("e", "i") else "g")
                i += 1
                continue

            if ch == "j":
                phones.append("ʒ"); i += 1; continue

            if ch == "s":
                previous = word[i - 1] if i > 0 else ""

                if last:
                    phones.append("ʃ")
                elif previous in VOWELS and nxt in VOWELS:
                    phones.append("z")
                elif nxt and nxt not in VOWELS:
                    phones.append("ʃ")
                else:
                    phones.append("s")

                i += 1
                continue

            if ch == "z":
                phones.append("ʃ" if last else "z")
                i += 1
                continue

            if ch == "x":
                phones.append("ʃ"); i += 1; continue

            if ch == "r":
                phones.append("ʁ" if i == 0 else "ɾ")
                i += 1
                continue

            if ch == "l":
                phones.append("ɫ" if (last or nxt not in VOWELS) else "l")
                i += 1
                continue

            if ch == "h":
                i += 1
                continue

            if ch in ("m", "n"):
                phones.append(ch); i += 1; continue

            # ---- vowels ----

            if ch in ACCENT_MAP:
                phones.append(ACCENT_MAP[ch]); i += 1; continue

            if ch in "aeiou":
                # unstressed final vowel reduction (European Portuguese)
                if last and ch == "o":
                    phones.append("u")
                elif last and ch == "e":
                    phones.append("ɨ")
                else:
                    phones.append(ch)

                i += 1
                continue

            # ---- everything else ----

            phones.append(ch)

            i += 1

        output.extend(phones)

    return " ".join(output)


# ============================================================
# BACKEND SELECTION
# ============================================================

def build_phonemizer(preference, epitran_code):
    """
    Returns (phonemiser function, backend name), trying the backends
    in the order given by `preference`.

    No backend needs external programs. With "rules" in the list it
    never fails: the rules need nothing beyond Python itself.
    """

    builders = {
        "epitran": (
            lambda: build_epitran_phonemizer(epitran_code),
            "epitran",
        ),
        "rules": (
            lambda: rule_based_phonemize,
            "rule-based-pt",
        ),
    }

    for choice in preference:

        if choice not in builders:
            print(
                f"WARNING: unknown G2P backend: '{choice}'",
                file=sys.stderr,
            )
            continue

        builder, backend_name = builders[choice]

        phonemize = builder()

        if phonemize is None:
            continue

        if backend_name == "rule-based-pt":
            print(
                "WARNING: falling back to the embedded rule-based G2P "
                "(approximate). PER stays comparable between BASE and LoRA.",
                file=sys.stderr,
            )

        # Shared cache — references repeat across speakers
        cache = {}

        def cached(text, _phonemize=phonemize, _cache=cache):
            if not text:
                return ""

            if text not in _cache:
                _cache[text] = _phonemize(text)

            return _cache[text]

        return cached, backend_name

    print(
        "WARNING: no G2P backend available — PER disabled.",
        file=sys.stderr,
    )

    return None, None
