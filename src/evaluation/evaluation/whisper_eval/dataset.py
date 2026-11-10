# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Loading and validation of the test manifest."""

import csv
import os

REQUIRED_COLUMNS = {
    "audio",
    "text",
}


def load_test_rows(path):
    """Reads the test CSV and checks it has the columns we rely on."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Test CSV not found:\n{path}")

    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Test CSV is empty.")

    missing = REQUIRED_COLUMNS - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Missing columns in {path}: " + ", ".join(sorted(missing))
        )

    return rows
