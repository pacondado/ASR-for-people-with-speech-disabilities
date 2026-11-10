# Copyright (c) 2026 Paulo Condado
# License: MIT
"""Assembly and serialisation of the results JSON."""

import json
import os


def build_output(
    settings,
    device,
    g2p_backend,
    acoustic_on,
    results_by_person,
    overall,
):
    """The full result document, mirroring what is printed."""

    return {
        "configuration": {
            "base_model": settings.MODEL_NAME,
            "lora_path": settings.LORA_PATH,
            "data_dir": settings.DATA_DIR,
            "test_csv": settings.TEST_CSV,
            "audio_dir": settings.AUDIO_DIR,
            "language": settings.LANGUAGE,
            "task": settings.TASK,
            "sample_rate": settings.SAMPLE_RATE,
            "max_length": settings.MAX_LENGTH,
            "no_repeat_ngram_size": settings.NO_REPEAT_NGRAM_SIZE,
            "wer_cer_on_normalized_text": settings.NORMALIZE_TEXT,
            "device": device,
            "processor": settings.MODEL_NAME,
            "forced_decoder_ids": None,
            "attention_mask": True,

            "phonetic_metrics": {
                "per_enabled": g2p_backend is not None,
                "g2p_backend": g2p_backend,
                "g2p_preference": (
                    settings.G2P_PREFERENCE if settings.ENABLE_PHONETIC else None
                ),
                "g2p_language": (
                    settings.EPITRAN_CODE
                    if g2p_backend == "epitran"
                    else ("pt" if g2p_backend else None)
                ),
                "acoustic_enabled": acoustic_on,
                "tts_voice": settings.TTS_VOICE if acoustic_on else None,
                "n_mfcc": settings.N_MFCC if acoustic_on else None,
                "dtw_metric": "cosine" if acoustic_on else None,
            },
        },

        "participants": {
            person: {
                "description": data["description"],
                "samples": data.get("summary", {}).get("samples", 0),
            }
            for person, data in results_by_person.items()
        },

        "metrics_by_person": {
            person: data.get("summary", {})
            for person, data in results_by_person.items()
        },

        "overall": overall,

        "results_by_person": results_by_person,
    }


def save(path, document):
    """Writes the document as UTF-8 JSON, creating the directory."""

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            document,
            handle,
            ensure_ascii=False,
            indent=2,
        )
