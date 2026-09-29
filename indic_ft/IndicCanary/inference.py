#!/usr/bin/env python3
"""Indic-Transcribe-core — inference.

The whole API, in three calls:

    import sys
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download("bodhan-ai/indic-transcribe-core")
    sys.path.insert(0, model_dir)          # the model code ships inside the download

    from indic_transcribe import IndicTranscribe
    asr = IndicTranscribe.from_pretrained(model_dir)
    print(asr("audio.wav", lang="hi"))

``snapshot_download`` is required, not decorative: the tokenizer loads its two
SentencePiece files with ``os.path.join``, so ``from_pretrained`` needs a real
directory on disk and will not accept a Hub repo id.

Everything below wraps those in a CLI and routes audio longer than the model's
whole-file limit through the chunked long-form path.

    python inference.py audio.wav --lang hi
    python inference.py audio.wav                 # detect the language first
    python inference.py long_call.wav --lang ta   # chunked automatically
    python inference.py *.wav --lang bn
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from indic_transcribe import IndicTranscribe, LANGUAGES, MODES  # noqa: E402
from long_form import transcribe_long, identify_long                            # noqa: E402

CHUNK_ABOVE_SECONDS = 30.0


def load_model(source: str | None, device: str | None):
    """Local checkout by default; pass --model to pull from the Hub instead."""
    if source is None:
        return IndicTranscribe.from_pretrained(HERE, device=device)
    if os.path.isdir(source):
        return IndicTranscribe.from_pretrained(source, device=device)
    from huggingface_hub import snapshot_download
    return IndicTranscribe.from_pretrained(snapshot_download(source), device=device)


def duration_seconds(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return info.frames / info.samplerate


def main() -> int:
    p = argparse.ArgumentParser(
        description="Transcribe audio with Indic-Transcribe-core.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Long audio is chunked at natural pauses automatically.",
    )
    p.add_argument("audio", nargs="+", help="audio file(s): .wav, .flac or .mp3")
    p.add_argument("--lang", default=None,
                   help=f"language code; omit to auto-detect. One of: {', '.join(LANGUAGES)}")
    p.add_argument("--mode", default="native", choices=sorted(set(MODES)),
                   help="output mode (this checkpoint produces native script)")
    p.add_argument("--model", default=None,
                   help="model dir or Hub id (default: this directory)")
    p.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto)")
    p.add_argument("--show-lang", action="store_true",
                   help="print the detected language alongside the transcript")
    args = p.parse_args()

    if args.lang is not None and args.lang not in LANGUAGES:
        p.error(f"unknown --lang {args.lang!r}; expected one of {', '.join(LANGUAGES)}")

    asr = load_model(args.model, args.device)
    # The model trains on clips of up to 30 s; anything longer goes through the chunked
    # path. (The wrapper itself still accepts whole files up to max_seconds=45.)
    limit = CHUNK_ABOVE_SECONDS
    failures = 0
    many = len(args.audio) > 1

    for path in args.audio:
        if not os.path.exists(path):
            print(f"{path}: not found", file=sys.stderr)
            failures += 1
            continue

        try:
            lang, tag = args.lang, ""
            if lang is None:
                # long files: identify from windows across the file, not the whole file
                import soundfile as _sf, torch as _t
                _d, _sr = _sf.read(path, dtype="float32", always_2d=True)
                _w = _t.as_tensor(_d).mean(dim=1)
                if _sr != 16000:
                    import torchaudio
                    _w = torchaudio.functional.resample(_w, _sr, 16000)
                lang = identify_long(asr, _w)
                tag = f" [detected {lang}]"

            secs = duration_seconds(path)
            if secs > limit:
                text = transcribe_long(asr, path, lang=lang, mode=args.mode)
            else:
                text = asr.transcribe(path, lang=lang, mode=args.mode)
                if isinstance(text, tuple):
                    text = text[0]
        except Exception as e:              # unreadable / corrupt file: report, keep going
            print(f"{path}: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
            continue

        if many or args.show_lang:
            print(f"=== {os.path.basename(path)} ({secs:.0f}s){tag if args.show_lang or tag else ''}")
        print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
