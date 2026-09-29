"""Scoring. References are normalised the same way training targets were, so the
model is not punished for characters it could never emit."""
from __future__ import annotations

import torch

from .text import normalize


def wer_cer(refs: list[str], hyps: list[str]) -> dict:
    import jiwer
    refs = [normalize(r) for r in refs]
    hyps = [normalize(h) for h in hyps]
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps), "n": len(refs)}


@torch.no_grad()
def transcribe_rows(model, feature_extractor, tokenizer, rows, device) -> list[str]:
    """Decode through the shipped wrapper so beam/prompt/stripping match production."""
    from indic_transcribe import IndicTranscribe

    was_training = model.training
    model.eval()
    # the front-end holds its window and mel filterbank as buffers; torch.stft
    # refuses to mix a cuda signal with a cpu window
    feature_extractor.to(device)
    asr = IndicTranscribe(model, feature_extractor, tokenizer, str(device))
    out = []
    for r in rows:
        text = asr.transcribe(r["path"], lang=r["lang"])
        out.append(text[0] if isinstance(text, tuple) else text)
    if was_training:
        model.train()
    return out
