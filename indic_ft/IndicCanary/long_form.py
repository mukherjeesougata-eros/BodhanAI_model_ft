"""Long-form transcription for bodhan-ai/indic-transcribe-core.

The wrapper decodes whole files and refuses audio past ~45 s. The checkpoint trains at
max_duration 30; past roughly that length one decoding pass collapses into repetition
(on a 154 s file, WER 88% -> 249% as the token budget rises), and the decoder's fixed
1024-position table caps any single pass. So we cut at natural pauses into <=30 s
pieces and decode each one.

Settings are measured, not tuned by eye. On a 42,764-clip benchmark the cost of cutting
falls with chunk length: cap 10 s +1.89 OIWER, 15 s +0.85, 20 s +0.29, 25 s +0.09,
30 s +0.00. DO NOT LOWER the 30 s cap; raising it past 30 s breaks the model.

Only needs what the model already needs: torch, torchaudio, soundfile.
"""
from __future__ import annotations
import torch, torchaudio


def split_on_silence(wav: torch.Tensor, sr: int = 16000, target: float = 25.0,
                     hard_max: float = 30.0, win_ms: float = 30.0,
                     rel_db: float = 25.0, min_sil: float = 0.25):
    """Cut `wav` into segments <= hard_max seconds, preferring silent boundaries.

    Returns a list of (start_sample, end_sample).
    """
    n = wav.shape[-1]
    if n <= int(hard_max * sr):
        return [(0, n)]

    hop = max(1, int(sr * win_ms / 1000))
    frames = wav[: (n // hop) * hop].reshape(-1, hop)
    rms = frames.pow(2).mean(dim=1).clamp_min(1e-12).sqrt()
    db = 20 * torch.log10(rms)
    thresh = db.max() - rel_db
    quiet = db < thresh                                   # True where silent

    # contiguous silent runs, in frames
    runs, start = [], None
    for i, q in enumerate(quiet.tolist()):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(quiet)))
    min_frames = int(min_sil * sr / hop)
    # midpoint of each sufficiently long silence, as a sample index
    cuts = [((a + b) // 2) * hop for a, b in runs if (b - a) >= min_frames]

    segs, pos = [], 0
    while n - pos > int(hard_max * sr):
        lo, hi = pos + int(5.0 * sr), pos + int(hard_max * sr)
        ideal = pos + int(target * sr)
        usable = [c for c in cuts if lo < c < hi]
        cut = min(usable, key=lambda c: abs(c - ideal)) if usable else hi
        segs.append((pos, cut)); pos = cut
    if pos < n:
        segs.append((pos, n))
    return segs


def identify_long(asr, wav, window: float = 20.0, n: int = 3):
    """Identify the language of audio of any length.

    ``asr.identify`` runs the encoder over its whole input, so on long audio it runs out
    of memory (an hour asked for 61 GiB). Instead, identify ``n`` windows of ``window``
    seconds spread across the file and sum their probabilities.
    """
    sr = 16000
    total = wav.shape[-1]
    w = int(window * sr)
    if total <= int(45 * sr):
        return asr.identify(wav)[0][0]
    starts = [int((total - w) * f) for f in ((i + 1) / (n + 1) for i in range(n))]
    score = {}
    for s0 in starts:
        for lang, prob in asr.identify(wav[s0:s0 + w]):
            score[lang] = score.get(lang, 0.0) + prob
    return max(score, key=score.get)


def transcribe_long(asr, audio, lang: str | None, mode: str = "native", **kw) -> str:
    """Transcribe audio of any length by cutting it at pauses. Returns one string.

    ``audio`` is a file path (any sample rate; downmixed and resampled) or an array /
    tensor that is ALREADY 16 kHz. A 2-D array is downmixed: (samples, channels) as
    ``soundfile`` returns it, or (channels, samples) as ``torchaudio`` does.

    ``lang=None`` identifies the language ONCE, from windows across the file, and uses
    it for every chunk -- identifying per chunk could give different chunks different
    languages.
    """
    if isinstance(audio, str):
        import soundfile as sf
        data, sr = sf.read(audio, dtype="float32", always_2d=True)
        wav = torch.as_tensor(data).mean(dim=1)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
    else:
        wav = torch.as_tensor(audio).float().squeeze()
        if wav.dim() == 2:                  # stereo: the shorter axis is the channels
            wav = wav.mean(dim=0 if wav.shape[0] < wav.shape[1] else 1)
        if wav.dim() != 1:
            raise ValueError(f"expected mono or stereo audio, got shape {tuple(wav.shape)}")
    if lang is None:
        lang = identify_long(asr, wav)

    # A full 30 s chunk can need more than the wrapper's default 256 output tokens --
    # romanised (Latin) text in particular spends several tokens per word. At 256 a
    # 25 s romanised chunk loses its last words silently (measured: 41 words over a
    # 10-minute file). 512 fits a 30 s chunk and stays inside the decoder's fixed
    # 1024-position table. A caller's own max_new_tokens still wins.
    kw.setdefault("max_new_tokens", CHUNK_MAX_NEW_TOKENS)

    out = []
    for a, b in split_on_silence(wav):
        piece = wav[a:b]
        if piece.numel() < 1600:            # skip < 0.1 s
            continue
        out.append(_decode_guarded(asr, piece, lang, mode, kw))
    return " ".join(t.strip() for t in out if t and t.strip())


def _repetition(words, n=4):
    """Fraction of n-gram positions that repeat an earlier n-gram."""
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return (len(grams) - len(set(grams))) / max(1, len(grams))


# A chunk whose output is mostly one phrase repeated has collapsed into a loop --
# e.g. a 25 s chunk returning "ஓகே சார்" 48 times. Normal speech on our benchmarks sits
# near 0-2% 4-gram repetition, so 50% only fires on genuine degeneration.
LOOP_REPETITION = 0.5
CHUNK_MAX_NEW_TOKENS = 512
LOOP_MIN_WORDS = 20


def _collapse_repeats(words, keep=2, max_n=4):
    """Shorten any run where the same 1..max_n-word phrase repeats back to back more than
    `keep` times, keeping `keep` copies. Used only on output that is still looping."""
    out = list(words)
    for n in range(1, max_n + 1):
        res, i = [], 0
        while i < len(out):
            gram = out[i:i + n]
            j = i + n
            reps = 1
            while len(gram) == n and out[j:j + n] == gram:
                reps += 1; j += n
            if reps > keep:
                res.extend(gram * keep); i = j
            else:
                res.append(out[i]); i += 1
        out = res
    return out


def _decode_guarded(asr, piece, lang, mode, kw, depth=0):
    """Decode one chunk; if it loops, split it at a pause and decode the halves.

    Splitting changes what the decoder conditions on and breaks the loop while keeping
    plain greedy decoding. Measured on a looping 25.4 s chunk: 96 words at 97.8%
    repetition -> 43 words at 0% (expected ~43 at that speaker's rate).
    """
    text = asr.transcribe(piece, lang=lang, mode=mode, **kw)
    text = text if isinstance(text, str) else text[0]
    words = text.split()
    looped = len(words) >= LOOP_MIN_WORDS and _repetition(words) > LOOP_REPETITION
    if not looped:
        return text
    if depth >= 2 or piece.numel() < 4 * 16000:
        # Retries did not break the loop. Last resort: collapse the repeated phrase so a
        # stuck loop cannot flood the transcript (measured: a 24.5 s chunk that still
        # looped after two splits returned 293 words, ~250 of them one filler word).
        return " ".join(_collapse_repeats(words))
    half = piece.numel() / 16000 / 2
    parts = split_on_silence(piece, target=half, hard_max=half + 2.0)
    if len(parts) < 2:                              # no pause found: cut in the middle
        m = piece.numel() // 2
        parts = [(0, m), (m, piece.numel())]
    return " ".join(
        t.strip() for t in (_decode_guarded(asr, piece[x:y], lang, mode, kw, depth + 1)
                            for x, y in parts if y - x >= 1600)
        if t and t.strip())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, sys.argv[1])
    from indic_transcribe import IndicTranscribe
    asr = IndicTranscribe.from_pretrained(sys.argv[1])
    print(transcribe_long(asr, sys.argv[2], lang=sys.argv[3]))
