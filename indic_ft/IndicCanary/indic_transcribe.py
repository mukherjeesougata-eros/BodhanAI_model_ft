# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Indic Transcribe — the short way in.

Three lines::

    from indic_transcribe import IndicTranscribe
    asr = IndicTranscribe.from_pretrained("<downloaded repo dir>")
    print(asr("clip.wav", lang="hi"))

Three ways to handle the language, in this order of precedence:

1. ``lang="hi"``      -- used as given. Identification never runs.
2. ``lang=None``      -- identification fills it, then transcription proceeds.
                         It costs one decoder step, not a second encoder pass.
3. ``asr.identify()`` -- identification only, returning ranked candidates.

A supplied language always wins, and supplying one is usually right:
identification scores 0.86 on the lattice benchmark and 0.78 on VOI (337k clips
in total), but that average hides a very uneven spread -- ml/ta 0.98, kn 0.97,
bn 0.96, against bho 0.05, hi 0.26, mai 0.36 and ur 0.49, each absorbed by a
close neighbour. A wrong language yields confidently wrong *script* rather than
obvious errors, so ``lang=None`` warns when it lands on one of the weak classes.

This wrapper deliberately does NOT chunk long audio. The checkpoint trains at
``max_duration: 30``; past roughly that length a single decoding pass collapses
into repetition rather than degrading gracefully (on a 154 s file, 4-gram repetition
rises 40% -> 91% and WER 88% -> 249% as the token budget is raised, so the extra
tokens are loops, not speech). The decoder also has a fixed 1024-position table, so
no single pass covers long audio at any setting. Anything longer than
``max_seconds`` therefore raises rather than returning a fluent wrong transcript. For long audio use ``long_form.transcribe_long``, which ships in this
repository: it cuts the audio at natural pauses into <=30 s pieces and decodes each.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence

import torch

# generation_config.json sets max_length=1024 (the decoder's position table), so every
# generate() call that passes max_new_tokens logs "Both `max_new_tokens` and `max_length`
# seem to have been set" -- once per chunk on long audio. max_new_tokens takes precedence
# either way; this drops only that line, and only while our own generate() runs.
_GEN_LOG = logging.getLogger("transformers.generation.utils")


class _QuietLengthClash(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ("`max_new_tokens`" in msg and "seem to have been set" in msg)


_QUIET_LENGTH_CLASH = _QuietLengthClash()


try:
    from .lid import RECOMMENDED_LANGS, TRAINED_LANGS, lid_from_encoder_states  # noqa: F401
    from .feature_extraction_indic_canary import IndicCanaryFeatureExtractor
    from .modeling_indic_canary import IndicCanaryForConditionalGeneration
    from .tokenization_indic_canary import IndicCanaryTokenizer
except ImportError:  # flat import from a downloaded directory
    from lid import RECOMMENDED_LANGS, TRAINED_LANGS, lid_from_encoder_states  # noqa: F401
    from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
    from modeling_indic_canary import IndicCanaryForConditionalGeneration
    from tokenization_indic_canary import IndicCanaryTokenizer

#: mode name -> (itn, romanized), the two prompt slots that select the output.
MODES: dict[str, tuple[bool, bool]] = {
    "native": (False, False),      # native script (default)
    "mixed": (True, False),        # mixed script: Latin loanwords, digits
    "romanised": (False, True),    # full Latin transliteration
}
MODES["romanized"] = MODES["romanised"]  # both spellings
MODES["itn"] = MODES["mixed"]

#: Identification is markedly worse for these -- a close neighbour eats them.
#: Measured top-1: bho 0.047, hi 0.258, mai 0.356, ur 0.490.
WEAK_LID_LANGS = frozenset({"bho", "hi", "mai", "ur"})

#: The 27 languages this checkpoint was trained on.
LANGUAGES = (
    "as", "bgc", "bhb", "bho", "bn", "brx", "doi", "en", "gu", "hi", "hne",
    "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa",
    "sat", "sd", "ta", "te", "ur",
)


class IndicTranscribe:
    """One model, one call. See the module docstring for the three-line form."""

    def __init__(self, model, fe, tokenizer, device: str, max_seconds: float = 45.0):
        self.model, self.fe, self.tokenizer = model, fe, tokenizer
        self.device, self.max_seconds = device, max_seconds

    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None,
                        dtype: torch.dtype | None = None, max_seconds: float = 45.0):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bf16 on GPU, fp32 on CPU: bf16 matmul on CPU is slower, not faster.
        dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
        model = IndicCanaryForConditionalGeneration.from_pretrained(path, dtype=dtype)
        model = model.to(device).eval()
        return cls(model, IndicCanaryFeatureExtractor.from_pretrained(path, device=device),
                   IndicCanaryTokenizer.from_pretrained(path), device, max_seconds)

    # -- audio ---------------------------------------------------------------

    def _load(self, audio) -> torch.Tensor:
        """Path, tensor or array -> mono float32 at the model's rate.

        numpy arrays are accepted because slicing a decoded waveform is the
        obvious way to handle audio longer than ``max_seconds``, and
        ``sf.read`` returns numpy.
        """
        if isinstance(audio, torch.Tensor):
            return audio.float()
        if hasattr(audio, "__array_interface__") or type(audio).__name__ == "ndarray":
            wav = torch.as_tensor(audio, dtype=torch.float32)
            return wav.mean(axis=1) if wav.ndim == 2 else wav
        import soundfile as sf

        wav, sr = sf.read(audio, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))          # downmix to mono
        if sr != self.fe.sample_rate:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, self.fe.sample_rate)
        return wav

    def _features(self, wav: torch.Tensor):
        """Mel features + attention mask for one waveform.

        Short audio is padded to 1 s CENTRED the way training did (production's
        ``pad_direction='both'``), and the padded length is what the encoder is
        told. Identification and transcription share this so they cannot end up
        answering about slightly different audio.
        """
        min_len = self.fe.sample_rate
        n = wav.shape[0]
        if n < min_len:
            batch = torch.zeros(1, min_len)
            off = round((min_len - n) / 2)
            batch[0, off : off + n] = wav
            lens = torch.tensor([min_len])
        else:
            batch, lens = wav.unsqueeze(0), torch.tensor([n])
        feats, feat_lens = self.fe(batch.to(self.device), lens.to(self.device))
        mask = (torch.arange(feats.size(2), device=self.device)[None, :]
                < feat_lens[:, None]).long()
        return feats, mask

    # -- transcription -------------------------------------------------------

    def __call__(self, audio, lang: str | None = None, mode: str = "native", **kw):
        return self.transcribe(audio, lang, mode, **kw)

    def transcribe(self, audio, lang: str | None = None, mode: str = "native",
                   max_new_tokens: int = 512, return_lid: bool = False,
                   allowed_langs: Sequence[str] | None = None, **generate_kwargs):
        """Transcribe one clip. ``mode`` is native / mixed / romanised.

        ``lang`` given -> used as-is. ``lang=None`` -> identification fills it.
        ``return_lid=True`` also returns ``{"lang", "source", "topk"}``, INCLUDING
        when you supplied the language -- so a disagreement between your label
        and the model is visible rather than silent.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(set(MODES))}, got {mode!r}")
        if lang is not None and lang not in LANGUAGES:
            raise ValueError(f"lang must be one of {LANGUAGES}, got {lang!r}")
        itn, romanized = MODES[mode]
        wav = self._load(audio)
        seconds = wav.shape[0] / self.fe.sample_rate
        if seconds > self.max_seconds:
            raise ValueError(
                f"audio is {seconds:.0f}s; this wrapper decodes whole files and the "
                f"checkpoint degrades past ~{self.max_seconds:.0f}s (it trains at "
                "max_duration 30). For long audio use the chunked helper that ships "
                "in this repository:\n"
                "    from long_form import transcribe_long\n"
                "    text = transcribe_long(asr, audio, lang='hi')\n"
                "Or raise max_seconds to accept a truncated transcript."
            )

        feats, mask = self._features(wav)

        lid, enc = None, None
        if lang is None or return_lid:
            # Run the encoder ONCE and hand the states to generate() below, so
            # identification costs a decoder step rather than a second encode.
            with torch.inference_mode():
                enc = self.model.model.encoder(feats, attention_mask=mask)
                top = lid_from_encoder_states(
                    self.model, enc.last_hidden_state, enc.lengths,
                    tokenizer=self.tokenizer, topk=5, allowed_langs=allowed_langs,
                )[0]
            source = "explicit"
            if lang is None:
                lang, source = top[0][0], "lid"
                if lang in WEAK_LID_LANGS:
                    warnings.warn(
                        f"identified {lang!r}, one of this model's weakest classes "
                        "(see the module docstring). Pass lang explicitly if you "
                        "know it -- a wrong language yields wrong script, not "
                        "obvious errors.", stacklevel=2,
                    )
            lid = {"lang": lang, "source": source, "topk": top}

        prompt = self.tokenizer.encode_prompt(lang, itn=itn, romanized=romanized)
        ids = torch.tensor([prompt], device=self.device)
        with torch.inference_mode():
            kw = dict(attention_mask=mask, decoder_input_ids=ids,
                      max_new_tokens=max_new_tokens, **generate_kwargs)
            _GEN_LOG.addFilter(_QUIET_LENGTH_CLASH)
            try:
                out = (self.model.generate(input_features=feats, **kw) if enc is None
                       else self.model.generate(encoder_outputs=enc, **kw))
            finally:
                _GEN_LOG.removeFilter(_QUIET_LENGTH_CLASH)
        # strip with the SAME prompt used to generate -- a mismatch raises here
        text = self.tokenizer.decode(
            self.tokenizer.strip_prompt_and_trim(out[0].tolist(), prompt)
        )
        return (text, lid) if return_lid else text

    def identify(self, audio, topk: int = 5,
                 allowed_langs: Sequence[str] | None = None) -> list[tuple[str, float]]:
        """Identify the language and stop. ``[(lang, prob), ...]``, best first.

        Returns the distribution rather than a bare string on purpose: for the
        confusable pairs the top-1 alone hides how close the decision was.

        ``allowed_langs`` narrows the candidate set. Measured over 337k clips:
        passing ``TRAINED_LANGS`` is a NO-OP -- the model already never emits the
        ~162 language tokens it was not trained on, so it changes not a single
        prediction. ``RECOMMENDED_LANGS`` (the 27 minus bgc/bhb, which only ever
        act as sinks) buys +1.4 points on VOI, almost all of it Punjabi rescued
        from bgc. Neither is the default, because narrowing is a HARD filter:
        audio genuinely in an excluded language is reassigned to the nearest
        permitted one, never flagged.
        """
        feats, mask = self._features(self._load(audio))
        with torch.inference_mode():
            enc = self.model.model.encoder(feats, attention_mask=mask)
            return lid_from_encoder_states(
                self.model, enc.last_hidden_state, enc.lengths,
                tokenizer=self.tokenizer, topk=topk, allowed_langs=allowed_langs,
            )[0]

    def all_modes(self, audio, lang: str | None = None, **kw) -> dict[str, str]:
        """Every output mode for one clip, as ``{mode: text}``."""
        wav = self._load(audio)   # decode the file once
        if lang is None:          # ...and identify once, not once per mode
            lang = self.identify(wav)[0][0]
        return {m: self.transcribe(wav, lang, m, **kw)
                for m in ("native", "mixed", "romanised")}
