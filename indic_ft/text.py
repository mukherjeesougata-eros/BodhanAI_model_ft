"""Stage 1 -- tokenization.

The shipped IndicCanaryTokenizer DECODES but has no text->ids method, so the
encode side lives here: multilingual SentencePiece ids offset by spl_size (1152).

It also owns text normalisation. That is not cosmetic: measured on IndicTTS
Marathi, 430/10939 raw transcripts contain characters the SP model cannot
represent (ZWJ x249, curly quotes x159, Devanagari digits x62, danda x10,
short-O x3, en-dash, angle quote). Training on an unrepresentable target simply
teaches the decoder to emit <unk>. After the map below plus NFC, exactly 1 row
still fails to round-trip and is dropped at manifest time.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .hub import code_dir

# Out-of-vocab -> in-vocab. Verified against tokenizer_multilingual.model.
CHAR_MAP: dict[int, str] = {
    0x200D: "", 0x200C: "", 0xFEFF: "",            # ZWJ / ZWNJ / BOM
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
    0x2013: "-", 0x2014: "-", 0x2039: "", 0x203A: "",
    0x0964: ".", 0x0965: ".",                      # danda, double danda
    0x0912: "ओ",                                   # Devanagari short O -> O
}
CHAR_MAP.update({0x0966 + i: str(i) for i in range(10)})  # Devanagari -> ASCII digits


def normalize(text: str) -> str:
    """NFC + OOV folding + whitespace collapse. Apply to references at scoring
    time too, or WER punishes the model for input it could never have produced."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text.translate(CHAR_MAP))).strip()


def load_tokenizer(repo_id: str = "", weights: bool = False):
    """From the vendored code -- no download. repo_id/weights kept for call
    compatibility but unused: the tokenizer assets are all local."""
    path = code_dir()                           # must run BEFORE the import below
    from tokenization_indic_canary import IndicCanaryTokenizer
    return IndicCanaryTokenizer.from_pretrained(path)


class TargetEncoder:
    """Transcript -> teacher-forced decoder ids for one language.

    Sequence layout: prompt(10) + transcript + eos. The prompt is the frozen
    canary2 layout the checkpoint was trained with; it is never scored.
    """

    def __init__(self, tokenizer, lang: str):
        self.tok = tokenizer
        self.lang = lang
        self.offset = tokenizer.spl_size          # 1152
        self.prompt = tokenizer.encode_prompt(lang)
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    def encode_text(self, text: str) -> list[int]:
        return [i + self.offset for i in self.tok.multi.encode(text)]

    def round_trips(self, text: str) -> bool:
        return self.tok.decode(self.encode_text(text)) == text

    def build(self, text: str) -> list[int]:
        return self.prompt + self.encode_text(text) + [self.eos_id]


class MultiLangEncoder:
    """One TargetEncoder per language, so a run can mix IndicTTS corpora."""

    def __init__(self, tokenizer, langs):
        self.tok = tokenizer
        self.by_lang = {l: TargetEncoder(tokenizer, l) for l in sorted(set(langs))}
        self.pad_id = tokenizer.pad_id

    def __getitem__(self, lang: str) -> TargetEncoder:
        return self.by_lang[lang]


def audit(rows: list[dict], tokenizer, lang: str) -> dict:
    """Report what normalisation fixes and what it cannot, before training.

    rows: [{"text": ...}, ...] raw transcripts.
    """
    tenc = TargetEncoder(tokenizer, lang)
    unk = tokenizer.multi.unk_id()
    oov = Counter()
    raw_bad = norm_bad = 0
    residual = []

    for r in rows:
        raw = r["text"].strip()
        if not tenc.round_trips(raw):
            raw_bad += 1
        norm = normalize(raw)
        if not tenc.round_trips(norm):
            norm_bad += 1
            if len(residual) < 20:
                residual.append({"text": norm, "decoded": tokenizer.decode(tenc.encode_text(norm))})
        for ch in set(raw):
            if unk in tokenizer.multi.encode(f"क {ch} क"):
                oov[ch] += 1

    return {
        "lang": lang,
        "rows": len(rows),
        "round_trip_failures_raw": raw_bad,
        "round_trip_failures_normalized": norm_bad,
        "oov_characters": {
            f"U+{ord(c):04X}": {"char": c, "name": unicodedata.name(c, "?"), "rows": n}
            for c, n in oov.most_common()
        },
        "residual_examples": residual,
    }
