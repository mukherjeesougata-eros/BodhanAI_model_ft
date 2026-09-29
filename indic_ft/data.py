"""Stage 3 -- dataloading.

Waveform -> NeMo-parity mel features, transcript -> teacher-forced ids.
The front-end runs on CPU in the dataloader workers, so the GPU only ever sees
the frozen encoder's input.
"""
from __future__ import annotations

from dataclasses import dataclass

import soundfile as sf
import torch
from torch.utils.data import Dataset

from .hub import code_dir
from .text import MultiLangEncoder

SAMPLE_RATE = 16000


def load_feature_extractor(repo_id: str = "", weights: bool = True):
    """The audio front-end: waveform -> (mel_bins, time) log-mel, plus resampling
    to 16 kHz. Lives here, not in text.py -- it never touches a transcript.
    Loaded from the vendored code; repo_id/weights are unused (all assets local)."""
    path = code_dir()                           # must run BEFORE the import below
    from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
    return IndicCanaryFeatureExtractor.from_pretrained(path)


class ManifestDataset(Dataset):
    """Rows straight from a split jsonl. `lengths` feeds length-grouped batching."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]

    @property
    def lengths(self) -> list[int]:
        # proxy for padded cost; mel frames ~ samples / hop
        return [int(r["duration"] * SAMPLE_RATE / 160) for r in self.rows]


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))              # downmix to mono
    if sr != sample_rate:
        import torchaudio
        # must be torchaudio's sinc_interp_hann; librosa/soxr do NOT match the
        # resampler the checkpoint was trained and evaluated with
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


@dataclass
class SpeechCollator:
    feature_extractor: object
    encoders: MultiLangEncoder
    sample_rate: int = SAMPLE_RATE

    def __call__(self, batch: list[dict]) -> dict:
        waves = [load_audio(r["path"], self.sample_rate) for r in batch]
        true_lens = [w.shape[0] for w in waves]

        # the front-end raises below one hop; pad short clips to 1s CENTRED,
        # matching inference (production pad_direction='both')
        min_len = self.sample_rate
        lens = [max(n, min_len) for n in true_lens]
        audio = torch.zeros(len(waves), max(lens))
        for i, (w, n) in enumerate(zip(waves, true_lens)):
            off = round((min_len - n) / 2) if n < min_len else 0
            audio[i, off:off + n] = w

        feats, feat_lens = self.feature_extractor(audio, torch.tensor(lens))
        attention_mask = (torch.arange(feats.size(2))[None, :] < feat_lens[:, None]).long()

        seqs, prompt_lens = [], []
        for r in batch:
            tenc = self.encoders[r["lang"]]
            seqs.append(tenc.build(r["text"]))
            prompt_lens.append(tenc.prompt_len)

        T = max(len(s) for s in seqs) - 1
        pad = self.encoders.pad_id
        decoder_input_ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        labels = torch.full((len(seqs), T), -100, dtype=torch.long)
        for i, (s, plen) in enumerate(zip(seqs, prompt_lens)):
            t = torch.tensor(s, dtype=torch.long)
            decoder_input_ids[i, :len(s) - 1] = t[:-1]
            labels[i, :len(s) - 1] = t[1:]
            labels[i, :plen - 1] = -100          # the fixed prompt is never scored
        return {
            "input_features": feats,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
