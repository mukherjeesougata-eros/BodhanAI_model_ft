"""Stage 2 -- manifest and split.

Produces one jsonl row per usable utterance, then a train/val split that is
WRITTEN TO DISK rather than recomputed from a seed. That matters: the manifest
is cached, so a split recomputed at each run would silently move eval clips into
train the moment anyone changed --seed, quietly invalidating every metric.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from .config import Config, CorpusSpec
from .text import TargetEncoder


def read_metadata(spec: CorpusSpec) -> list[dict]:
    with open(spec.metadata, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build(cfg: Config, tokenizer, verbose: bool = True) -> list[dict]:
    """Scan every corpus, filter, and write cfg.manifest_path."""
    cfg.ensure_dirs()
    rows_out: list[dict] = []
    stats: dict[str, dict] = {}

    for spec in cfg.corpora:
        tenc = TargetEncoder(tokenizer, spec.lang)
        rows = read_metadata(spec)
        dropped = {"unrepresentable": 0, "too_short": 0, "too_long": 0, "missing": 0}

        for i, r in enumerate(rows):
            if verbose and i % 2000 == 0:
                print(f"  [{spec.lang}] {i}/{len(rows)}", flush=True)
            path = spec.resolve(r["file_name"])
            # Judge the transcript as distributed. A row is kept only if the
            # model's own tokenizer encodes it and decodes it back byte for
            # byte; anything it cannot represent is dropped rather than
            # rewritten, so no orthographic decision is imposed on the corpus.
            text = r["text"].strip()
            if not tenc.round_trips(text):
                dropped["unrepresentable"] += 1
                continue
            try:
                info = sf.info(path)
            except Exception:
                dropped["missing"] += 1
                continue
            dur = info.frames / info.samplerate
            if dur < cfg.min_seconds:
                dropped["too_short"] += 1
                continue
            if dur > cfg.max_seconds:
                dropped["too_long"] += 1
                continue
            rows_out.append({
                "path": path,
                "text": text,
                "lang": spec.lang,
                "duration": round(dur, 3),
                "gender": r.get("gender", "unknown"),
            })
        stats[spec.lang] = {"read": len(rows), "dropped": dropped}

    write_jsonl(cfg.manifest_path, rows_out)
    if verbose:
        hours = sum(r["duration"] for r in rows_out) / 3600
        print(f"[manifest] {len(rows_out)} utterances, {hours:.2f}h -> {cfg.manifest_path}")
        for lang, s in stats.items():
            print(f"  {lang}: read {s['read']} dropped {s['dropped']}")
    return rows_out


def split(cfg: Config, rows: list[dict], verbose: bool = True) -> tuple[list[dict], list[dict]]:
    """Grouped by transcript, deterministic, and persisted.

    Utterances sharing a transcript MUST land on the same side. IndicTTS has each
    sentence read by several speakers -- 6312 of 10939 Marathi rows are part of
    such a group -- and this is a decoder fine-tune, so a transcript seen during
    training is memorised by the very component being trained. Splitting per
    utterance leaked 192 of 328 val transcripts into train and inflated the metric.

    Gender needs no explicit stratification: most groups already contain both
    speakers, so both voices land in val on their own.

    Note the split is still in-domain by construction -- same speakers and
    recording chain as training. Sound for checkpoint selection, NOT a
    generalisation estimate; use a separate corpus for a headline WER.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["lang"], r["text"])].append(r)
    buckets: dict[str, list[list[dict]]] = defaultdict(list)
    for (lang, _), items in groups.items():
        buckets[lang].append(items)

    rng = random.Random(cfg.seed)
    train, val = [], []
    for lang in sorted(buckets):
        gs = sorted(buckets[lang], key=lambda g: g[0]["path"])   # stable before shuffling
        rng.shuffle(gs)
        target = cfg.val_frac * sum(len(g) for g in gs)
        n = 0
        for g in gs:
            if n < target:
                val.extend(g)
                n += len(g)
            else:
                train.extend(g)
    rng.shuffle(train)
    rng.shuffle(val)

    write_jsonl(cfg.train_split, train)
    write_jsonl(cfg.val_split, val)
    if verbose:
        print(f"[split] train {len(train)} ({sum(r['duration'] for r in train)/3600:.2f}h) "
              f"-> {cfg.train_split}")
        print(f"[split] val   {len(val)} ({sum(r['duration'] for r in val)/3600:.2f}h) "
              f"-> {cfg.val_split}")
        overlap = {r["path"] for r in train} & {r["path"] for r in val}
        assert not overlap, f"{len(overlap)} utterances leaked across the split"
        shared = {r["text"] for r in train} & {r["text"] for r in val}
        assert not shared, f"{len(shared)} transcripts appear on both sides of the split"
    return train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
