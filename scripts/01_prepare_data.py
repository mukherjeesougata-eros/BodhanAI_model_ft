#!/usr/bin/env python3
"""Stage 1 -- tokenization audit, manifest, persisted split.

    python scripts/01_prepare_data.py

Writes into <data_dir>: reports/text_audit.json, manifest.jsonl,
splits/{train,val}.jsonl. Everything downstream reads those files, so this is the
only stage that touches the raw corpus.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indic_ft import manifest as mf
from indic_ft.config import DEFAULT_CONFIG, Config
from indic_ft.runlog import tee_stdout
from indic_ft.text import audit, load_tokenizer


class _Formatter(argparse.ArgumentDefaultsHelpFormatter,
                 argparse.RawDescriptionHelpFormatter):
    """Keep the module docstring intact AND show each default in --help."""


def parse_args() -> Config:
    """Every setting comes from the config file; --config only chooses which one."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="settings file (hand-edited; never written back)")
    return Config.load(p.parse_args().config)


def main() -> int:
    cfg = parse_args()
    # everything printed below is mirrored into the shared run log, which
    # stage 2 then appends its training lines to
    with tee_stdout(cfg.log_dir / "train.log", "data preparation",
                    header={"repo_id": cfg.repo_id,
                            "corpora": [f"{s.lang}:{s.root}" for s in cfg.corpora],
                            "min_seconds": cfg.min_seconds,
                            "max_seconds": cfg.max_seconds,
                            "val_frac": cfg.val_frac,
                            "seed": cfg.seed}):
        return _prepare(cfg)


def _prepare(cfg: Config) -> int:
    print(f"[config] read {DEFAULT_CONFIG}")

    tokenizer = load_tokenizer(cfg.repo_id)
    print(f"[tokenizer] vocab={tokenizer.vocab_size} spl={tokenizer.spl_size} "
          f"multi={tokenizer.multi_size}")
    for spec in cfg.corpora:
        print(f"[tokenizer] prompt({spec.lang}) = "
              f"{tokenizer.encode_prompt(spec.lang)}")

    # ---- stage 1: what can the tokenizer actually represent? ----------------
    reports = {}
    for spec in cfg.corpora:
        rows = mf.read_metadata(spec)
        rep = audit(rows, tokenizer, spec.lang)
        reports[spec.lang] = rep
        print(f"[audit {spec.lang}] round-trip failures: "
              f"{rep['round_trip_failures_raw']} raw -> "
              f"{rep['round_trip_failures_normalized']} after normalisation "
              f"({rep['rows']} rows)")
        for code, d in list(rep["oov_characters"].items())[:8]:
            print(f"    {code} {d['char']!r:6} {d['name'][:34]:36} rows={d['rows']}")
    cfg.audit_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.audit_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    print(f"[audit] -> {cfg.audit_path}")

    # ---- stage 2: manifest + split -----------------------------------------
    rows = mf.build(cfg, tokenizer)
    if not rows:
        print("no usable utterances", file=sys.stderr)
        return 1
    mf.split(cfg, rows)
    #print("\nNext: python scripts/02_preflight.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
