"""Model code and weights resolution.

The model's architecture is remote code -- classes defined in .py files, not in
transformers. Those files, plus the tokenizer and feature-extractor assets, are
vendored once into indic_ft/IndicCanary/ and committed, so importing the classes
never touches the network. Only the 4.9 GB weights come from the Hub, and only
the first time on a given machine; afterwards they are read from the local cache
with no Hub contact.

The vendored files import their siblings by bare name (`from
modeling_indic_canary import ...`), so the directory is placed on sys.path rather
than imported as a subpackage -- exactly how a Hub snapshot would be consumed.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
#: vendored model code + tokenizer/feature-extractor assets (committed, ~0.8 MB)
CODE_DIR = REPO_ROOT / "indic_ft" / "IndicCanary"

#: only these come from the Hub; everything else is vendored
WEIGHT_PATTERNS = ["config.json", "generation_config.json", "model.safetensors"]


@lru_cache(maxsize=None)
def code_dir() -> str:
    """Path to the vendored model code, guaranteed importable. No download, ever.

    Returns a real directory (the tokenizer joins its two SentencePiece files
    with os.path.join, so a Hub id will not do) whose classes -- IndicCanary*,
    IndicTranscribe -- can be imported by bare name once it is on sys.path.
    """
    if not (CODE_DIR / "modeling_indic_canary.py").exists():
        raise SystemExit(
            f"vendored model code missing at {CODE_DIR}\n"
            f"Expected the IndicCanary/ files committed with the repo.")
    p = str(CODE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


@lru_cache(maxsize=None)
def weights_dir(repo_id: str = "bodhan-ai/indic-transcribe-core") -> str:
    """Directory holding config.json + model.safetensors.

    Downloaded once per machine, then reused with local_files_only so no Hub
    request is made on subsequent runs. The repo is gated: `huggingface-cli
    login` (or HF_TOKEN) with access granted, for that one first fetch.
    """
    from huggingface_hub import snapshot_download

    try:
        # already cached -> no network at all
        return snapshot_download(repo_id, allow_patterns=WEIGHT_PATTERNS,
                                 local_files_only=True)
    except Exception:
        # first time on this machine: fetch the weights, then it is cached
        return snapshot_download(repo_id, allow_patterns=WEIGHT_PATTERNS)
