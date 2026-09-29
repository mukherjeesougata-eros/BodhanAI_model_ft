"""Loader for conf/config.json -- the only source of settings.

Nothing in this file supplies a value. It knows the *names* of the keys the code
reads (it has to, to read them), and it derives the paths that are always inside
data_dir so they cannot drift out of sync with it.

Three path inputs, all in the JSON:
  data_dir   everything stage 1 produces lives here
  ckpt_dir   training checkpoints
  log_dir    TensorBoard event files

ckpt_dir and log_dir are joined to data_dir when written relative
("checkpoints"), and used as-is when written absolute ("/mnt/big/ckpts"), so
either can be pushed to another disk without moving the data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
#: the hand-edited config every script reads by default
DEFAULT_CONFIG = REPO_ROOT / "conf" / "config.json"

#: path inputs read from the JSON; relative ones resolve against data_dir
PATH_KEYS = ("ckpt_dir", "log_dir", "final_ckpt_for_inference")
REQUIRED = ("repo_id", "data_dir", "corpora", "gpus") + PATH_KEYS
#: computed from data_dir -- putting these in the JSON is an error, not an override
DERIVED = ("manifest_path", "train_split", "val_split", "audit_path",
           "reports_dir", "final_dir")


@dataclass
class CorpusSpec:
    """One language directory: <root>/<split_dir>/{metadata.csv,audio/*.wav}."""
    root: str
    lang: str                       # prompt language code, e.g. "mr"
    split_dir: str = "train"

    @property
    def metadata(self) -> Path:
        return Path(self.root) / self.split_dir / "metadata.csv"

    def resolve(self, file_name: str) -> str:
        return str(Path(self.root) / self.split_dir / file_name)


class Config(SimpleNamespace):
    """Attribute view over conf/config.json. cfg.max_steps is raw["max_steps"]."""

    # ---- derived: always inside data_dir ------------------------------------
    @property
    def manifest_path(self) -> Path: return self.data_dir / "manifest.jsonl"
    @property
    def train_split(self) -> Path: return self.data_dir / "splits" / "train.jsonl"
    @property
    def val_split(self) -> Path: return self.data_dir / "splits" / "val.jsonl"
    @property
    def reports_dir(self) -> Path: return self.data_dir / "reports"
    @property
    def audit_path(self) -> Path: return self.reports_dir / "text_audit.json"
    @property
    def final_dir(self) -> Path:
        """Kept so an old config naming it still fails loudly rather than
        silently writing somewhere else. Use final_ckpt_for_inference."""
        return self.final_ckpt_for_inference

    # ---- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "Config":
        path = Path(path)
        if not path.exists():
            raise SystemExit(
                f"config not found: {path}\n"
                f"Expected the hand-edited settings file. Create it, or pass --config."
            )
        raw = json.loads(path.read_text())

        missing = [k for k in REQUIRED if k not in raw]
        if missing:
            raise SystemExit(f"{path}: missing required key(s): {', '.join(missing)}")
        # a property would silently win over the JSON value, so say so out loud
        clash = [k for k in DERIVED if k in raw]
        if clash:
            raise SystemExit(
                f"{path}: {', '.join(clash)} is derived from data_dir -- remove it")

        gpus = raw["gpus"]
        if not isinstance(gpus, list) or not gpus or not all(isinstance(g, int) for g in gpus):
            raise SystemExit(f'{path}: "gpus" must be a non-empty list of ints, e.g. [0] or [0, 1]')

        raw["corpora"] = [CorpusSpec(**c) for c in raw["corpora"]]
        data_dir = Path(raw["data_dir"]).expanduser()
        raw["data_dir"] = data_dir
        for k in PATH_KEYS:
            p = Path(raw[k]).expanduser()
            raw[k] = p if p.is_absolute() else data_dir / p
        return cls(**raw)

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.data_dir / "splits", self.reports_dir,
                  self.ckpt_dir, self.log_dir, self.final_ckpt_for_inference):
            p.mkdir(parents=True, exist_ok=True)
