"""The shared plain-text run log.

TensorBoard's event files are protobuf and the console scrolls away, so every
stage also appends here. Stage 1 has no Trainer to hang a callback off, so it
simply mirrors stdout; stage 2 adds structured lines through
training.TextLogger. Both append to the same file, in order, behind a banner.

Deliberately free of torch and transformers: stage 1 imports this, and
importing training.py would drag in the whole stack for a text-only job.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class _Tee:
    """Write to the terminal and the log at once."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.fh.flush()

    # tqdm and friends probe these before deciding how to render
    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()


def append(path, message: str) -> None:
    """Append one timestamped line to the run log.

    For notes that are neither a metric row nor captured stdout -- e.g. where the
    final checkpoint landed -- so the log records the same milestones the console
    shows during training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now():%H:%M:%S}  {message}\n")


@contextmanager
def tee_stdout(path, title: str, header: dict | None = None):
    """Mirror everything printed inside the block into `path`, appending."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", buffering=1, encoding="utf-8")
    fh.write(f"\n{'=' * 78}\n{title} -- {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    for k, v in (header or {}).items():
        fh.write(f"  {k:24} {v}\n")
    fh.write(f"{'=' * 78}\n")

    saved = sys.stdout
    sys.stdout = _Tee(saved, fh)
    try:
        yield
    finally:
        sys.stdout = saved
        fh.close()
