"""Disk-backed probe results, keyed by everything that determines them.

A probe is a pure function of (dataset, frame count, pipeline, condition,
seed), so running it twice buys nothing. Persisting results makes long sweeps
resumable and keeps a re-run from paying again for work already done -- the
same argument the tool makes about cloud spend, applied to itself.

Storage is bookkeeping, not judgment: what counts as a failure is still
decided by `criterion.py` on the stored mAP.
"""

from __future__ import annotations

import json
import pathlib


class ProbeStore:
    def __init__(self, path: pathlib.Path | str):
        self.path = pathlib.Path(path)
        self.data: dict[str, dict] = {}
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(dataset: str, n_frames: int, pipeline: str, condition: list[dict], seed: int) -> str:
        cond = json.dumps(condition, sort_keys=True, separators=(",", ":"))
        return f"{dataset}|{n_frames}|{pipeline}|{cond}|{seed}"

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, record: dict) -> None:
        self.data[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
