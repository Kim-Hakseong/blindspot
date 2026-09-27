"""Decision ledger (rule A3): every agent tool call, accepted or rejected.

Record shape: {tool, input, output, rationale, accepted_by_scheduler}, plus a
sequence number and timestamp. Local runs append JSONL; cloud runs write the
same records to bs-decisions.
"""

from __future__ import annotations

import json
import pathlib
import time


class DecisionLedger:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = sum(1 for _ in self.path.open()) if self.path.is_file() else 0

    def record(self, entry: dict) -> dict:
        self.seq += 1
        full = {"seq": self.seq, "at": round(time.time(), 3), **entry}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(full, sort_keys=True, default=str) + "\n")
        return full
