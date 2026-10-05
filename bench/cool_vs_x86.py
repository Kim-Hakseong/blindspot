"""Same probe batch, different CPU or OpenCV build (rule K2).

Thin wrapper: the implementation is `blindspot.stages`, which the AWS Batch
workers run as `blindspot bench-stages`, so local and cloud arms execute the
same code. See LOG W5-1 for the planned arms.

    uv run python bench/cool_vs_x86.py --label local-m4
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from blindspot.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "bench-stages", *sys.argv[1:]]
    app()
