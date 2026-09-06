"""The pipeline-under-test interface.

Blindspot does not build detectors; it interrogates them. Anything that can
turn an image into boxes can be measured, so the contract is deliberately one
method wide. A user wraps their own pipeline in this and everything else --
degradation, probing, boundary search -- works unchanged.
"""

from __future__ import annotations

import abc

import numpy as np

from ..metrics import Detection


class Pipeline(abc.ABC):
    """A detector under test."""

    #: Identifies the pipeline in reports and reproduction commands.
    name: str

    @abc.abstractmethod
    def predict(self, image: np.ndarray, image_id: str) -> list[Detection]:
        """Run the pipeline on one BGR uint8 image."""

    def describe(self) -> dict[str, str]:
        """Provenance for the report: what was actually run."""
        return {"name": self.name}
