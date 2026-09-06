"""Loading a validation set and its ground truth.

Only the categories the set is labelled for are scored. A detector that
correctly finds a handbag in a set annotated only for vehicles and pedestrians
would otherwise be punished for being right, and that noise would move the
boundary we report.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import cv2
import numpy as np

from ..metrics import GroundTruth


@dataclass(frozen=True)
class Frame:
    image_id: str
    path: pathlib.Path
    image: np.ndarray
    truths: tuple[GroundTruth, ...]


class ValidationSet:
    """A manifest-described set of frames with labels."""

    def __init__(self, root: str | pathlib.Path, class_names: tuple[str, ...]) -> None:
        self.root = pathlib.Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path} not found. Build it with:\n"
                f"  uv run python tools/fetch_dataset.py --name {self.root.name}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Model class index <- label name, for the categories this set covers.
        self.name_to_index = {name: i for i, name in enumerate(class_names)}
        self.scored_labels = frozenset(
            self.name_to_index[name]
            for name in self.manifest["categories"]
            if name in self.name_to_index
        )

    @property
    def name(self) -> str:
        return self.manifest["name"]

    def __len__(self) -> int:
        return len(self.manifest["images"])

    def load(self, limit: int | None = None) -> list[Frame]:
        frames: list[Frame] = []
        for record in self.manifest["images"][: limit or len(self.manifest["images"])]:
            path = self.root / "images" / record["file_name"]
            if not path.is_file():
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue

            truths = tuple(
                GroundTruth(
                    image_id=record["image_id"],
                    label=self.name_to_index[obj["label"]],
                    box=tuple(obj["box"]),
                )
                for obj in record["objects"]
                if obj["label"] in self.name_to_index
            )
            frames.append(Frame(record["image_id"], path, image, truths))
        return frames

    def filter_predictions(self, detections):
        """Drop predictions for classes this set carries no labels for."""
        return [d for d in detections if d.label in self.scored_labels]
