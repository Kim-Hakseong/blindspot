"""The report schema: what a run publishes and what the viewer reads.

Three rules are enforced by the types, so a report that breaks them cannot be
written at all:

- **R1** `uncovered_regions` is the first field. What was never tested comes
  before anything that passed.
- **R2** every boundary finding carries `{axis, value, unit, seed,
  source_frame, command}`. A boundary without its recipe is an anecdote.
- A quantity that was not measured is the literal string ``"not measured"``.
  Zero, null, blank and dashes are rejected, because a chart will happily
  draw any of them as a measurement.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NOT_MEASURED = "not measured"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Interval(_Strict):
    from_: float = Field(alias="from")
    to: float
    unit: str

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class UncoveredRegion(_Strict):
    axis: str
    unit: str
    coverage_percent: float = Field(ge=0.0, le=100.0)
    uncovered: list[Interval]
    inferred_from: str | None = None


class Run(_Strict):
    run_id: str
    pipeline: str
    dataset: str
    frames: int
    objects: int
    seed: int
    baseline_map50: float
    threshold_map50: float
    criterion: str


class Reproduce(_Strict):
    axis: str
    value: float
    unit: str
    seed: int
    source_frame: str
    command: str


class Finding(_Strict):
    axis: str
    unit: str
    status: Literal["located", "passes_throughout", "fails_throughout",
                    "not_monotone", "budget_exhausted"]
    lower: float | None = None
    upper: float | None = None
    probes_used: int | None = None
    reproduce: Reproduce

    @model_validator(mode="after")
    def _interval(self):
        if self.status == "located":
            if self.lower is None or self.upper is None:
                raise ValueError("a located boundary needs both ends of its interval")
            if self.lower > self.upper:
                raise ValueError("boundary interval must be ordered lower <= upper")
        return self


class Measured(_Strict):
    value: float
    unit: str
    source: str


Measurement = Union[Measured, Literal["not measured"]]


class Measurements(_Strict):
    sim2real_gap: Measurement
    cool_vs_x86: Measurement


class Report(_Strict):
    # Field order is the serialisation order. uncovered_regions stays first (R1).
    uncovered_regions: list[UncoveredRegion]
    run: Run
    findings: list[Finding]
    map2d: dict | None
    curves: list[dict]
    efficiency: dict | None
    evidence_frames: list[dict]
    measurements: Measurements
    limitations: list[str] = Field(min_length=5)

    @field_validator("limitations")
    @classmethod
    def _nonblank(cls, v):
        if any(not item.strip() for item in v):
            raise ValueError("limitations must not be blank")
        return v

    def dump_json(self) -> str:
        return self.model_dump_json(indent=1, by_alias=True)
