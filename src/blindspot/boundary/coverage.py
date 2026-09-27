"""Which parts of condition space the validation set never exercised.

This is the report's first screen, before anything that passed. A validation
set is a sample of capture conditions, and the conditions it happens not to
contain are exactly the ones nobody has evidence about. Reporting the passes
first would invite the reader to conclude the pipeline is fine in regions that
were never looked at.

Method. Each axis has an objective measurement that moves monotonically with
it (Laplacian variance under blur, SNR under illuminance, and so on). We
measure that quantity on the user's *undegraded* frames to find where the set
natively sits, and we know from the sweep what the axis can reach. Inverting
the sweep's monotone mapping turns the native measurement range back into axis
units, so an uncovered region is reported in ms or lux rather than as an
abstract fraction.

Limitation, stated because it changes how the number should be read: this
infers capture conditions from image statistics rather than from capture
metadata. A scene that is intrinsically low-contrast reads as foggier than it
was shot. Where EXIF or a capture log is available it should be preferred, and
that is not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..measure import measure


@dataclass(frozen=True)
class AxisCoverage:
    """How much of one axis the validation set actually spans."""

    axis: str
    unit: str
    axis_range: tuple[float, float]
    covered_range: tuple[float, float] | None
    coverage_fraction: float
    uncovered_regions: list[dict]
    metric: str
    native_metric_range: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "unit": self.unit,
            "axis_range": list(self.axis_range),
            "covered_range": list(self.covered_range) if self.covered_range else None,
            "coverage_fraction": round(self.coverage_fraction, 4),
            "coverage_percent": round(self.coverage_fraction * 100.0, 2),
            "uncovered_regions": self.uncovered_regions,
            "inferred_from": self.metric,
            "native_metric_range": list(self.native_metric_range),
        }


def _invert(sweep_values: np.ndarray, sweep_metric: np.ndarray, target: float) -> float:
    """Axis value whose swept measurement equals ``target``.

    The sweep is monotone in the metric by construction (enforced by
    tests/test_units_roundtrip.py), so this is a 1-D interpolation once the
    metric is put in ascending order.
    """
    order = np.argsort(sweep_metric)
    metric_sorted = sweep_metric[order]
    values_sorted = sweep_values[order]
    return float(np.interp(target, metric_sorted, values_sorted))


def axis_coverage(
    axis_id: str,
    unit: str,
    axis_range: tuple[float, float],
    metric: str,
    sweep_values: list[float],
    sweep_metric: list[float],
    native_metric_values: list[float],
) -> AxisCoverage:
    """Coverage of one axis, given a sweep and the set's native measurements."""
    values = np.asarray(sweep_values, float)
    metrics = np.asarray(sweep_metric, float)

    native_lo = float(np.min(native_metric_values))
    native_hi = float(np.max(native_metric_values))

    # The sweep can only speak about metric values it actually reached.
    reachable_lo, reachable_hi = float(np.min(metrics)), float(np.max(metrics))
    clipped_lo = max(native_lo, reachable_lo)
    clipped_hi = min(native_hi, reachable_hi)

    lo, hi = axis_range
    span = hi - lo

    if clipped_hi < clipped_lo or span <= 0:
        return AxisCoverage(
            axis=axis_id,
            unit=unit,
            axis_range=axis_range,
            covered_range=None,
            coverage_fraction=0.0,
            uncovered_regions=[
                {"from": lo, "to": hi, "unit": unit, "width": span}
            ],
            metric=metric,
            native_metric_range=(native_lo, native_hi),
        )

    a = _invert(values, metrics, clipped_lo)
    b = _invert(values, metrics, clipped_hi)
    covered_lo, covered_hi = (min(a, b), max(a, b))
    covered_lo = max(covered_lo, lo)
    covered_hi = min(covered_hi, hi)

    fraction = max(covered_hi - covered_lo, 0.0) / span

    uncovered: list[dict] = []
    if covered_lo - lo > span * 1e-6:
        uncovered.append(
            {"from": lo, "to": covered_lo, "unit": unit, "width": covered_lo - lo}
        )
    if hi - covered_hi > span * 1e-6:
        uncovered.append(
            {"from": covered_hi, "to": hi, "unit": unit, "width": hi - covered_hi}
        )

    return AxisCoverage(
        axis=axis_id,
        unit=unit,
        axis_range=axis_range,
        covered_range=(covered_lo, covered_hi),
        coverage_fraction=fraction,
        uncovered_regions=uncovered,
        metric=metric,
        native_metric_range=(native_lo, native_hi),
    )


def native_measurements(frames, metric: str) -> list[float]:
    """The objective measurement of each undegraded validation frame."""
    return [measure(frame.image)[metric] for frame in frames]


def population_sweep(
    frames,
    degradation: str,
    field: str,
    values,
    metric: str,
    seed: int,
) -> list[float]:
    """Median of ``metric`` across ``frames`` at each axis value.

    This is the curve native measurements are inverted through. It must
    describe the population rather than one frame: sharpness and contrast are
    scene-dependent, so a single-frame curve would read another scene's
    texture as a different capture condition. Degradation and measurement
    only -- no inference -- so it costs a fraction of one probe.
    """
    from ..degrade import REGISTRY

    deg = REGISTRY.get(degradation)
    base = {a.field: (a.lo + a.hi) / 2.0 for a in deg.axes}
    medians = []
    for value in values:
        params = deg.Params(**(base | {field: float(value)}))
        samples = [measure(deg.apply(f.image, params, seed=seed))[metric] for f in frames]
        medians.append(float(np.median(samples)))
    return medians
