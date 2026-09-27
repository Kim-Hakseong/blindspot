"""2-D failure boundaries by level-set estimation.

Bisection needs a line to walk along. In two dimensions the boundary is a
curve, and the exposure x illuminance map is not monotone in exposure at all:
too short starves the sensor, too long smears the image, so the passing region
is a window with edges on both sides.

This is the Level Set Estimation algorithm (Gotovos, Casati, Hitz and Krause,
IJCAI 2013). A Gaussian process is fitted to the mAP@50 measured so far. Every
unmeasured cell gets a confidence interval ``mu +/- beta*sigma``; a cell whose
whole interval lies on one side of the failure threshold is classified, the
rest are ambiguous. The next probe goes to the most ambiguous cell,

    argmax  beta*sigma - |mu - threshold|,

and the search stops when no ambiguous cell remains. The GP is a *sampling
heuristic*: every cell that was actually probed is classified by its measured
mAP, and only unprobed cells take the model's side. Deterministic: fixed
initial design, kernel hyper-parameters fitted without random restarts.

No model of any kind decides anything here beyond where to look next; pass or
fail is the criterion applied to a measurement or, for unprobed cells, to a
prediction that the result records as such.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

BETA = 1.96


@dataclass
class LevelSetResult:
    failed: np.ndarray                         # [ny, nx] bool
    measured: np.ndarray                       # [ny, nx] bool: probed, not predicted
    probes: dict[tuple[int, int], float] = field(default_factory=dict)
    status: str = "converged"

    @property
    def probes_used(self) -> int:
        return len(self.probes)


def _initial_design(nx: int, ny: int) -> list[tuple[int, int]]:
    """A 3x3 nested lattice: corners, edge midpoints and centre."""
    xs = sorted({0, (nx - 1) // 2, nx - 1})
    ys = sorted({0, (ny - 1) // 2, ny - 1})
    return [(i, j) for j in ys for i in xs]


def estimate_level_set(
    evaluate: Callable[[int, int], float],
    nx: int,
    ny: int,
    threshold: float,
    max_probes: int | None = None,
    beta: float = BETA,
) -> LevelSetResult:
    """Classify an nx-by-ny grid of conditions as pass/fail with few probes.

    ``evaluate(i, j)`` runs the probe at column i, row j and returns mAP@50.
    Coordinates are grid indices scaled to [0, 1], so a geometric axis (lux)
    is modelled in its own natural spacing.
    """
    budget = max_probes if max_probes is not None else nx * ny
    probes: dict[tuple[int, int], float] = {}

    def probe(i: int, j: int) -> None:
        if (i, j) not in probes and len(probes) < budget:
            probes[(i, j)] = float(evaluate(i, j))

    for i, j in _initial_design(nx, ny):
        probe(i, j)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    coords = np.column_stack([ii.ravel() / max(nx - 1, 1), jj.ravel() / max(ny - 1, 1)])
    kernel = (ConstantKernel(0.1, (1e-3, 10.0)) * RBF(0.3, (0.05, 2.0))
              + WhiteKernel(1e-4, (1e-8, 1e-2)))

    status = "converged"
    mu = np.zeros(nx * ny)
    while True:
        keys = sorted(probes)
        X = np.array([[i / max(nx - 1, 1), j / max(ny - 1, 1)] for i, j in keys])
        y = np.array([probes[k] for k in keys])

        if np.ptp(y) < 1e-9:
            # A flat response gives the GP nothing to fit; every probe agrees.
            mu = np.full(nx * ny, y[0])
            sigma = np.zeros(nx * ny)
        else:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=0, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(X, y)
                # numpy 2.x on Apple Accelerate emits spurious floating-point
                # warnings from matmul on finite data; they are silenced here and
                # the result is checked instead, so a genuinely non-finite
                # posterior stops the search rather than classifying cells.
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    mu, sigma = gp.predict(coords, return_std=True)
            if not (np.isfinite(mu).all() and np.isfinite(sigma).all()):
                raise FloatingPointError("non-finite GP posterior; refusing to classify cells")

        acquisition = beta * sigma - np.abs(mu - threshold)
        for i, j in keys:
            acquisition[j * nx + i] = -np.inf
        best = int(np.argmax(acquisition))
        if acquisition[best] <= 0.0:
            break
        if len(probes) >= budget:
            status = "budget_exhausted"
            break
        probe(best % nx, best // nx)

    failed = (mu < threshold).reshape(ny, nx)
    measured = np.zeros((ny, nx), bool)
    for (i, j), value in probes.items():
        failed[j, i] = value < threshold
        measured[j, i] = True
    return LevelSetResult(failed=failed, measured=measured, probes=probes, status=status)
