"""2-D boundary sampling by level-set estimation (W3-3).

Tested on synthetic response surfaces whose pass/fail regions are known
exactly: a two-sided window (the shape the exposure x illuminance map has), a
diagonal edge, and degenerate all-pass / all-fail surfaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from blindspot.boundary.levelset import LevelSetResult, estimate_level_set

N = 17
THR = 0.5


def surface(fn):
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
    return fn(ii / (N - 1), jj / (N - 1))


def window(x, y):
    # Passes inside an off-centre ellipse; smooth falloff like mAP.
    r = ((x - 0.3) / 0.25) ** 2 + ((y - 0.35) / 0.3) ** 2
    return 0.9 * np.exp(-0.7 * r)


def diagonal(x, y):
    return 0.2 + 0.6 * (1.0 - (x + y) / 2.0) ** 1.5


def oracle(values):
    calls = []

    def evaluate(i, j):
        calls.append((i, j))
        return float(values[j, i])

    return evaluate, calls


def truth(values):
    return values < THR  # failed


def cell_errors(result: LevelSetResult, values):
    wrong = np.argwhere(result.failed != truth(values))
    return wrong


def distance_to_true_boundary(values, cells):
    """Chebyshev distance from each cell to the nearest cell of opposite truth."""
    t = truth(values)
    out = []
    for j, i in cells:
        opposite = np.argwhere(t != t[j, i])
        out.append(int(np.min(np.max(np.abs(opposite - [j, i]), axis=1))) if len(opposite) else 99)
    return out


@pytest.mark.parametrize("fn", [window, diagonal], ids=["window", "diagonal"])
def test_classifies_every_cell_within_one_cell_of_the_boundary(fn):
    values = surface(fn)
    evaluate, _ = oracle(values)
    result = estimate_level_set(evaluate, N, N, THR)
    wrong = cell_errors(result, values)
    assert all(d <= 1 for d in distance_to_true_boundary(values, wrong)), wrong


@pytest.mark.parametrize("fn", [window, diagonal], ids=["window", "diagonal"])
def test_uses_far_fewer_probes_than_the_grid(fn):
    evaluate, calls = oracle(surface(fn))
    result = estimate_level_set(evaluate, N, N, THR)
    assert result.probes_used == len(set(calls))
    assert result.probes_used < N * N / 2


def test_never_probes_a_cell_twice():
    evaluate, calls = oracle(surface(window))
    estimate_level_set(evaluate, N, N, THR)
    assert len(calls) == len(set(calls))


def test_is_deterministic():
    values = surface(window)
    a = estimate_level_set(oracle(values)[0], N, N, THR)
    b = estimate_level_set(oracle(values)[0], N, N, THR)
    assert a.probes == b.probes
    assert np.array_equal(a.failed, b.failed)


def test_respects_the_probe_budget():
    result = estimate_level_set(oracle(surface(window))[0], N, N, THR, max_probes=12)
    assert result.probes_used <= 12
    assert result.status == "budget_exhausted"


def test_sampled_cells_are_classified_by_their_measurement_not_the_model():
    values = surface(diagonal)
    result = estimate_level_set(oracle(values)[0], N, N, THR)
    for (i, j), v in result.probes.items():
        assert result.failed[j, i] == (v < THR)


def test_all_pass_surface_is_reported_without_a_boundary():
    values = np.full((N, N), 0.9)
    result = estimate_level_set(oracle(values)[0], N, N, THR)
    assert not result.failed.any()
    assert result.probes_used <= 12


def test_all_fail_surface():
    values = np.full((N, N), 0.1)
    result = estimate_level_set(oracle(values)[0], N, N, THR)
    assert result.failed.all()


def test_posterior_is_finite_on_the_surfaces_it_classifies():
    """numpy on Apple Accelerate prints overflow warnings from GP matmuls. The
    search now raises on a non-finite posterior instead of classifying with
    it; reaching a result on these surfaces proves the posterior was finite."""
    for fn in (window, diagonal):
        result = estimate_level_set(oracle(surface(fn))[0], N, N, THR)
        assert result.status == "converged"
