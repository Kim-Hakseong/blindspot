"""The budget contract: a spending limit that is actually binding.

The limit is fixed when a run starts and cannot be raised while it is running.
That is the whole point -- a cap that the run can lift when it finds the work
interesting is not a cap, and an agent that can widen its own budget has no
budget at all.

Reaching the limit stops the run and preserves what was already bought. It
never overspends, and it never silently drops probes to fit.

This module is on the judgment path: it decides what a run may do, so it has no
model and no cloud dependency, enforced by
`tests/test_no_llm_in_judgment.py`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised when a charge would take a run past its contract."""


class RunState(enum.Enum):
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"


@dataclass(frozen=True)
class Decision:
    """The answer to 'may I spend this?', with the reason recorded."""

    state: RunState
    approved: bool
    reason: str
    requested_probes: int
    would_cost_usd: float


class BudgetContract:
    """A fixed spending limit for one run.

    ``limit_usd`` is read-only after construction: assigning to it raises, so
    the contract cannot be rewritten by the code operating under it.
    """

    __slots__ = ("_limit_usd", "_cost_per_probe_usd", "_spent_usd", "_probes", "_ledger")

    def __init__(self, limit_usd: float, cost_per_probe_usd: float) -> None:
        if limit_usd <= 0.0:
            raise ValueError("limit_usd must be positive")
        if cost_per_probe_usd <= 0.0:
            raise ValueError("cost_per_probe_usd must be positive")
        object.__setattr__(self, "_limit_usd", float(limit_usd))
        object.__setattr__(self, "_cost_per_probe_usd", float(cost_per_probe_usd))
        object.__setattr__(self, "_spent_usd", 0.0)
        object.__setattr__(self, "_probes", 0)
        object.__setattr__(self, "_ledger", [])

    # -- read-only contract terms -----------------------------------------

    @property
    def limit_usd(self) -> float:
        return self._limit_usd

    @property
    def cost_per_probe_usd(self) -> float:
        return self._cost_per_probe_usd

    def __setattr__(self, name, value):
        raise AttributeError(
            f"{name!r} is fixed for the lifetime of the run; a budget that can be "
            "raised mid-run is not a budget"
        )

    # -- state -------------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def probes_charged(self) -> int:
        return self._probes

    @property
    def remaining_usd(self) -> float:
        return max(self._limit_usd - self._spent_usd, 0.0)

    @property
    def probes_remaining(self) -> int:
        """How many further probes fit, rounded down. Never optimistic."""
        return int(self.remaining_usd / self._cost_per_probe_usd + 1e-9)

    @property
    def ledger(self) -> list[dict]:
        return list(self._ledger)

    # -- operations --------------------------------------------------------

    def can_afford(self, probes: int) -> bool:
        cost = probes * self._cost_per_probe_usd
        return self._spent_usd + cost <= self._limit_usd + 1e-9

    def request(self, probes: int) -> Decision:
        """Ask whether a spend is permitted. Never spends by itself."""
        cost = probes * self._cost_per_probe_usd
        if self.can_afford(probes):
            return Decision(
                state=RunState.RUNNING,
                approved=True,
                reason=f"within contract: {cost:.4f} of {self.remaining_usd:.4f} remaining",
                requested_probes=probes,
                would_cost_usd=cost,
            )
        return Decision(
            state=RunState.AWAITING_APPROVAL,
            approved=False,
            reason=(
                f"exceeds budget contract: {probes} probes cost {cost:.4f} USD but only "
                f"{self.remaining_usd:.4f} USD remains of a {self._limit_usd:.4f} USD limit"
            ),
            requested_probes=probes,
            would_cost_usd=cost,
        )

    def charge(self, probes: int) -> float:
        """Spend for ``probes``. Raises rather than exceeding the contract."""
        decision = self.request(probes)
        self._ledger.append(
            {
                "event": "charge",
                "probes": probes,
                "cost_usd": decision.would_cost_usd,
                "granted": decision.approved,
                "reason": decision.reason,
                "spent_after_usd": self._spent_usd
                + (decision.would_cost_usd if decision.approved else 0.0),
            }
        )
        if not decision.approved:
            raise BudgetExceeded(decision.reason)

        object.__setattr__(self, "_spent_usd", self._spent_usd + decision.would_cost_usd)
        object.__setattr__(self, "_probes", self._probes + probes)
        return self._spent_usd

    def describe(self) -> dict:
        return {
            "limit_usd": self._limit_usd,
            "cost_per_probe_usd": self._cost_per_probe_usd,
            "spent_usd": round(self._spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "probes_charged": self._probes,
            "probes_remaining": self.probes_remaining,
        }
