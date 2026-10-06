"""One line per decision-ledger record, for a person reading a run's trace."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from show_decisions import line  # noqa: E402


def test_a_model_call_shows_its_charge_and_what_is_left():
    rec = {"decision_id": "0000-agent-003", "tool": "agent.model_call", "accepted_by_scheduler": True,
           "input": {"call": 2}, "output": {"usd": 0.00203, "contract_remaining_usd": 0.39658}}
    s = line(rec)
    assert "agent.model_call" in s and "0.00203" in s and "0.39658" in s


def test_a_refused_proposal_shows_its_size_and_the_gates_reason():
    rec = {"decision_id": "0000-agent-006", "tool": "reallocate_budget", "accepted_by_scheduler": False,
           "input": {"probes_per_axis": {"a": 150, "b": 49}},
           "output": {"reason": "exceeds budget contract: 199 probes cost 0.3980 USD but only 0.3933 USD remains"}}
    s = line(rec)
    assert "REFUSED" in s and "199 probes" in s and "0.3933" in s


def test_the_envelope_read_shows_the_measured_boundaries_the_agent_saw():
    rec = {"decision_id": "0000-agent-002", "tool": "get_envelope", "accepted_by_scheduler": True,
           "input": {}, "output": {"findings": [{"axis": "fog.beta_per_m", "lower": 0.06, "upper": 0.06375,
                                                 "unit": "1/m"}]}}
    assert "fog.beta_per_m 0.06–0.06375 1/m" in line(rec)
