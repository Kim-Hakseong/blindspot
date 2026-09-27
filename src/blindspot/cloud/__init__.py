"""Cloud plumbing: the probe worker, ledger sinks, and AWS clients.

This package may talk to AWS. The judgment packages (metrics, boundary, cost,
degrade, measure) may not import it; tests/test_no_llm_in_judgment.py enforces
that boto3 never reaches them.
"""
