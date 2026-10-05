"""Export one run's decision ledger (bs-decisions) to bench/out/cloud_runs/.

    AWS_PROFILE=blindspot uv run --group cloud python tools/export_decisions.py --run-id <id>

Records are written as stored, numbers as floats, sorted by decision id.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

OUT = pathlib.Path(__file__).resolve().parents[1] / "bench" / "out" / "cloud_runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    table = boto3.resource("dynamodb").Table("bs-decisions")
    items, kwargs = [], {"KeyConditionExpression": Key("run_id").eq(args.run_id)}
    while True:
        page = table.query(**kwargs)
        items += page["Items"]
        if "LastEvaluatedKey" not in page:
            break
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    plain = json.loads(json.dumps(sorted(items, key=lambda i: i["decision_id"]),
                                  default=lambda o: float(o) if isinstance(o, Decimal) else str(o)))
    path = OUT / f"{args.run_id}-decisions.json"
    path.write_text(json.dumps(plain, indent=1) + "\n")
    print(f"{len(plain)} decisions -> {path.relative_to(OUT.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
