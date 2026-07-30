#!/usr/bin/env python3
"""Create a randomized within-instance order for cumulative PDHG ablations."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


CONFIGS = (
    "pdhg",
    "pdhg_restart",
    "pdhg_restart_scaling",
    "pdhg_restart_scaling_reflection",
    "pdhg_restart_scaling_reflection_adaptive_primal_weight",
    "pdhg_restart_scaling_reflection_adaptive",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, default=20260730)
    parser.add_argument("--expected-count", type=int, default=63)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    if len(manifest) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} manifest rows, found {len(manifest)}"
        )

    rng = random.Random(args.order_seed)
    rows: list[dict[str, object]] = []
    task_index = 0
    for instance in manifest:
        configurations = list(CONFIGS)
        rng.shuffle(configurations)
        for within_instance_order, configuration in enumerate(
            configurations, start=1
        ):
            task_index += 1
            rows.append(
                {
                    "task_index": task_index,
                    "instance_id": instance["instance_id"],
                    "filename": instance["filename"],
                    "configuration": configuration,
                    "within_instance_order": within_instance_order,
                    "order_seed": args.order_seed,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "PROGRESSIVE_RUN_ORDER_COMPLETE "
        f"instances={len(manifest)} tasks={len(rows)}"
    )


if __name__ == "__main__":
    main()
