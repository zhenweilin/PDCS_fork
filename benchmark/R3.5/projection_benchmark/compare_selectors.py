#!/usr/bin/env python3
"""Compare the original and final hierarchy selectors on forced same inputs."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
from collections import defaultdict
from pathlib import Path


def truth(value: str) -> bool:
    return value.lower() == "true"


def legacy_strategy(total_blocks: int, maximum_dimension: int) -> str:
    """The selector at base commit d0dab7b9, for this no-RSOC matrix."""
    if total_blocks <= 3:
        return "gridWise"
    if total_blocks <= 1_000 or maximum_dimension >= 2_000:
        return "blockWise"
    if total_blocks <= 60_000 or maximum_dimension >= 150:
        return "warpWise"
    return "threadWise"


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted({Path(path).resolve() for pattern in args.inputs
                    for path in glob.glob(pattern)})
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("category") not in ("crossover", "represent_layout"):
                    continue
                if row.get("status") != "OK" or not truth(row.get("correct", "")):
                    continue
                key = (
                    row.get("device", ""), row.get("artifact_sha256", ""),
                    row.get("input_key", ""),
                )
                groups[key].append(row)

    comparisons: list[dict[str, object]] = []
    incomplete: list[str] = []
    for (_, _, input_key), rows in sorted(groups.items()):
        by_strategy = {row["strategy"]: row for row in rows}
        sample = rows[0]
        old = legacy_strategy(
            int(sample["total_blocks"]), int(sample["max_dimension"]),
        )
        new = sample["natural_strategy"]
        if old not in by_strategy or new not in by_strategy:
            incomplete.append(input_key)
            continue
        old_us = float(by_strategy[old]["latency_median_us"])
        new_us = float(by_strategy[new]["latency_median_us"])
        oracle = min(float(row["latency_median_us"]) for row in rows)
        comparisons.append({
            "input_key": input_key,
            "category": sample["category"],
            "family": input_key.split("_")[1] if input_key.startswith("crossover_")
                      else "solver_layout",
            "dimension": sample["max_dimension"],
            "structured_cones": sample["structured_cones"],
            "legacy_strategy": old,
            "final_strategy": new,
            "tested_strategies": ";".join(sorted(by_strategy)),
            "legacy_latency_us": old_us,
            "final_latency_us": new_us,
            "oracle_latency_us": oracle,
            "legacy_to_final_speedup": old_us / new_us,
            "final_to_oracle_ratio": new_us / oracle,
        })

    if not comparisons:
        raise SystemExit("no complete selector comparisons")
    speedups = [float(row["legacy_to_final_speedup"]) for row in comparisons]
    oracle_ratios = [float(row["final_to_oracle_ratio"]) for row in comparisons]
    speedup_gmean = geometric_mean(speedups)
    oracle_gmean = geometric_mean(oracle_ratios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = ["# Final hierarchy selector vs original selector", ""]
    lines.append(
        f"Same-device, same-input comparisons: {len(comparisons)}; final wins: "
        f"{sum(value > 1.0 for value in speedups)}; ties: "
        f"{sum(value == 1.0 for value in speedups)}; regressions: "
        f"{sum(value < 1.0 for value in speedups)}."
    )
    lines.extend([
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Geometric-mean original→final speedup | {speedup_gmean:.3f}x |",
        f"| Median original→final speedup | {statistics.median(speedups):.3f}x |",
        f"| Geometric-mean final/oracle latency | {oracle_gmean:.3f}x |",
        f"| Workloads at least 1.5x faster | {sum(value >= 1.5 for value in speedups)} |",
        f"| Incomplete forced-strategy groups (excluded) | {len(incomplete)} |",
        "",
        "The geometric-mean speedup is the primary selector metric. A 1.5x "
        "gate corresponds to the requested approximately-50% improvement in "
        "speed; final/oracle reports how much measured performance remains "
        "available from a per-input oracle selector.",
        "",
        "| Input | Old | Final | Old us | Final us | Speedup | Final/oracle |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(
        comparisons, key=lambda item: float(item["legacy_to_final_speedup"]),
        reverse=True,
    ):
        lines.append(
            f"| {row['input_key']} | {row['legacy_strategy']} | "
            f"{row['final_strategy']} | {float(row['legacy_latency_us']):.3f} | "
            f"{float(row['final_latency_us']):.3f} | "
            f"{float(row['legacy_to_final_speedup']):.3f}x | "
            f"{float(row['final_to_oracle_ratio']):.3f}x |"
        )
    lines.append("")
    report = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    if speedup_gmean < 1.5:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
