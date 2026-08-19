#!/usr/bin/env python3
"""Direct old-kernel/old-selector versus final-kernel/final-selector A/B."""

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
    """The no-RSOC selector at base commit d0dab7b9."""
    if total_blocks <= 3:
        return "gridWise"
    if total_blocks <= 1_000 or maximum_dimension >= 2_000:
        return "blockWise"
    if total_blocks <= 60_000 or maximum_dimension >= 150:
        return "warpWise"
    return "threadWise"


def read_groups(patterns: list[str]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    paths = sorted({Path(path).resolve() for pattern in patterns
                    for path in glob.glob(pattern)})
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("category") not in ("crossover", "represent_layout"):
                    continue
                if row.get("status") != "OK" or not truth(row.get("correct", "")):
                    continue
                groups[(row.get("device", ""), row.get("input_key", ""))].append(row)
    return groups


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--final", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_groups(args.baseline)
    final = read_groups(args.final)
    comparisons: list[dict[str, object]] = []
    incomplete: list[str] = []
    for key in sorted(baseline.keys() & final.keys()):
        old_rows, new_rows = baseline[key], final[key]
        old_by_strategy = {row["strategy"]: row for row in old_rows}
        new_by_strategy = {row["strategy"]: row for row in new_rows}
        sample = new_rows[0]
        old_strategy = legacy_strategy(
            int(sample["total_blocks"]), int(sample["max_dimension"]),
        )
        final_strategy = sample["natural_strategy"]
        if old_strategy not in old_by_strategy or final_strategy not in new_by_strategy:
            incomplete.append(key[1])
            continue
        old_us = float(old_by_strategy[old_strategy]["latency_median_us"])
        final_us = float(new_by_strategy[final_strategy]["latency_median_us"])
        comparisons.append({
            "device": key[0],
            "input_key": key[1],
            "category": sample["category"],
            "total_blocks": sample["total_blocks"],
            "max_dimension": sample["max_dimension"],
            "legacy_strategy": old_strategy,
            "final_strategy": final_strategy,
            "legacy_latency_us": old_us,
            "final_latency_us": final_us,
            "full_stack_speedup": old_us / final_us,
        })

    if not comparisons:
        raise SystemExit("no complete full-stack comparisons")
    speedups = [float(row["full_stack_speedup"]) for row in comparisons]
    speedup_gmean = geometric_mean(speedups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# Direct full-stack projection A/B",
        "",
        "This comparison uses measured rows directly: the baseline row uses "
        "the base-commit hierarchy selector and bisection artifact; the final "
        "row uses the accepted selector and `final_newton2_v3` artifact.",
        "",
        f"Same-device, same-input comparisons: {len(comparisons)}; final wins: "
        f"{sum(value > 1.0 for value in speedups)}; ties: "
        f"{sum(value == 1.0 for value in speedups)}; regressions: "
        f"{sum(value < 1.0 for value in speedups)}.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Geometric-mean full-stack speedup | {speedup_gmean:.3f}x |",
        f"| Median full-stack speedup | {statistics.median(speedups):.3f}x |",
        f"| Workloads at least 1.5x faster | {sum(value >= 1.5 for value in speedups)} |",
        f"| Incomplete forced-strategy groups (excluded) | {len(incomplete)} |",
        "",
        "The `1.5x` acceptance gate is applied to the geometric-mean speedup.",
        "",
        "| Input | Old | Final | Old us | Final us | Speedup |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(
        comparisons, key=lambda item: float(item["full_stack_speedup"]),
        reverse=True,
    ):
        lines.append(
            f"| {row['input_key']} | {row['legacy_strategy']} | "
            f"{row['final_strategy']} | {float(row['legacy_latency_us']):.3f} | "
            f"{float(row['final_latency_us']):.3f} | "
            f"{float(row['full_stack_speedup']):.3f}x |"
        )
    lines.append("")
    report = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    if speedup_gmean < 1.5:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
