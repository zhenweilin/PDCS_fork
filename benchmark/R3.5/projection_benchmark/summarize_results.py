#!/usr/bin/env python3
"""Merge projection stress shards and summarize correctness/performance.

Only Python's standard library is used so the result directory remains
self-contained. The script never edits solver source; strategy winners are an
auditable input to the subsequent selector patch and PDCS validation gate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics
from collections import Counter, defaultdict


STRATEGY_ORDER = {"gridWise": 0, "blockWise": 1, "warpWise": 2, "threadWise": 3}


def as_float(row, key, default=math.nan):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def is_true(row, key):
    return row.get(key, "").lower() == "true"


def median(values):
    values = [value for value in values if math.isfinite(value)]
    return statistics.median(values) if values else math.nan


def geometric_mean(values):
    values = [value for value in values if value > 0 and math.isfinite(value)]
    return math.exp(statistics.mean(map(math.log, values))) if values else math.nan


def load_rows(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    paths = sorted(set(os.path.abspath(path) for path in paths))
    rows = []
    for path in paths:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                row["source_csv"] = path
                rows.append(row)
    return paths, rows


def write_csv(path, rows, columns=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def strategy_winners(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("category") not in ("crossover", "represent_layout") or \
                row.get("status") != "OK":
            continue
        if not is_true(row, "correct"):
            continue
        latency = as_float(row, "latency_median_us")
        if math.isfinite(latency) and latency > 0:
            # Some very large real layouts need separate invocations because a
            # single grid-wise sample can take tens of seconds.  Combine those
            # invocations only when input, GPU, and artifact are identical;
            # synthetic crossover suites retain their variant separation.
            if row.get("category") == "represent_layout":
                variant = "represent_same_device_artifact_{}_{}".format(
                    row.get("device", "unknown"),
                    row.get("artifact_sha256", "unknown")[:12],
                )
            else:
                variant = row.get("variant", "")
            groups[(variant, row.get("input_key", ""))].append(row)
    winners = []
    for (variant, input_key), candidates in sorted(groups.items()):
        candidates.sort(key=lambda row: (
            as_float(row, "latency_median_us"),
            STRATEGY_ORDER.get(row.get("strategy", ""), 99),
        ))
        winner = candidates[0]
        natural = next((row for row in candidates
                        if row.get("strategy") == row.get("natural_strategy")), None)
        natural_latency = as_float(natural, "latency_median_us") if natural else math.nan
        winner_latency = as_float(winner, "latency_median_us")
        runners = sorted(as_float(row, "latency_median_us") for row in candidates)
        second = runners[1] if len(runners) > 1 else math.nan
        match = re.match(r"^crossover_(soc|exp|dual_exp)_d", input_key)
        family = match.group(1) if match else (
            "solver_layout" if input_key.startswith("represent_") else "unknown"
        )
        winners.append({
            "variant": variant,
            "input_key": input_key,
            "family": family,
            "dimension": winner.get("max_dimension", ""),
            "structured_cones": winner.get("structured_cones", ""),
            "natural_strategy": winner.get("natural_strategy", ""),
            "winner": winner.get("strategy", ""),
            "winner_latency_us": winner_latency,
            "second_best_latency_us": second,
            "winner_margin": second / winner_latency if winner_latency > 0 else math.nan,
            "natural_latency_us": natural_latency,
            "speedup_over_natural": natural_latency / winner_latency
                if natural_latency > 0 and winner_latency > 0 else math.nan,
            "tested_strategies": ";".join(row.get("strategy", "") for row in candidates),
        })
    return winners


def ab_comparisons(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "OK" and is_true(row, "correct"):
            groups[row.get("case_id", "")].append(row)
    comparisons = []
    for case_id, candidates in sorted(groups.items()):
        by_variant = {row.get("variant", ""): row for row in candidates}
        baseline_name = next((name for name in by_variant if "baseline" in name), None)
        enhanced_name = next((name for name in by_variant
                              if name == "enhanced" or "enhanced" in name), None)
        if not baseline_name or not enhanced_name:
            continue
        baseline = by_variant[baseline_name]
        enhanced = by_variant[enhanced_name]
        baseline_time = as_float(baseline, "latency_median_us")
        enhanced_time = as_float(enhanced, "latency_median_us")
        comparisons.append({
            "case_id": case_id,
            "category": enhanced.get("category", ""),
            "strategy": enhanced.get("strategy", ""),
            "baseline_variant": baseline_name,
            "enhanced_variant": enhanced_name,
            "baseline_latency_us": baseline_time,
            "enhanced_latency_us": enhanced_time,
            "speedup": baseline_time / enhanced_time
                if baseline_time > 0 and enhanced_time > 0 else math.nan,
            "baseline_reductions": baseline.get("profile_reductions", ""),
            "enhanced_reductions": enhanced.get("profile_reductions", ""),
            "baseline_bisections": baseline.get("profile_bisections", ""),
            "enhanced_bisections": enhanced.get("profile_bisections", ""),
            "enhanced_newton_attempts": enhanced.get("profile_newton_attempts", ""),
            "enhanced_newton_accepts": enhanced.get("profile_newton_accepts", ""),
            "enhanced_warm_attempts": enhanced.get("profile_warm_attempts", ""),
            "enhanced_warm_accepts": enhanced.get("profile_warm_accepts", ""),
        })
    return comparisons


def fmt(value, digits=3):
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def make_report(paths, rows, winners, comparisons):
    lines = ["# Projection stress benchmark summary", ""]
    lines.append(f"Input CSV files: {len(paths)}; rows: {len(rows)}.")
    lines.append("")
    lines.extend(["## Correctness gate", ""])
    lines.append("| Variant | Rows | Gate pass | Reference-invalid diagnostics | Not applicable | Errors | Nonfinite | Max gate violation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row.get("variant", "unknown")].append(row)
    for variant, group in sorted(by_variant.items()):
        eligible = [row for row in group
                    if row.get("status") == "OK" and
                    is_true(row, "reference_valid")]
        correct = sum(is_true(row, "correct") for row in eligible)
        reference_invalid = sum(
            row.get("status") == "OK" and
            not is_true(row, "reference_valid") for row in group
        )
        not_applicable = sum(row.get("status") == "NOT_APPLICABLE"
                             for row in group)
        errors = sum(row.get("status") not in ("OK", "NOT_APPLICABLE")
                     for row in group)
        nonfinite = sum(not is_true(row, "output_finite") for row in group
                        if row.get("status") == "OK")
        violation = max((as_float(row, "max_normalized_cone_violation", 0.0)
                         for row in eligible), default=math.nan)
        lines.append(f"| {variant} | {len(group)} | {correct}/{len(eligible)} | "
                     f"{reference_invalid} | {not_applicable} | {errors} | {nonfinite} | "
                     f"{fmt(violation, 3)} |")
    lines.append("")

    if comparisons:
        lines.extend(["## Root-search A/B", ""])
        speedups = [as_float(row, "speedup") for row in comparisons]
        lines.append(f"Matched correct cases: {len(comparisons)}; geometric-mean speedup: "
                     f"{fmt(geometric_mean(speedups))}x; median: {fmt(median(speedups))}x.")
        lines.append("")
        lines.append("| Case | Strategy | Speedup | Bisections old→new | Reductions old→new |")
        lines.append("|---|---|---:|---:|---:|")
        for row in sorted(comparisons, key=lambda item: as_float(item, "speedup"), reverse=True)[:20]:
            lines.append(f"| {row['case_id']} | {row['strategy']} | {fmt(as_float(row, 'speedup'))}x | "
                         f"{row['baseline_bisections']}→{row['enhanced_bisections']} | "
                         f"{row['baseline_reductions']}→{row['enhanced_reductions']} |")
        lines.append("")

    lines.extend(["## Hierarchy winners", ""])
    winner_counts = Counter(row["winner"] for row in winners)
    lines.append("Winner counts: " + ", ".join(
        f"{name}={winner_counts.get(name, 0)}" for name in STRATEGY_ORDER))
    lines.append("")
    lines.append("| Input | Variant | Family | Dimension | Cones | Current | Winner | Speedup | Margin |")
    lines.append("|---|---|---|---:|---:|---|---|---:|---:|")
    for row in winners:
        lines.append(f"| {row['input_key']} | {row['variant']} | {row['family']} | {row['dimension']} | {row['structured_cones']} | "
                     f"{row['natural_strategy']} | {row['winner']} | "
                     f"{fmt(as_float(row, 'speedup_over_natural'))}x | "
                     f"{fmt(as_float(row, 'winner_margin'))}x |")
    lines.append("")
    lines.append("A selector change is accepted only after the winning rule also passes the "
                 "hard-case and 62-instance PDCS gates; microbenchmark wins alone are not final.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths, rows = load_rows(args.inputs)
    if not rows:
        raise SystemExit("no result rows found")
    os.makedirs(args.output_dir, exist_ok=True)
    winners = strategy_winners(rows)
    comparisons = ab_comparisons(rows)
    write_csv(os.path.join(args.output_dir, "merged.csv"), rows)
    write_csv(os.path.join(args.output_dir, "strategy_winners.csv"), winners)
    write_csv(os.path.join(args.output_dir, "ab_comparisons.csv"), comparisons)
    report = make_report(paths, rows, winners, comparisons)
    with open(os.path.join(args.output_dir, "summary.md"), "w") as stream:
        stream.write(report)
    print(os.path.join(args.output_dir, "summary.md"))


if __name__ == "__main__":
    main()
