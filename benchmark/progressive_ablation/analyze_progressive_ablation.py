#!/usr/bin/env python3
"""Write adjacent-stage comparisons for the cumulative PDHG ablation."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path


CONFIGS = (
    "pdhg",
    "pdhg_restart",
    "pdhg_restart_scaling",
    "pdhg_restart_scaling_reflection",
    "pdhg_restart_scaling_reflection_adaptive_primal_weight",
    "pdhg_restart_scaling_reflection_adaptive",
)

LABELS = {
    "pdhg": "Pure PDHG",
    "pdhg_restart": "+ Restart",
    "pdhg_restart_scaling": "+ Diagonal rescaling",
    "pdhg_restart_scaling_reflection": "+ Reflection",
    "pdhg_restart_scaling_reflection_adaptive_primal_weight": (
        "+ Adaptive primal weight"
    ),
    "pdhg_restart_scaling_reflection_adaptive": "+ Adaptive step",
}

FLAGS = {
    "pdhg": (False, False, False, False, False),
    "pdhg_restart": (True, False, False, False, False),
    "pdhg_restart_scaling": (True, True, False, False, False),
    "pdhg_restart_scaling_reflection": (True, True, False, False, True),
    "pdhg_restart_scaling_reflection_adaptive_primal_weight": (
        True,
        True,
        False,
        True,
        True,
    ),
    "pdhg_restart_scaling_reflection_adaptive": (
        True,
        True,
        True,
        True,
        True,
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def geometric_mean(values: list[float]) -> float:
    usable = [value for value in values if value > 0 and math.isfinite(value)]
    if not usable:
        return math.nan
    return math.exp(statistics.mean(math.log(value) for value in usable))


def bootstrap_ratio(
    ratios: list[float], samples: int, rng: random.Random
) -> tuple[float, float, float]:
    if not ratios:
        return math.nan, math.nan, math.nan
    logs = [math.log(value) for value in ratios]
    estimate = math.exp(statistics.mean(logs))
    draws = sorted(
        math.exp(statistics.mean(rng.choice(logs) for _ in logs))
        for _ in range(samples)
    )
    return (
        estimate,
        draws[int(0.025 * (len(draws) - 1))],
        draws[int(0.975 * (len(draws) - 1))],
    )


def adjacent_effects(
    raw_rows: list[dict[str, str]], bootstrap_samples: int, seed: int
) -> list[dict[str, object]]:
    indexed = {
        (row["instance_id"], row["configuration"]): row for row in raw_rows
    }
    instance_ids = sorted({row["instance_id"] for row in raw_rows})
    rng = random.Random(seed)
    output: list[dict[str, object]] = []
    for previous, added in zip(CONFIGS, CONFIGS[1:]):
        time_ratios: list[float] = []
        iteration_ratios: list[float] = []
        previous_only = 0
        added_only = 0
        both_failed = 0
        for instance_id in instance_ids:
            before = indexed.get((instance_id, previous))
            after = indexed.get((instance_id, added))
            before_ok = bool(before and before["verified_solved"] == "True")
            after_ok = bool(after and after["verified_solved"] == "True")
            if before_ok and after_ok:
                before_time = finite(before["wall_seconds"])
                after_time = finite(after["wall_seconds"])
                before_iterations = finite(before["iterations"])
                after_iterations = finite(after["iterations"])
                if before_time > 0 and after_time > 0:
                    time_ratios.append(after_time / before_time)
                if before_iterations > 0 and after_iterations > 0:
                    iteration_ratios.append(after_iterations / before_iterations)
            elif before_ok:
                previous_only += 1
            elif after_ok:
                added_only += 1
            else:
                both_failed += 1
        estimate, lower, upper = bootstrap_ratio(
            time_ratios, bootstrap_samples, rng
        )
        output.append(
            {
                "previous_configuration": previous,
                "added_configuration": added,
                "component_added": LABELS[added].removeprefix("+ "),
                "jointly_solved": len(time_ratios),
                "runtime_ratio_after_over_before": estimate,
                "runtime_ratio_ci95_lower": lower,
                "runtime_ratio_ci95_upper": upper,
                "iteration_ratio_after_over_before": geometric_mean(
                    iteration_ratios
                ),
                "previous_only_solved": previous_only,
                "added_only_solved": added_only,
                "both_failed_or_missing": both_failed,
            }
        )
    return output


def fmt(value: object, digits: int = 4) -> str:
    number = finite(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    manifest = read_csv(args.run_dir / "manifest.csv")
    raw_rows = read_csv(args.run_dir / "raw_results.csv")
    summary = read_csv(args.run_dir / "summary_overall.csv")
    effects = adjacent_effects(
        raw_rows, args.bootstrap_samples, args.bootstrap_seed
    )
    write_csv(args.run_dir / "adjacent_effects.csv", effects)

    summary_by_config = {row["configuration"]: row for row in summary}
    expected_records = len(manifest) * len(CONFIGS)
    complete = (
        len(raw_rows) == expected_records
        and {
            (row["instance_id"], row["configuration"]) for row in raw_rows
        }
        == {
            (row["instance_id"], configuration)
            for row in manifest
            for configuration in CONFIGS
        }
    )

    lines = [
        "# Progressive cuPDCS ablation results",
        "",
        f"Run directory: `{args.run_dir}`",
        "",
        f"Status: **{'COMPLETE' if complete else 'PARTIAL'}** "
        f"({len(raw_rows)}/{expected_records} formal records).",
        "",
        f"Settings: tolerance = `{args.tolerance:g}`; time limit = "
        f"`{args.time_limit:g}` seconds per instance/configuration.",
        "",
        "Halpern is disabled in all six stages. Components are added in the "
        "importance order suggested by the earlier one-at-a-time ablation. The "
        "adaptive-primal-weight stage enables dynamic omega while keeping eta "
        "fixed; the final adaptive-step stage then enables adaptive eta.",
        "",
        "## Cumulative configurations",
        "",
        "| Stage | Restart | Scaling | Adaptive step | Adaptive weight | Reflection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for configuration in CONFIGS:
        restart, scaling, adaptive, weight, reflection = FLAGS[configuration]
        yes_no = lambda value: "on" if value else "off"
        lines.append(
            f"| {LABELS[configuration]} | {yes_no(restart)} | "
            f"{yes_no(scaling)} | {yes_no(adaptive)} | {yes_no(weight)} | "
            f"{yes_no(reflection)} |"
        )

    lines.extend(
        [
            "",
            "## Overall results",
            "",
            "| Stage | Verified solved / 63 | Change | SGM(10) wall seconds | "
            "GM iterations | Timeouts |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    previous_solved: int | None = None
    for configuration in CONFIGS:
        row = summary_by_config[configuration]
        solved = int(row["verified_solved"])
        change = "—" if previous_solved is None else f"{solved - previous_solved:+d}"
        lines.append(
            f"| {LABELS[configuration]} | {solved}/63 | {change} | "
            f"{fmt(row['sgm10_wall_seconds'])} | "
            f"{fmt(row['gm_iterations_verified'])} | {row['timeouts']} |"
        )
        previous_solved = solved

    lines.extend(
        [
            "",
            "SGM(10) assigns the full time limit to every unsolved or missing "
            "record. A run is verified solved only if primal infeasibility, dual "
            "infeasibility, and relative gap are all at most the common tolerance.",
            "",
            "## Adjacent-stage effects",
            "",
            "Every ratio is `after adding the component / before adding it`; "
            "a ratio below one is faster.",
            "",
            "| Added component | Jointly solved | Runtime ratio | 95% bootstrap CI | "
            "Iteration ratio | Before-only | After-only |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['component_added']} | {row['jointly_solved']} | "
            f"{fmt(row['runtime_ratio_after_over_before'])} | "
            f"[{fmt(row['runtime_ratio_ci95_lower'])}, "
            f"{fmt(row['runtime_ratio_ci95_upper'])}] | "
            f"{fmt(row['iteration_ratio_after_over_before'])} | "
            f"{row['previous_only_solved']} | {row['added_only_solved']} |"
        )

    lines.extend(
        [
            "",
            "## Result files",
            "",
            f"- `{args.run_dir / 'raw_results.csv'}`",
            f"- `{args.run_dir / 'summary_overall.csv'}`",
            f"- `{args.run_dir / 'summary_by_size.csv'}`",
            f"- `{args.run_dir / 'summary_by_cone_mix.csv'}`",
            f"- `{args.run_dir / 'adjacent_effects.csv'}`",
            f"- `{args.run_dir / 'gpu_assignments.csv'}`",
            "",
        ]
    )
    if not complete:
        lines.extend(
            [
                "Do not use this partial report in the paper. Resume the run and "
                "regenerate the analysis first.",
                "",
            ]
        )

    report = args.report or args.run_dir / "progressive_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"PROGRESSIVE_ANALYSIS_COMPLETE records={len(raw_rows)} "
        f"expected={expected_records} report={report}"
    )


if __name__ == "__main__":
    main()
