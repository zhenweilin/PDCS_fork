#!/usr/bin/env python3
"""Aggregate cuPDCS ablation results without discarding failed cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


CONFIG_ORDER = (
    "full",
    "no_scaling",
    "no_adaptive_step",
    "no_adaptive_primal_weight",
    "no_restart",
    "no_reflection",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value, default=math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def latest_result(case_directory: Path) -> tuple[Path, dict[str, object]] | None:
    done = case_directory / "DONE"
    if not done.is_file():
        return None
    attempt_name = done.read_text(encoding="utf-8").strip()
    attempt = case_directory / attempt_name
    result_path = attempt / "result.json"
    if not result_path.is_file():
        return None
    with result_path.open(encoding="utf-8") as stream:
        return attempt, json.load(stream)


def collect(run_dir: Path, manifest_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    manifest = {row["instance_id"]: row for row in manifest_rows}
    rows: list[dict[str, object]] = []
    for instance_id in sorted(manifest):
        for configuration in CONFIG_ORDER:
            case_directory = run_dir / "cases" / instance_id / configuration
            loaded = latest_result(case_directory)
            if loaded is None:
                continue
            attempt, result = loaded
            metrics = result.get("metrics", {})
            verified = bool(result.get("verified_solved", False))
            run_status = str(result.get("run_status", "UNKNOWN"))
            if run_status != "COMPLETED":
                failure_class = "RUNTIME_ERROR"
            elif verified:
                failure_class = "SOLVED"
            else:
                termination = str(result.get("termination_status", "UNKNOWN"))
                failure_class = termination.replace("MathOptInterface.", "").upper()
            row = {
                "instance_id": instance_id,
                "filename": manifest[instance_id]["filename"],
                "configuration": configuration,
                "size_class": manifest[instance_id]["size_class"],
                "cone_mix": manifest[instance_id]["cone_mix"],
                "nnz_a": manifest[instance_id]["nnz_a"],
                "task_index": result.get("task_index", ""),
                "within_instance_order": result.get("within_instance_order", ""),
                "run_status": run_status,
                "termination_status": result.get("termination_status", ""),
                "failure_class": failure_class,
                "verified_solved": verified,
                "solver_seconds": finite_float(metrics.get("solve_time_sec")),
                "wall_seconds": finite_float(result.get("wall_seconds")),
                "iterations": finite_float(metrics.get("iterations")),
                "restart_count": finite_float(metrics.get("restart_count")),
                "restart_current_count": finite_float(
                    metrics.get("restart_current_count")
                ),
                "restart_mean_count": finite_float(
                    metrics.get("restart_mean_count")
                ),
                "restart_halpern_count": finite_float(
                    metrics.get("restart_halpern_count")
                ),
                "l_inf_rel_primal_res": finite_float(
                    metrics.get("l_inf_rel_primal_res")
                ),
                "l_inf_rel_dual_res": finite_float(
                    metrics.get("l_inf_rel_dual_res")
                ),
                "l_2_rel_primal_res": finite_float(
                    metrics.get("l_2_rel_primal_res")
                ),
                "l_2_rel_dual_res": finite_float(
                    metrics.get("l_2_rel_dual_res")
                ),
                "relative_gap": finite_float(metrics.get("relative_gap")),
                "objective_value": finite_float(metrics.get("objective_value")),
                "dual_objective_value": finite_float(
                    metrics.get("dual_objective_value")
                ),
                "verification_metric_max": finite_float(
                    result.get("verification_metric_max")
                ),
                "raw_log": str(attempt / "solver.raw.log"),
                "result_json": str(attempt / "result.json"),
            }
            rows.append(row)
    return rows


def geometric_mean(values: list[float]) -> float:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return math.nan
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def shifted_geometric_mean(values: list[float], shift: float) -> float:
    if not values:
        return math.nan
    return math.exp(sum(math.log(value + shift) for value in values) / len(values)) - shift


def median_and_max(rows, key) -> tuple[float, float]:
    values = [finite_float(row[key]) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan
    return statistics.median(values), max(values)


def summarize_group(
    rows: list[dict[str, object]],
    expected_instances: int,
    timeout_value: float,
    shift: float,
) -> list[dict[str, object]]:
    summaries = []
    for configuration in CONFIG_ORDER:
        config_rows = [row for row in rows if row["configuration"] == configuration]
        solved = [row for row in config_rows if row["verified_solved"]]
        penalized_times = []
        for row in config_rows:
            if row["verified_solved"]:
                value = finite_float(row["wall_seconds"])
                penalized_times.append(min(value, timeout_value))
            else:
                penalized_times.append(timeout_value)
        penalized_times.extend(
            [timeout_value] * max(0, expected_instances - len(config_rows))
        )
        primal_median, primal_max = median_and_max(config_rows, "l_inf_rel_primal_res")
        dual_median, dual_max = median_and_max(config_rows, "l_inf_rel_dual_res")
        gap_median, gap_max = median_and_max(config_rows, "relative_gap")
        summaries.append(
            {
                "configuration": configuration,
                "completed_records": len(config_rows),
                "verified_solved": len(solved),
                "expected_instances": expected_instances,
                "sgm10_wall_seconds": shifted_geometric_mean(penalized_times, shift),
                "gm_iterations_verified": geometric_mean(
                    [finite_float(row["iterations"]) for row in solved]
                ),
                "median_l_inf_primal": primal_median,
                "max_l_inf_primal": primal_max,
                "median_l_inf_dual": dual_median,
                "max_l_inf_dual": dual_max,
                "median_relative_gap": gap_median,
                "max_relative_gap": gap_max,
                "timeouts": sum(
                    str(row["termination_status"]).upper().endswith("TIME_LIMIT")
                    for row in config_rows
                ),
                "runtime_errors": sum(
                    row["failure_class"] == "RUNTIME_ERROR" for row in config_rows
                ),
                "mean_restart_count": (
                    statistics.mean(
                        finite_float(row["restart_count"]) for row in config_rows
                    )
                    if config_rows
                    else math.nan
                ),
                "total_restart_current": sum(
                    finite_float(row["restart_current_count"], 0.0)
                    for row in config_rows
                ),
                "total_restart_mean": sum(
                    finite_float(row["restart_mean_count"], 0.0)
                    for row in config_rows
                ),
                "total_restart_halpern": sum(
                    finite_float(row["restart_halpern_count"], 0.0)
                    for row in config_rows
                ),
            }
        )
    return summaries


def bootstrap_geomean_ratio(
    ratios: list[float], samples: int, rng: random.Random
) -> tuple[float, float, float]:
    if not ratios:
        return math.nan, math.nan, math.nan
    log_values = [math.log(value) for value in ratios]
    estimate = math.exp(statistics.mean(log_values))
    draws = []
    for _ in range(samples):
        sampled = [rng.choice(log_values) for _ in log_values]
        draws.append(math.exp(statistics.mean(sampled)))
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return estimate, lower, upper


def paired_effects(
    rows: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
    baseline_configuration: str | None = None,
) -> list[dict[str, object]]:
    baseline_configuration = baseline_configuration or CONFIG_ORDER[0]
    indexed = {
        (str(row["instance_id"]), str(row["configuration"])): row for row in rows
    }
    instance_ids = sorted({str(row["instance_id"]) for row in rows})
    rng = random.Random(seed)
    output = []
    for configuration in (
        item for item in CONFIG_ORDER if item != baseline_configuration
    ):
        runtime_ratios = []
        iteration_ratios = []
        full_only = 0
        ablation_only = 0
        both_failed = 0
        for instance_id in instance_ids:
            full = indexed.get((instance_id, baseline_configuration))
            ablated = indexed.get((instance_id, configuration))
            full_ok = bool(full and full["verified_solved"])
            ablated_ok = bool(ablated and ablated["verified_solved"])
            if full_ok and ablated_ok:
                full_time = finite_float(full["wall_seconds"])
                ablated_time = finite_float(ablated["wall_seconds"])
                full_iter = finite_float(full["iterations"])
                ablated_iter = finite_float(ablated["iterations"])
                if full_time > 0 and ablated_time > 0:
                    runtime_ratios.append(ablated_time / full_time)
                if full_iter > 0 and ablated_iter > 0:
                    iteration_ratios.append(ablated_iter / full_iter)
            elif full_ok:
                full_only += 1
            elif ablated_ok:
                ablation_only += 1
            else:
                both_failed += 1
        estimate, lower, upper = bootstrap_geomean_ratio(
            runtime_ratios, bootstrap_samples, rng
        )
        output.append(
            {
                "configuration": configuration,
                "jointly_solved": len(runtime_ratios),
                "runtime_ratio_geomean": estimate,
                "runtime_ratio_ci95_lower": lower,
                "runtime_ratio_ci95_upper": upper,
                "iteration_ratio_geomean": geometric_mean(iteration_ratios),
                "full_only_solved": full_only,
                "ablation_only_solved": ablation_only,
                "both_failed_or_missing": both_failed,
            }
        )
    return output


def format_number(value, digits=4) -> str:
    value = finite_float(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def write_report(
    path: Path,
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    paired: list[dict[str, object]],
    timeout_value: float,
    tolerance: float,
    baseline_configuration: str | None = None,
) -> None:
    baseline_configuration = baseline_configuration or CONFIG_ORDER[0]
    expected = len(manifest_rows) * len(CONFIG_ORDER)
    complete = len(rows) == expected
    lines = [
        "# cuPDCS ablation study results",
        "",
        f"Run directory: `{run_dir}`",
        "",
        f"Status: **{'COMPLETE' if complete else 'PARTIAL'}** "
        f"({len(rows)}/{expected} formal records).",
        "",
        f"Formal settings: time limit = {timeout_value:g} seconds per case; "
        f"tolerance = {tolerance:g}.",
        "",
        "Every case retains its complete `solver.raw.log`; `raw_results.csv` "
        "contains the corresponding path.",
        "",
        "## Overall results",
        "",
        "| Configuration | Verified solved / 63 | SGM(10) wall seconds | "
        "GM iterations | Median / max primal | Median / max dual | "
        "Median / max gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {configuration} | {solved}/63 | {sgm} | {iterations} | "
            "{pmed} / {pmax} | {dmed} / {dmax} | {gmed} / {gmax} |".format(
                configuration=item["configuration"],
                solved=item["verified_solved"],
                sgm=format_number(item["sgm10_wall_seconds"]),
                iterations=format_number(item["gm_iterations_verified"]),
                pmed=format_number(item["median_l_inf_primal"]),
                pmax=format_number(item["max_l_inf_primal"]),
                dmed=format_number(item["median_l_inf_dual"]),
                dmax=format_number(item["max_l_inf_dual"]),
                gmed=format_number(item["median_relative_gap"]),
                gmax=format_number(item["max_relative_gap"]),
            )
        )
    lines.extend(
        [
            "",
            "A run is counted as verified solved only when primal infeasibility, "
            "dual infeasibility, and relative gap are all at most the common "
            f"threshold `{tolerance:g}`. Failures and missing formal records are "
            f"assigned `{timeout_value:g}` seconds in SGM(10).",
            "",
            "## Restart candidate selections",
            "",
            "| Configuration | Current | Mean | Halpern |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in summary:
        lines.append(
            "| {configuration} | {current} | {mean} | {halpern} |".format(
                configuration=item["configuration"],
                current=format_number(item["total_restart_current"]),
                mean=format_number(item["total_restart_mean"]),
                halpern=format_number(item["total_restart_halpern"]),
            )
        )
    lines.extend(
        [
            "",
            f"## Paired effect relative to {baseline_configuration}",
            "",
            "| Ablation | Jointly solved | Runtime ratio | 95% bootstrap CI | "
            "Iteration ratio | Baseline-only solved | Alternative-only solved |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in paired:
        lines.append(
            "| {configuration} | {joint} | {ratio} | [{lower}, {upper}] | "
            "{iteration} | {full_only} | {ablation_only} |".format(
                configuration=item["configuration"],
                joint=item["jointly_solved"],
                ratio=format_number(item["runtime_ratio_geomean"]),
                lower=format_number(item["runtime_ratio_ci95_lower"]),
                upper=format_number(item["runtime_ratio_ci95_upper"]),
                iteration=format_number(item["iteration_ratio_geomean"]),
                full_only=item["full_only_solved"],
                ablation_only=item["ablation_only_solved"],
            )
        )
    lines.extend(["", "## Result files", ""])
    result_filenames = [
        "raw_results.csv",
        "summary_overall.csv",
        "summary_by_size.csv",
        "summary_by_cone_mix.csv",
        "paired_effects.csv",
        "manifest.csv",
        "source_map.csv",
        "run_order.csv",
    ]
    for filename in result_filenames:
        result_path = run_dir / filename
        if result_path.is_file():
            lines.append(f"- `{result_path}`")
    lines.append("")
    if not complete:
        lines.extend(
            [
                "Do not use a PARTIAL report for the paper. Resolve all missing "
                "or infrastructure-failure records first, then rerun this analyzer.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global CONFIG_ORDER
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--timeout-value", type=float, default=600.0)
    parser.add_argument("--sgm-shift", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument(
        "--configs",
        default=",".join(CONFIG_ORDER),
        help="Comma-separated configuration order; the first is the baseline.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    CONFIG_ORDER = tuple(
        item.strip() for item in args.configs.split(",") if item.strip()
    )
    if len(CONFIG_ORDER) < 2 or len(set(CONFIG_ORDER)) != len(CONFIG_ORDER):
        raise SystemExit("--configs must contain at least two unique names")

    manifest_rows = read_csv(args.run_dir / "manifest.csv")
    rows = collect(args.run_dir, manifest_rows)
    raw_fields = [
        "instance_id",
        "filename",
        "configuration",
        "size_class",
        "cone_mix",
        "nnz_a",
        "task_index",
        "within_instance_order",
        "run_status",
        "termination_status",
        "failure_class",
        "verified_solved",
        "solver_seconds",
        "wall_seconds",
        "iterations",
        "restart_count",
        "restart_current_count",
        "restart_mean_count",
        "restart_halpern_count",
        "l_inf_rel_primal_res",
        "l_inf_rel_dual_res",
        "l_2_rel_primal_res",
        "l_2_rel_dual_res",
        "relative_gap",
        "objective_value",
        "dual_objective_value",
        "verification_metric_max",
        "raw_log",
        "result_json",
    ]
    write_csv(args.run_dir / "raw_results.csv", rows, raw_fields)

    overall = summarize_group(
        rows, len(manifest_rows), args.timeout_value, args.sgm_shift
    )
    write_csv(args.run_dir / "summary_overall.csv", overall)

    size_rows = []
    for size_class in ("small", "medium", "large"):
        subset = [row for row in rows if row["size_class"] == size_class]
        expected = sum(row["size_class"] == size_class for row in manifest_rows)
        for item in summarize_group(
            subset, expected, args.timeout_value, args.sgm_shift
        ):
            size_rows.append({"size_class": size_class, **item})
    write_csv(args.run_dir / "summary_by_size.csv", size_rows)

    mix_rows = []
    cone_mixes = sorted({row["cone_mix"] for row in manifest_rows})
    for cone_mix in cone_mixes:
        subset = [row for row in rows if row["cone_mix"] == cone_mix]
        expected = sum(row["cone_mix"] == cone_mix for row in manifest_rows)
        for item in summarize_group(
            subset, expected, args.timeout_value, args.sgm_shift
        ):
            mix_rows.append({"cone_mix": cone_mix, **item})
    write_csv(args.run_dir / "summary_by_cone_mix.csv", mix_rows)

    paired = paired_effects(
        rows,
        args.bootstrap_samples,
        args.bootstrap_seed,
        CONFIG_ORDER[0],
    )
    write_csv(args.run_dir / "paired_effects.csv", paired)

    report = args.report or args.run_dir / "report.md"
    write_report(
        report,
        args.run_dir,
        manifest_rows,
        rows,
        overall,
        paired,
        args.timeout_value,
        args.tolerance,
        CONFIG_ORDER[0],
    )
    expected_records = len(manifest_rows) * len(CONFIG_ORDER)
    print(
        f"ANALYSIS_COMPLETE records={len(rows)} expected={expected_records} "
        f"report={report}"
    )


if __name__ == "__main__":
    main()
