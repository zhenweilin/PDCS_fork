#!/usr/bin/env python3
"""Aggregate the 62-instance R3.5 scalar-cone rescaling experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


MODES = ("diagonal", "scalar_cone")
STRUCTURED_CONES = {"Q", "QR", "EXP", "EXP*"}
PAIR_FIELDS = (
    "iterations",
    "preprocessing_time_sec",
    "pdhg_solve_time_sec",
    "solver_end_to_end_sec",
    "optimize_wall_sec",
    "projection_time_sec",
    "projection_time_per_iteration_sec",
    "projection_fraction",
)
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_SAMPLES = 20_000
SIZE_FIELDS = (
    "total_dimension",
    "nnz_a",
    "structured_cone_dimension",
    "max_structured_cone_dimension",
)


def clean_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        lines = []
        for raw_line in stream:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                lines.append(line)
        return lines


def section_index(lines: list[str], name: str) -> int | None:
    try:
        return lines.index(name)
    except ValueError:
        return None


def cone_blocks(lines: list[str], section: str) -> list[tuple[str, int]]:
    index = section_index(lines, section)
    if index is None:
        return []
    _, block_count = (int(value) for value in lines[index + 1].split())
    blocks = []
    for offset in range(block_count):
        name, dimension = lines[index + 2 + offset].split()
        blocks.append((name, int(dimension)))
    return blocks


def count_section(lines: list[str], section: str) -> int:
    index = section_index(lines, section)
    return 0 if index is None else int(lines[index + 1].split()[0])


def inspect_instance(path: Path) -> dict[str, object]:
    lines = clean_lines(path)
    var_index = section_index(lines, "VAR")
    con_index = section_index(lines, "CON")
    num_variables = int(lines[var_index + 1].split()[0]) if var_index is not None else 0
    num_constraints = int(lines[con_index + 1].split()[0]) if con_index is not None else 0
    blocks = cone_blocks(lines, "VAR") + cone_blocks(lines, "CON")
    structured_dimensions = [
        dimension for name, dimension in blocks if name in STRUCTURED_CONES
    ]
    counts = Counter(name for name, _ in blocks)
    dimensions = Counter()
    for name, dimension in blocks:
        dimensions[name] += dimension
    has_soc = counts["Q"] > 0 or counts["QR"] > 0
    has_exp = counts["EXP"] > 0 or counts["EXP*"] > 0
    if has_soc and has_exp:
        cone_mix = "soc_exp"
    elif has_soc:
        cone_mix = "soc"
    elif has_exp:
        cone_mix = "exp"
    else:
        cone_mix = "linear_only"
    return {
        "instance_id": path.name.removesuffix(".cbf.gz"),
        "filename": path.name,
        "num_variables": num_variables,
        "num_constraints": num_constraints,
        "total_dimension": num_variables + num_constraints,
        "nnz_a": count_section(lines, "ACOORD"),
        "cone_mix": cone_mix,
        "soc_blocks": counts["Q"],
        "rsoc_blocks": counts["QR"],
        "exp_blocks": counts["EXP"],
        "dual_exp_blocks": counts["EXP*"],
        "soc_dimension": dimensions["Q"],
        "rsoc_dimension": dimensions["QR"],
        "exp_dimension": dimensions["EXP"],
        "dual_exp_dimension": dimensions["EXP*"],
        "structured_cone_dimension": sum(
            dimensions[name] for name in STRUCTURED_CONES
        ),
        "max_structured_cone_dimension": max(structured_dimensions, default=0),
    }


def assign_size_terciles(
    metadata: dict[str, dict[str, object]],
) -> dict[str, dict[str, int]]:
    """Attach reproducible small/medium/large labels from the formal corpus."""
    cutoffs = {}
    for field in SIZE_FIELDS:
        values = sorted(int(item[field]) for item in metadata.values())
        lower = values[(len(values) - 1) // 3]
        upper = values[2 * (len(values) - 1) // 3]
        cutoffs[field] = {"small_max": lower, "medium_max": upper}
        label_field = f"{field}_tercile"
        for item in metadata.values():
            value = int(item[field])
            item[label_field] = (
                "small" if value <= lower else "medium" if value <= upper else "large"
            )
    return cutoffs


def finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    return math.exp(sum(math.log(value) for value in positive) / len(positive)) if positive else None


def shifted_geomean(values: list[float], shift: float = 10.0) -> float | None:
    nonnegative = [value for value in values if value >= 0 and math.isfinite(value)]
    if not nonnegative:
        return None
    return math.exp(statistics.fmean(math.log(value + shift) for value in nonnegative)) - shift


def bootstrap_geomean_ci(values: list[float]) -> list[float] | None:
    """Return a deterministic paired-bootstrap 95% CI for a geometric mean."""
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    log_values = [math.log(value) for value in positive]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(
        math.exp(statistics.fmean(rng.choices(log_values, k=len(log_values))))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return [
        estimates[int(0.025 * BOOTSTRAP_SAMPLES)],
        estimates[int(0.975 * BOOTSTRAP_SAMPLES)],
    ]


def log_ratio_trend(points: list[tuple[float, float]]) -> dict[str, float] | None:
    """Fit log(ratio) = intercept + slope*log(size) and report correlation."""
    positive = [(size, ratio) for size, ratio in points if size > 0 and ratio > 0]
    if len(positive) < 2:
        return None
    x_values = [math.log(size) for size, _ in positive]
    y_values = [math.log(ratio) for _, ratio in positive]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    xx = sum((value - x_mean) ** 2 for value in x_values)
    yy = sum((value - y_mean) ** 2 for value in y_values)
    xy = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    return {
        "instances": len(positive),
        "log_log_slope": xy / xx if xx > 0 else 0.0,
        "log_log_correlation": xy / math.sqrt(xx * yy) if xx > 0 and yy > 0 else 0.0,
    }


def fmt(value: float | None, digits: int = 6) -> str:
    return "NA" if value is None else f"{value:.{digits}g}"


def result_row(result: dict[str, object], metadata: dict[str, object]) -> dict[str, object]:
    metrics = result.get("metrics") or {}
    row = dict(metadata)
    row.update(
        {
            "configuration": result["configuration"],
            "repetition": int(result.get("repetition", 1)),
            "order_position": result.get("order_position"),
            "time_limit_sec": result.get("time_limit_sec"),
            "run_status": result.get("run_status"),
            "termination_status": result.get("termination_status"),
            "verified_solved": bool(result.get("verified_solved", False)),
            "verification_metric_max": result.get("verification_metric_max"),
            "iterations": metrics.get("iterations"),
            "preprocessing_time_sec": metrics.get("preprocessing_time_sec"),
            "pdhg_solve_time_sec": metrics.get("solve_time_sec"),
            "solver_end_to_end_sec": result.get("solver_end_to_end_sec"),
            "optimize_wall_sec": result.get("optimize_wall_sec"),
            "projection_time_sec": metrics.get("projection_time_sec"),
            "primal_projection_time_sec": metrics.get("primal_projection_time_sec"),
            "dual_slack_projection_time_sec": metrics.get("dual_slack_projection_time_sec"),
            "projection_fraction": result.get("projection_fraction"),
            "projection_time_per_iteration_sec": result.get(
                "projection_time_per_iteration_sec"
            ),
            "l_inf_rel_primal_res": metrics.get("l_inf_rel_primal_res"),
            "l_inf_rel_dual_res": metrics.get("l_inf_rel_dual_res"),
            "relative_gap": metrics.get("relative_gap"),
            "objective_value": metrics.get("objective_value"),
            "dual_objective_value": metrics.get("dual_objective_value"),
            "exit_status": metrics.get("exit_status"),
            "cuda_device": result.get("cuda_device"),
            "cuda_visible_devices": result.get("cuda_visible_devices"),
        }
    )
    return row


def capped_optimize_wall(row: dict[str, object]) -> float | None:
    """Wall time for all-instance analysis, charging failures the time limit."""
    wall = finite(row["optimize_wall_sec"])
    time_limit = finite(row["time_limit_sec"])
    if not row["verified_solved"]:
        return time_limit
    if wall is None:
        return time_limit
    return min(wall, time_limit) if time_limit is not None else wall


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_instance: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_instance[str(row["instance_id"])][str(row["configuration"])] = row

    complete_instances = sorted(
        instance_id
        for instance_id, mode_rows in by_instance.items()
        if set(mode_rows) == set(MODES)
    )
    common_verified = [
        instance_id
        for instance_id in complete_instances
        if all(by_instance[instance_id][mode]["verified_solved"] for mode in MODES)
    ]

    mode_summary: dict[str, object] = {}
    for mode in MODES:
        mode_rows = [row for row in rows if row["configuration"] == mode]
        verified_rows = [row for row in mode_rows if row["verified_solved"]]
        common_rows = [by_instance[instance_id][mode] for instance_id in common_verified]
        all_capped_wall = [
            value
            for row in mode_rows
            if (value := capped_optimize_wall(row)) is not None
        ]
        mode_summary[mode] = {
            "runs": len(mode_rows),
            "verified": len(verified_rows),
            "exit_status_counts": dict(Counter(str(row["exit_status"]) for row in mode_rows)),
            "sgm10_capped_optimize_wall_all_sec": shifted_geomean(all_capped_wall),
            "geomean_capped_optimize_wall_all_sec": geomean(all_capped_wall),
            "common_verified_runs": len(common_rows),
            "median_iterations": median(
                [float(row["iterations"]) for row in common_rows if finite(row["iterations"]) is not None]
            ),
            "geomean_iterations": geomean(
                [float(row["iterations"]) for row in common_rows if finite(row["iterations"]) is not None]
            ),
            "median_preprocessing_sec": median(
                [value for row in common_rows if (value := finite(row["preprocessing_time_sec"])) is not None]
            ),
            "geomean_preprocessing_sec": geomean(
                [value for row in common_rows if (value := finite(row["preprocessing_time_sec"])) is not None]
            ),
            "median_pdhg_solve_sec": median(
                [value for row in common_rows if (value := finite(row["pdhg_solve_time_sec"])) is not None]
            ),
            "geomean_pdhg_solve_sec": geomean(
                [value for row in common_rows if (value := finite(row["pdhg_solve_time_sec"])) is not None]
            ),
            "median_solver_end_to_end_sec": median(
                [value for row in common_rows if (value := finite(row["solver_end_to_end_sec"])) is not None]
            ),
            "geomean_solver_end_to_end_sec": geomean(
                [value for row in common_rows if (value := finite(row["solver_end_to_end_sec"])) is not None]
            ),
            "geomean_optimize_wall_sec": geomean(
                [value for row in common_rows if (value := finite(row["optimize_wall_sec"])) is not None]
            ),
            "median_projection_time_sec": median(
                [value for row in common_rows if (value := finite(row["projection_time_sec"])) is not None]
            ),
            "median_projection_fraction": median(
                [value for row in common_rows if (value := finite(row["projection_fraction"])) is not None]
            ),
            "geomean_projection_fraction": geomean(
                [value for row in common_rows if (value := finite(row["projection_fraction"])) is not None]
            ),
            "geomean_projection_time_per_iteration_sec": geomean(
                [
                    value
                    for row in common_rows
                    if (value := finite(row["projection_time_per_iteration_sec"])) is not None
                ]
            ),
            "median_kkt_max": median(
                [value for row in common_rows if (value := finite(row["verification_metric_max"])) is not None]
            ),
            "maximum_kkt_max": max(
                value
                for row in common_rows
                if (value := finite(row["verification_metric_max"])) is not None
            ),
        }

    pairwise: dict[str, object] = {}
    for numerator, denominator in (("scalar_cone", "diagonal"),):
        ratios: dict[str, list[float]] = defaultdict(list)
        wins = Counter()
        for instance_id in common_verified:
            num = by_instance[instance_id][numerator]
            den = by_instance[instance_id][denominator]
            for field in PAIR_FIELDS:
                num_value = finite(num[field])
                den_value = finite(den[field])
                if num_value is not None and den_value is not None and den_value > 0:
                    ratios[field].append(num_value / den_value)
            num_time = finite(num["solver_end_to_end_sec"])
            den_time = finite(den["solver_end_to_end_sec"])
            if num_time is not None and den_time is not None:
                if num_time < den_time:
                    wins["numerator"] += 1
                elif num_time > den_time:
                    wins["denominator"] += 1
                else:
                    wins["tie"] += 1
        key = f"{numerator}_over_{denominator}"
        pairwise[key] = {
            "instances": len(common_verified),
            "geomean_ratios": {field: geomean(values) for field, values in ratios.items()},
            "median_ratios": {field: median(values) for field, values in ratios.items()},
            "geomean_ratio_bootstrap_95pct_ci": {
                field: bootstrap_geomean_ci(values) for field, values in ratios.items()
            },
            "end_to_end_wins": dict(wins),
        }

    all_instance_ratios = []
    all_instance_wins = Counter()
    for instance_id in complete_instances:
        scalar_wall = capped_optimize_wall(by_instance[instance_id]["scalar_cone"])
        diagonal_wall = capped_optimize_wall(by_instance[instance_id]["diagonal"])
        if scalar_wall is None or diagonal_wall is None or diagonal_wall <= 0:
            continue
        ratio = scalar_wall / diagonal_wall
        all_instance_ratios.append(ratio)
        if scalar_wall < diagonal_wall:
            all_instance_wins["scalar_cone"] += 1
        elif scalar_wall > diagonal_wall:
            all_instance_wins["diagonal"] += 1
        else:
            all_instance_wins["tie"] += 1
    all_complete_pairwise = {
        "instances": len(all_instance_ratios),
        "capped_optimize_wall_geomean_ratio": geomean(all_instance_ratios),
        "capped_optimize_wall_median_ratio": median(all_instance_ratios),
        "capped_optimize_wall_geomean_ratio_bootstrap_95pct_ci":
            bootstrap_geomean_ci(all_instance_ratios),
        "wins": dict(all_instance_wins),
    }

    cone_mix_breakdown: dict[str, object] = {}
    for cone_mix in sorted({str(row["cone_mix"]) for row in rows}):
        instance_ids = [
            instance_id
            for instance_id in common_verified
            if by_instance[instance_id]["diagonal"]["cone_mix"] == cone_mix
        ]
        ratios: dict[str, list[float]] = defaultdict(list)
        for instance_id in instance_ids:
            numerator = by_instance[instance_id]["scalar_cone"]
            denominator = by_instance[instance_id]["diagonal"]
            for field in PAIR_FIELDS:
                num_value = finite(numerator[field])
                den_value = finite(denominator[field])
                if num_value is not None and den_value is not None and den_value > 0:
                    ratios[field].append(num_value / den_value)
        cone_mix_breakdown[cone_mix] = {
            "instances": len(instance_ids),
            "geomean_ratios": {
                field: geomean(values) for field, values in ratios.items()
            },
            "median_ratios": {
                field: median(values) for field, values in ratios.items()
            },
        }

    size_breakdown: dict[str, object] = {}
    size_trend: dict[str, object] = {}
    for size_field in SIZE_FIELDS:
        label_field = f"{size_field}_tercile"
        groups: dict[str, object] = {}
        for group in ("small", "medium", "large"):
            instance_ids = [
                instance_id
                for instance_id in complete_instances
                if by_instance[instance_id]["diagonal"][label_field] == group
            ]
            common_ids = [
                instance_id for instance_id in instance_ids if instance_id in common_verified
            ]
            paired_ratios: dict[str, list[float]] = defaultdict(list)
            for instance_id in common_ids:
                numerator = by_instance[instance_id]["scalar_cone"]
                denominator = by_instance[instance_id]["diagonal"]
                for field in PAIR_FIELDS:
                    num_value = finite(numerator[field])
                    den_value = finite(denominator[field])
                    if num_value is not None and den_value is not None and den_value > 0:
                        paired_ratios[field].append(num_value / den_value)

            mode_results = {}
            for mode in MODES:
                mode_rows = [by_instance[instance_id][mode] for instance_id in instance_ids]
                capped_walls = [
                    value
                    for row in mode_rows
                    if (value := capped_optimize_wall(row)) is not None
                ]
                mode_results[mode] = {
                    "verified": sum(bool(row["verified_solved"]) for row in mode_rows),
                    "runs": len(mode_rows),
                    "sgm10_capped_optimize_wall_sec": shifted_geomean(capped_walls),
                }

            capped_ratios = []
            for instance_id in instance_ids:
                scalar_wall = capped_optimize_wall(by_instance[instance_id]["scalar_cone"])
                diagonal_wall = capped_optimize_wall(by_instance[instance_id]["diagonal"])
                if scalar_wall is not None and diagonal_wall is not None and diagonal_wall > 0:
                    capped_ratios.append(scalar_wall / diagonal_wall)
            size_values = [
                int(by_instance[instance_id]["diagonal"][size_field])
                for instance_id in instance_ids
            ]
            groups[group] = {
                "formal_instances": len(instance_ids),
                "common_verified_instances": len(common_ids),
                "minimum": min(size_values) if size_values else None,
                "maximum": max(size_values) if size_values else None,
                "mode_summary": mode_results,
                "common_verified_geomean_ratios": {
                    field: geomean(values) for field, values in paired_ratios.items()
                },
                "all_instance_capped_optimize_wall_geomean_ratio":
                    geomean(capped_ratios),
            }
        size_breakdown[size_field] = groups

        common_wall_points = []
        common_projection_points = []
        for instance_id in common_verified:
            size = float(by_instance[instance_id]["diagonal"][size_field])
            scalar = by_instance[instance_id]["scalar_cone"]
            diagonal = by_instance[instance_id]["diagonal"]
            scalar_wall = finite(scalar["optimize_wall_sec"])
            diagonal_wall = finite(diagonal["optimize_wall_sec"])
            scalar_projection = finite(scalar["projection_time_per_iteration_sec"])
            diagonal_projection = finite(diagonal["projection_time_per_iteration_sec"])
            if scalar_wall is not None and diagonal_wall is not None and diagonal_wall > 0:
                common_wall_points.append((size, scalar_wall / diagonal_wall))
            if (
                scalar_projection is not None
                and diagonal_projection is not None
                and diagonal_projection > 0
            ):
                common_projection_points.append(
                    (size, scalar_projection / diagonal_projection)
                )
        all_capped_points = []
        for instance_id in complete_instances:
            size = float(by_instance[instance_id]["diagonal"][size_field])
            scalar_wall = capped_optimize_wall(by_instance[instance_id]["scalar_cone"])
            diagonal_wall = capped_optimize_wall(by_instance[instance_id]["diagonal"])
            if scalar_wall is not None and diagonal_wall is not None and diagonal_wall > 0:
                all_capped_points.append((size, scalar_wall / diagonal_wall))
        size_trend[size_field] = {
            "common_verified_optimize_wall": log_ratio_trend(common_wall_points),
            "common_verified_projection_time_per_iteration":
                log_ratio_trend(common_projection_points),
            "all_instance_capped_optimize_wall": log_ratio_trend(all_capped_points),
        }

    objective_differences = []
    for instance_id in common_verified:
        scalar_objective = finite(by_instance[instance_id]["scalar_cone"]["objective_value"])
        diagonal_objective = finite(by_instance[instance_id]["diagonal"]["objective_value"])
        if scalar_objective is not None and diagonal_objective is not None:
            scale = max(1.0, abs(scalar_objective), abs(diagonal_objective))
            objective_differences.append(abs(scalar_objective - diagonal_objective) / scale)

    cone_mix_common = Counter(
        by_instance[instance_id]["diagonal"]["cone_mix"] for instance_id in common_verified
    )
    cone_mix_formal = Counter(
        mode_rows["diagonal"]["cone_mix"]
        for mode_rows in by_instance.values()
        if "diagonal" in mode_rows
    )
    verified_outcome_mismatches = [
        instance_id
        for instance_id in complete_instances
        if by_instance[instance_id]["diagonal"]["verified_solved"]
        != by_instance[instance_id]["scalar_cone"]["verified_solved"]
    ]
    return {
        "formal_instances": len(by_instance),
        "complete_instances": len(complete_instances),
        "common_verified_instances": len(common_verified),
        "formal_cone_mix": dict(cone_mix_formal),
        "common_verified_cone_mix": dict(cone_mix_common),
        "verified_outcome_mismatches": verified_outcome_mismatches,
        "objective_relative_difference": {
            "median": median(objective_differences),
            "maximum": max(objective_differences) if objective_differences else None,
        },
        "mode_summary": mode_summary,
        "pairwise": pairwise,
        "all_complete_pairwise": all_complete_pairwise,
        "cone_mix_breakdown": cone_mix_breakdown,
        "size_breakdown": size_breakdown,
        "size_trend": size_trend,
    }


def summarize_all_repetitions(rows: list[dict[str, object]]) -> dict[str, object]:
    by_instance: dict[str, dict[int, dict[str, dict[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        by_instance[str(row["instance_id"])][int(row["repetition"])][
            str(row["configuration"])
        ] = row
    repetitions = sorted({int(row["repetition"]) for row in rows})
    complete_instances = sorted(
        instance_id
        for instance_id, repetition_rows in by_instance.items()
        if all(set(repetition_rows.get(rep, {})) == set(MODES) for rep in repetitions)
    )

    mode_summary = {}
    for mode in MODES:
        mode_rows = [row for row in rows if row["configuration"] == mode]
        capped_walls = [
            value
            for row in mode_rows
            if (value := capped_optimize_wall(row)) is not None
        ]
        verified_by_instance = {
            instance_id: [
                bool(by_instance[instance_id][rep][mode]["verified_solved"])
                for rep in repetitions
            ]
            for instance_id in complete_instances
        }
        per_instance_capped_geomeans = []
        for instance_id in complete_instances:
            values = [
                capped_optimize_wall(by_instance[instance_id][rep][mode])
                for rep in repetitions
            ]
            finite_values = [value for value in values if value is not None]
            if len(finite_values) == len(repetitions):
                per_instance_capped_geomeans.append(geomean(finite_values))
        mode_summary[mode] = {
            "runs": len(mode_rows),
            "verified_runs": sum(bool(row["verified_solved"]) for row in mode_rows),
            "instances_verified_in_all_repetitions": sum(
                all(values) for values in verified_by_instance.values()
            ),
            "instances_verified_in_any_repetition": sum(
                any(values) for values in verified_by_instance.values()
            ),
            "run_level_sgm10_capped_optimize_wall_sec": shifted_geomean(capped_walls),
            "per_instance_two_repetition_sgm10_capped_optimize_wall_sec":
                shifted_geomean(per_instance_capped_geomeans),
        }

    paired_verified_ratios: dict[str, list[float]] = defaultdict(list)
    paired_verified_runs = 0
    by_repetition = {}
    for rep in repetitions:
        rep_complete = [
            instance_id
            for instance_id in complete_instances
            if set(by_instance[instance_id][rep]) == set(MODES)
        ]
        common_verified = [
            instance_id
            for instance_id in rep_complete
            if all(
                by_instance[instance_id][rep][mode]["verified_solved"]
                for mode in MODES
            )
        ]
        rep_ratios: dict[str, list[float]] = defaultdict(list)
        capped_ratios = []
        for instance_id in common_verified:
            scalar = by_instance[instance_id][rep]["scalar_cone"]
            diagonal = by_instance[instance_id][rep]["diagonal"]
            for field in PAIR_FIELDS:
                scalar_value = finite(scalar[field])
                diagonal_value = finite(diagonal[field])
                if (
                    scalar_value is not None
                    and diagonal_value is not None
                    and diagonal_value > 0
                ):
                    ratio = scalar_value / diagonal_value
                    rep_ratios[field].append(ratio)
                    paired_verified_ratios[field].append(ratio)
            paired_verified_runs += 1
        for instance_id in rep_complete:
            scalar_wall = capped_optimize_wall(
                by_instance[instance_id][rep]["scalar_cone"]
            )
            diagonal_wall = capped_optimize_wall(
                by_instance[instance_id][rep]["diagonal"]
            )
            if scalar_wall is not None and diagonal_wall is not None and diagonal_wall > 0:
                capped_ratios.append(scalar_wall / diagonal_wall)
        by_repetition[str(rep)] = {
            "complete_instances": len(rep_complete),
            "common_verified_instances": len(common_verified),
            "verified_by_mode": {
                mode: sum(
                    bool(by_instance[instance_id][rep][mode]["verified_solved"])
                    for instance_id in rep_complete
                )
                for mode in MODES
            },
            "verified_outcome_mismatches": [
                instance_id
                for instance_id in rep_complete
                if bool(
                    by_instance[instance_id][rep]["diagonal"]["verified_solved"]
                )
                != bool(
                    by_instance[instance_id][rep]["scalar_cone"][
                        "verified_solved"
                    ]
                )
            ],
            "common_verified_geomean_ratios": {
                field: geomean(values) for field, values in rep_ratios.items()
            },
            "all_instance_capped_optimize_wall_geomean_ratio": geomean(capped_ratios),
        }

    per_instance_ratios = {}
    for instance_id in complete_instances:
        mode_scores = {}
        for mode in MODES:
            values = [
                capped_optimize_wall(by_instance[instance_id][rep][mode])
                for rep in repetitions
            ]
            finite_values = [value for value in values if value is not None]
            if len(finite_values) == len(repetitions):
                mode_scores[mode] = geomean(finite_values)
        if set(mode_scores) == set(MODES) and mode_scores["diagonal"] > 0:
            per_instance_ratios[instance_id] = (
                mode_scores["scalar_cone"] / mode_scores["diagonal"]
            )

    wins = Counter()
    for ratio in per_instance_ratios.values():
        if ratio < 1:
            wins["scalar_cone"] += 1
        elif ratio > 1:
            wins["diagonal"] += 1
        else:
            wins["tie"] += 1

    def verified_pair_metrics(instance_ids: set[str]) -> dict[str, object]:
        ratios: dict[str, list[float]] = defaultdict(list)
        paired_runs = 0
        for instance_id in sorted(instance_ids):
            for rep in repetitions:
                scalar = by_instance[instance_id][rep]["scalar_cone"]
                diagonal = by_instance[instance_id][rep]["diagonal"]
                if not (
                    scalar["verified_solved"] and diagonal["verified_solved"]
                ):
                    continue
                paired_runs += 1
                for field in PAIR_FIELDS:
                    scalar_value = finite(scalar[field])
                    diagonal_value = finite(diagonal[field])
                    if (
                        scalar_value is not None
                        and diagonal_value is not None
                        and diagonal_value > 0
                    ):
                        ratios[field].append(scalar_value / diagonal_value)
        return {
            "paired_verified_runs": paired_runs,
            "paired_verified_geomean_ratios": {
                field: geomean(values) for field, values in ratios.items()
            },
        }

    cone_mix_breakdown = {}
    cone_mixes = sorted({
        str(by_instance[instance_id][repetitions[0]]["diagonal"]["cone_mix"])
        for instance_id in per_instance_ratios
    })
    for cone_mix in cone_mixes:
        group_ratios = [
            ratio
            for instance_id, ratio in per_instance_ratios.items()
            if by_instance[instance_id][repetitions[0]]["diagonal"]["cone_mix"]
            == cone_mix
        ]
        group_wins = Counter()
        for ratio in group_ratios:
            if ratio < 1:
                group_wins["scalar_cone"] += 1
            elif ratio > 1:
                group_wins["diagonal"] += 1
            else:
                group_wins["tie"] += 1
        cone_mix_breakdown[cone_mix] = {
            "instances": len(group_ratios),
            "two_repetition_capped_optimize_wall_geomean_ratio":
                geomean(group_ratios),
            "two_repetition_capped_optimize_wall_median_ratio":
                median(group_ratios),
            "two_repetition_capped_optimize_wall_geomean_ratio_bootstrap_95pct_ci":
                bootstrap_geomean_ci(group_ratios),
            "wins": dict(group_wins),
            **verified_pair_metrics({
                instance_id
                for instance_id in per_instance_ratios
                if by_instance[instance_id][repetitions[0]]["diagonal"][
                    "cone_mix"
                ]
                == cone_mix
            }),
        }

    size_breakdown = {}
    size_trend = {}
    for size_field in SIZE_FIELDS:
        label_field = f"{size_field}_tercile"
        groups = {}
        for group in ("small", "medium", "large"):
            group_instance_ratios = {
                instance_id: ratio
                for instance_id, ratio in per_instance_ratios.items()
                if by_instance[instance_id][repetitions[0]]["diagonal"][label_field]
                == group
            }
            group_ratios = list(group_instance_ratios.values())
            group_wins = Counter()
            for ratio in group_ratios:
                if ratio < 1:
                    group_wins["scalar_cone"] += 1
                elif ratio > 1:
                    group_wins["diagonal"] += 1
                else:
                    group_wins["tie"] += 1
            groups[group] = {
                "instances": len(group_ratios),
                "two_repetition_capped_optimize_wall_geomean_ratio":
                    geomean(group_ratios),
                "two_repetition_capped_optimize_wall_median_ratio":
                    median(group_ratios),
                "two_repetition_capped_optimize_wall_geomean_ratio_bootstrap_95pct_ci":
                    bootstrap_geomean_ci(group_ratios),
                "wins": dict(group_wins),
                **verified_pair_metrics(set(group_instance_ratios)),
            }
        size_breakdown[size_field] = groups
        trend_points = [
            (
                float(
                    by_instance[instance_id][repetitions[0]]["diagonal"][size_field]
                ),
                ratio,
            )
            for instance_id, ratio in per_instance_ratios.items()
        ]
        size_trend[size_field] = log_ratio_trend(trend_points)

    ratio_values = list(per_instance_ratios.values())
    stable_common_verified = [
        instance_id
        for instance_id in complete_instances
        if all(
            by_instance[instance_id][rep][mode]["verified_solved"]
            for rep in repetitions
            for mode in MODES
        )
    ]
    return {
        "repetitions": repetitions,
        "complete_instances": len(complete_instances),
        "mode_summary": mode_summary,
        "by_repetition": by_repetition,
        "paired_verified_runs": paired_verified_runs,
        "paired_verified_geomean_ratios": {
            field: geomean(values) for field, values in paired_verified_ratios.items()
        },
        "stable_common_verified_instances": len(stable_common_verified),
        "all_instance_two_repetition_capped_optimize_wall": {
            "instances": len(ratio_values),
            "geomean_ratio": geomean(ratio_values),
            "median_ratio": median(ratio_values),
            "geomean_ratio_bootstrap_95pct_ci": bootstrap_geomean_ci(ratio_values),
            "wins": dict(wins),
        },
        "cone_mix_breakdown": cone_mix_breakdown,
        "size_breakdown": size_breakdown,
        "size_trend": size_trend,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, object]) -> None:
    print(
        "COUNTS",
        f"formal={summary['formal_instances']}",
        f"complete={summary['complete_instances']}",
        f"common_verified={summary['common_verified_instances']}",
    )
    for mode in MODES:
        item = summary["mode_summary"][mode]
        print(
            "MODE",
            mode,
            f"verified={item['verified']}/{item['runs']}",
            f"sgm10_all_sec={fmt(item['sgm10_capped_optimize_wall_all_sec'])}",
            f"median_iterations={fmt(item['median_iterations'])}",
            f"geomean_end_to_end_sec={fmt(item['geomean_solver_end_to_end_sec'])}",
            f"median_projection_fraction={fmt(item['median_projection_fraction'])}",
            f"median_kkt={fmt(item['median_kkt_max'])}",
        )
    for name, item in summary["pairwise"].items():
        ratios = item["geomean_ratios"]
        print(
            "PAIR",
            name,
            f"iterations={fmt(ratios.get('iterations'))}",
            f"pdhg={fmt(ratios.get('pdhg_solve_time_sec'))}",
            f"end_to_end={fmt(ratios.get('solver_end_to_end_sec'))}",
            f"projection_per_iter={fmt(ratios.get('projection_time_per_iteration_sec'))}",
            f"wins={item['end_to_end_wins']}",
        )
    all_repetitions = summary.get("all_repetitions_summary")
    if all_repetitions is not None:
        primary = all_repetitions[
            "all_instance_two_repetition_capped_optimize_wall"
        ]
        print(
            "ALL_REPETITIONS",
            f"complete={all_repetitions['complete_instances']}",
            f"capped_wall_ratio={fmt(primary['geomean_ratio'])}",
            f"ci={primary['geomean_ratio_bootstrap_95pct_ci']}",
            f"wins={primary['wins']}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--expected-repetitions", type=int, default=2)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    args.expected_repetitions > 0 or parser.error(
        "--expected-repetitions must be positive"
    )

    formal_paths = sorted(args.input_dir.glob("*.cbf.gz"))
    if len(formal_paths) != 62:
        raise SystemExit(f"expected 62 formal CBF files, found {len(formal_paths)}")
    metadata = {item["instance_id"]: item for item in map(inspect_instance, formal_paths)}
    size_cutoffs = assign_size_terciles(metadata)

    all_rows = []
    for path in sorted((args.run_dir / "cases").glob("*/*.json")):
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        instance_id = str(result["instance_id"])
        all_rows.append(result_row(result, metadata[instance_id]))
    if not all_rows:
        raise SystemExit("no result JSON files found")
    logical_keys = [
        (
            str(row["instance_id"]),
            int(row["repetition"]),
            str(row["configuration"]),
        )
        for row in all_rows
    ]
    duplicate_keys = sorted(
        key for key, count in Counter(logical_keys).items() if count > 1
    )
    if duplicate_keys:
        raise SystemExit(f"duplicate logical result records: {duplicate_keys}")
    failed_keys = [
        key
        for key, row in zip(logical_keys, all_rows)
        if row["run_status"] != "COMPLETED"
    ]
    if failed_keys:
        raise SystemExit(f"non-completed result records: {failed_keys}")
    if not args.allow_incomplete:
        expected_keys = {
            (instance_id, repetition, mode)
            for instance_id in metadata
            for repetition in range(1, args.expected_repetitions + 1)
            for mode in MODES
        }
        observed_keys = set(logical_keys)
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        if missing or unexpected:
            raise SystemExit(
                "formal result set is incomplete or unexpected: "
                f"observed={len(observed_keys)} expected={len(expected_keys)} "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
    latest_repetition: dict[tuple[str, str], dict[str, object]] = {}
    for row in all_rows:
        key = (str(row["instance_id"]), str(row["configuration"]))
        previous = latest_repetition.get(key)
        if previous is None or int(row["repetition"]) > int(previous["repetition"]):
            latest_repetition[key] = row
    rows = list(latest_repetition.values())
    rows.sort(key=lambda row: (str(row["instance_id"]), MODES.index(str(row["configuration"]))))
    all_rows.sort(key=lambda row: (
        str(row["instance_id"]),
        int(row["repetition"]),
        MODES.index(str(row["configuration"])),
    ))
    summary = summarize(rows)
    summary["size_stratification_cutoffs"] = size_cutoffs
    summary["all_repetitions_summary"] = summarize_all_repetitions(all_rows)

    csv_path = args.csv or args.run_dir / "results.csv"
    all_csv_path = args.run_dir / "all_repetitions.csv"
    summary_path = args.summary_json or args.run_dir / "summary.json"
    write_csv(csv_path, rows)
    write_csv(all_csv_path, all_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print_summary(summary)


if __name__ == "__main__":
    main()
