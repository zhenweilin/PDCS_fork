#!/usr/bin/env python3
"""Analyze AE-11 result TOMLs using only the Python standard library."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VALID_STATUSES = {
    "solved_verified",
    "solver_claimed_solved_but_inaccurate",
    "timeout",
    "out_of_memory",
    "numerical_failure",
    "input_or_conversion_failure",
    "license_failure",
    "not_attempted_hardware_limit",
}
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_SAMPLES = 20_000

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - exercised on the benchmark host
    tomllib = None


def load_toml_documents(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Parse all TOMLs once; use the existing Julia env on Python 3.10."""
    if tomllib is not None:
        documents = {}
        for path in paths:
            with path.open("rb") as stream:
                documents[str(path)] = tomllib.load(stream)
        return documents
    repo_root = Path(__file__).resolve().parents[2]
    sibling = repo_root.parent / "PDCS_fork"
    julia = sibling / ".julia-bin" / "julia-1.12.6" / "bin" / "julia"
    code = (
        "using TOML, JSON; "
        "print(JSON.json(Dict(path => TOML.parsefile(path) for path in ARGS); "
        "allownan=true))"
    )
    environment = os.environ.copy()
    environment.setdefault("JULIA_DEPOT_PATH", str(sibling / ".julia-depot"))
    environment.setdefault("JULIA_PKG_OFFLINE", "true")
    completed = subprocess.run(
        [str(julia), "--startup-file=no", f"--project={repo_root}",
         "-e", code, *(str(path) for path in paths)],
        check=True, capture_output=True, text=True, env=environment,
    )
    return json.loads(completed.stdout)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def geomean(values: Iterable[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    return math.exp(statistics.fmean(map(math.log, positive))) if positive else None


def shifted_geomean(values: Iterable[float], shift: float = 1.0) -> float | None:
    usable = [value for value in values if value >= 0 and math.isfinite(value)]
    if not usable:
        return None
    return math.exp(statistics.fmean(math.log(value + shift) for value in usable)) - shift


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_geomean_ci(values: list[float]) -> list[float] | None:
    logs = [math.log(value) for value in values if value > 0 and math.isfinite(value)]
    if not logs:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(
        math.exp(statistics.fmean(rng.choices(logs, k=len(logs))))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return [
        estimates[int(0.025 * BOOTSTRAP_SAMPLES)],
        estimates[int(0.975 * BOOTSTRAP_SAMPLES)],
    ]


def parameter_key(row: dict[str, Any]) -> tuple[str, float]:
    panel = str(row["panel"])
    value = finite(row["beta"] if panel == "A" else row["alpha"])
    if value is None:
        raise ValueError(f"missing penalty parameter in {row.get('_path')}")
    return panel, value


def solver_label(row: dict[str, Any]) -> str:
    solver = str(row["solver"])
    if solver == "cupdcs":
        return f"cupdcs_{'on' if row.get('rescaling_on') is True else 'off'}"
    return solver


def capped_time(row: dict[str, Any]) -> float | None:
    if row.get("_timing_excluded"):
        return None
    # Environment and hardware-availability records establish whether a run
    # could be attempted; they are not censored algorithm runtimes and must
    # not be charged the solver time limit in performance summaries.
    if row.get("status") in {
        "input_or_conversion_failure", "license_failure",
        "not_attempted_hardware_limit",
    }:
        return None
    limit = finite(row.get("time_limit_seconds"))
    elapsed = finite(row.get("solve_seconds"))
    if row.get("status") != "solved_verified" or elapsed is None:
        return limit
    return min(elapsed, limit) if limit is not None else elapsed


def apply_execution_incidents(
    rows: list[dict[str, Any]], config: dict[str, Any],
    incident_document: dict[str, Any] | None,
) -> dict[str, Any]:
    incidents = [] if incident_document is None else incident_document.get(
        "incidents", []
    )
    profile_dimensions = {
        profile: (int(config[profile]["m"]), int(config[profile]["n"]))
        for profile in ("pilot", "medium", "large")
    }
    present_dimensions = {(int(row["m"]), int(row["n"])) for row in rows}
    present_profiles = {
        profile for profile, dimensions in profile_dimensions.items()
        if dimensions in present_dimensions
    }
    pending = [
        item for item in incidents
        if item.get("status") != "resolved"
        and str(item.get("profile")) in present_profiles
    ]
    matched: dict[str, int] = Counter()
    for incident in pending:
        incident_id = str(incident["id"])
        for row in rows:
            if (
                str(row["solver_label"]) == str(incident["solver"])
                and int(row["seed"]) == int(incident["seed"])
                and float(row["target_kappa"]) == float(incident["target_kappa"])
                and str(row["panel"]) == str(incident["panel"])
                and float(row["tolerance"]) == float(incident["tolerance"])
            ):
                row["_timing_excluded"] = incident_id
                matched[incident_id] += 1
    return {
        "pending": len(pending),
        "pending_ids": [str(item["id"]) for item in pending],
        "matched_rows": dict(matched),
    }


def load_rows(
    result_paths: list[Path], documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        row = documents[str(path)]
        if "solver" not in row or "instance_id" not in row:
            continue
        row["_path"] = str(path)
        row["solver_label"] = solver_label(row)
        row["capped_solve_seconds"] = capped_time(row)
        rows.append(row)
    return rows


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_statuses = sorted({str(row.get("status")) for row in rows} - VALID_STATUSES)
    if invalid_statuses:
        raise SystemExit(f"invalid statuses: {invalid_statuses}")
    logical_keys = [
        (
            row["solver_label"], int(row["m"]), int(row["n"]),
            int(row["seed"]), float(row["target_kappa"]),
            *parameter_key(row), float(row["tolerance"]),
        )
        for row in rows
    ]
    duplicates = [key for key, count in Counter(logical_keys).items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate logical rows: {duplicates[:5]}")

    pattern_groups: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    value_groups: dict[tuple[int, int, int, float], set[str]] = defaultdict(set)
    b_groups: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for row in rows:
        design_key = (int(row["m"]), int(row["n"]), int(row["seed"]))
        value_key = (*design_key, float(row["target_kappa"]))
        pattern_groups[design_key].add(str(row["matrix_pattern_hash"]))
        value_groups[value_key].add(str(row["matrix_value_hash"]))
        b_groups[design_key].add(str(row["b_hash"]))
    bad_patterns = [key for key, values in pattern_groups.items() if len(values) != 1]
    bad_values = [key for key, values in value_groups.items() if len(values) != 1]
    bad_b = [key for key, values in b_groups.items() if len(values) != 1]
    if bad_patterns or bad_values or bad_b:
        raise SystemExit(
            "instance hash audit failed: "
            f"patterns={bad_patterns[:3]} values={bad_values[:3]} b={bad_b[:3]}"
        )
    precision_errors = [
        row["_path"] for row in rows if row.get("value_type") != "Float64"
    ]
    if precision_errors:
        raise SystemExit(f"non-Float64 rows: {precision_errors[:5]}")
    provenance_fields = (
        "config_sha256", "common_sha256", "runner_sha256",
        "generator_git_commit", "solver_environment_git_commit",
    )
    provenance = {
        field: sorted({str(row[field]) for row in rows if row.get(field)})
        for field in provenance_fields
    }
    cpu_thread_settings = Counter(
        (str(row["solver_label"]), int(row["julia_threads"]),
         int(row["blas_threads"]))
        for row in rows
        if row.get("julia_threads") is not None
        and row.get("blas_threads") is not None
    )
    cpu_affinity_settings = Counter(
        (str(row["solver_label"]), str(row["cpu_affinity_list"]))
        for row in rows if row.get("cpu_affinity_list") is not None
    )
    return {
        "rows": len(rows),
        "logical_keys_unique": True,
        "pattern_groups": len(pattern_groups),
        "value_groups": len(value_groups),
        "pattern_hash_consistent_across_kappa": True,
        "matrix_value_hash_consistent_across_solvers_and_penalties": True,
        "b_hash_consistent_across_kappa": True,
        "all_float64": True,
        "provenance_hashes": provenance,
        "config_hash_consistent": len(provenance["config_sha256"]) == 1,
        "common_hash_consistent": len(provenance["common_sha256"]) == 1,
        "cpu_thread_settings": {
            f"{solver}:julia={julia_threads}:blas={blas_threads}": count
            for (solver, julia_threads, blas_threads), count
            in sorted(cpu_thread_settings.items())
        },
        "cpu_affinity_settings": {
            f"{solver}:cpus={affinity}": count
            for (solver, affinity), count in sorted(cpu_affinity_settings.items())
        },
    }


def expected_grid_audit(
    rows: list[dict[str, Any]], config: dict[str, Any], allow_incomplete: bool,
) -> dict[str, Any]:
    """Audit the full configured grid for every profile present in the input."""
    dimensions = {
        (int(config[profile]["m"]), int(config[profile]["n"])): profile
        for profile in ("pilot", "medium", "large")
    }
    present_profiles = sorted({
        dimensions[(int(row["m"]), int(row["n"]))]
        for row in rows
        if (int(row["m"]), int(row["n"])) in dimensions
    })
    if not present_profiles:
        raise SystemExit("result dimensions do not match a configured profile")

    expected: set[tuple[Any, ...]] = set()
    expected_by_profile: Counter[str] = Counter()
    for profile in present_profiles:
        section = config[profile]
        m, n = int(section["m"]), int(section["n"])
        panels = section.get("panels", ["A", "B"])
        penalties = []
        if "A" in panels:
            penalties.append(("A", float(config["panel_a_beta"])))
        if "B" in panels:
            penalties.extend(
                ("B", float(alpha)) for alpha in config["panel_b_alphas"]
            )
        for solver in config["solvers"][profile]:
            for seed in section["seeds"]:
                for kappa in config["kappas"]:
                    for panel, parameter in penalties:
                        for tolerance in config["tolerances"]:
                            expected.add((
                                str(solver), m, n, int(seed), float(kappa),
                                panel, parameter, float(tolerance),
                            ))
                            expected_by_profile[profile] += 1

    actual = {
        (
            str(row["solver_label"]), int(row["m"]), int(row["n"]),
            int(row["seed"]), float(row["target_kappa"]),
            *parameter_key(row), float(row["tolerance"]),
        )
        for row in rows
    }
    missing = sorted(expected - actual, key=str)
    configured_actual = actual & expected
    missing_by_profile: Counter[str] = Counter()
    missing_by_solver: Counter[str] = Counter()
    for solver, m, n, *_ in missing:
        missing_by_profile[dimensions[(m, n)]] += 1
        missing_by_solver[solver] += 1
    result = {
        "profiles": present_profiles,
        "expected_records": len(expected),
        "configured_records_present": len(configured_actual),
        "missing_records": len(missing),
        "extra_records": len(actual - expected),
        "complete": not missing,
        "expected_by_profile": dict(expected_by_profile),
        "missing_by_profile": dict(missing_by_profile),
        "missing_by_solver": dict(missing_by_solver),
        "first_missing_keys": [list(key) for key in missing[:20]],
    }
    if missing and not allow_incomplete:
        raise SystemExit(
            "configured grid is incomplete: "
            f"{len(configured_actual)}/{len(expected)} records present; "
            f"missing by solver={dict(missing_by_solver)}"
        )
    return result


def add_reference_errors(
    rows: list[dict[str, Any]], reference_tolerance: float,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(
            int(row["m"]), int(row["n"]), int(row["seed"]),
            float(row["target_kappa"]), *parameter_key(row),
        )].append(row)
    priority = {"mosek": 0, "clarabel_cpu": 1, "pdcs_cpu": 2,
                "scs_indirect": 3}
    reference_count = 0
    reference_solvers: Counter[str] = Counter()
    for candidates in groups.values():
        verified = [
            row for row in candidates
            if row.get("status") == "solved_verified"
            and finite(row.get("objective_value")) is not None
            and str(row["solver_label"]) in priority
            and (finite(row.get("tolerance")) or math.inf)
            <= reference_tolerance
        ]
        if not verified:
            continue
        reference = min(
            verified,
            key=lambda row: (
                finite(row.get("tolerance")) or math.inf,
                priority.get(str(row["solver_label"]), 99),
                finite(row.get("independent_kkt")) or math.inf,
            ),
        )
        reference_objective = float(reference["objective_value"])
        reference_count += 1
        reference_solvers[str(reference["solver_label"])] += 1
        for row in candidates:
            objective = finite(row.get("objective_value"))
            row["reference_objective"] = reference_objective
            row["reference_solver"] = reference["solver_label"]
            row["relative_objective_error"] = (
                abs(objective - reference_objective) /
                (1 + abs(reference_objective))
                if objective is not None else None
            )
    return {
        "required_tolerance": reference_tolerance,
        "reference_groups": reference_count,
        "reference_solvers": dict(reference_solvers),
    }


def grouped_summary(rows: list[dict[str, Any]], shift: float) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(
            int(row["m"]), int(row["n"]), row["solver_label"],
            *parameter_key(row), float(row["target_kappa"]),
            float(row["tolerance"]),
        )].append(row)
    output = []
    for key, items in sorted(groups.items(), key=str):
        m, n, solver, panel, parameter, kappa, tolerance = key
        verified = [item for item in items if item["status"] == "solved_verified"]
        times = [value for item in items if (value := capped_time(item)) is not None]
        iterations = [
            value for item in items
            if (value := finite(item.get("iterations"))) is not None and value >= 0
        ]
        kkt = [
            value for item in items
            if (value := finite(item.get("independent_kkt"))) is not None
            and value >= 0
        ]
        memory = [
            value for item in items
            for field in ("peak_cpu_memory", "peak_gpu_memory")
            if (value := finite(item.get(field))) is not None and value >= 0
        ]
        output.append({
            "m": m, "n": n, "solver": solver, "panel": panel,
            "parameter": parameter, "target_kappa": kappa,
            "tolerance": tolerance, "verified": len(verified),
            "total": len(items), "sgm_time_seconds": shifted_geomean(times, shift),
            "gm_iterations": geomean(iterations),
            "median_independent_kkt": statistics.median(kkt) if kkt else None,
            "peak_memory_bytes": max(memory) if memory else None,
            "status_counts": dict(Counter(str(item["status"]) for item in items)),
        })
    return output


def rescaling_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cupdcs = [row for row in rows if row["solver"] == "cupdcs"]
    groups: dict[tuple[Any, ...], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in cupdcs:
        groups[(
            int(row["m"]), int(row["n"]), int(row["seed"]),
            float(row["target_kappa"]), *parameter_key(row),
            float(row["tolerance"]),
        )][bool(row["rescaling_on"])] = row
    output = []
    for key, pair in sorted(groups.items(), key=str):
        if set(pair) != {False, True}:
            continue
        off, on = pair[False], pair[True]
        off_time, on_time = capped_time(off), capped_time(on)
        off_iter, on_iter = finite(off.get("iterations")), finite(on.get("iterations"))
        off_iter = off_iter if off_iter is not None and off_iter >= 0 else None
        on_iter = on_iter if on_iter is not None and on_iter >= 0 else None
        output.append({
            "m": key[0], "n": key[1], "seed": key[2],
            "target_kappa": key[3], "panel": key[4], "parameter": key[5],
            "tolerance": key[6], "off_status": off["status"],
            "on_status": on["status"],
            "time_on_over_off": (
                on_time / off_time if on_time and off_time and off_time > 0 else None
            ),
            "iteration_on_over_off": (
                on_iter / off_iter
                if on_iter is not None and off_iter not in (None, 0) else None
            ),
            "kkt_on": finite(on.get("independent_kkt")),
            "kkt_off": finite(off.get("independent_kkt")),
        })
    return output


def conditioning_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[(
            int(row["m"]), int(row["n"]), row["solver_label"],
            int(row["seed"]), *parameter_key(row), float(row["tolerance"]),
        )][float(row["target_kappa"])] = row
    output = []
    for key, by_kappa in groups.items():
        baseline = by_kappa.get(1.0)
        if baseline is None:
            continue
        base_time = capped_time(baseline)
        base_iter = finite(baseline.get("iterations"))
        base_iter = base_iter if base_iter is not None and base_iter >= 0 else None
        for kappa, row in by_kappa.items():
            run_time = capped_time(row)
            run_iter = finite(row.get("iterations"))
            run_iter = run_iter if run_iter is not None and run_iter >= 0 else None
            output.append({
                "m": key[0], "n": key[1], "solver": key[2],
                "seed": key[3], "panel": key[4], "parameter": key[5],
                "tolerance": key[6], "target_kappa": kappa,
                "time_ratio_to_k1": (
                    (run_time + 1) / (base_time + 1)
                    if run_time is not None and base_time is not None else None
                ),
                "iteration_ratio_to_k1": (
                    (run_iter + 1) / (base_iter + 1)
                    if run_iter is not None and base_iter is not None else None
                ),
            })
    return output


def conditioning_ratio_summary(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[(
            item["m"], item["n"], item["solver"], item["panel"],
            item["parameter"], item["tolerance"], item["target_kappa"],
        )].append(item)
    output = []
    for key, group in sorted(groups.items(), key=str):
        time_values = [
            value for item in group
            if (value := finite(item["time_ratio_to_k1"])) is not None
        ]
        iteration_values = [
            value for item in group
            if (value := finite(item["iteration_ratio_to_k1"])) is not None
        ]
        output.append({
            "m": key[0], "n": key[1], "solver": key[2],
            "panel": key[3], "parameter": key[4], "tolerance": key[5],
            "target_kappa": key[6], "seeds": len(group),
            "time_ratio_geomean": geomean(time_values),
            "time_ratio_bootstrap_95pct_ci": bootstrap_geomean_ci(time_values),
            "iteration_ratio_geomean": geomean(iteration_values),
            "iteration_ratio_bootstrap_95pct_ci": bootstrap_geomean_ci(iteration_values),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row if key != "_path"})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row[field], sort_keys=True)
                if isinstance(row.get(field), (dict, list)) else row.get(field)
                for field in fields
            })


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "--" if number is None else f"{number:.{digits}g}"


def svg_plot(
    path: Path, rows: list[dict[str, Any]], metric: str, title: str,
    panel: str = "A",
) -> None:
    selected = [row for row in rows if row.get("panel") == panel]
    if not selected:
        return
    groups: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        if metric == "success_rate":
            value = 1.0 if row.get("status") == "solved_verified" else 0.0
        elif metric == "solve_seconds":
            value = capped_time(row)
        elif metric in {"iterations", "independent_kkt"}:
            # Include every finite returned iterate, including solver-claimed
            # solutions rejected by the common verifier.  Restricting these
            # plots to verified rows would hide conditioning-driven accuracy
            # failures through survivorship bias.
            value = finite(row.get(metric))
        else:
            value = finite(row.get(metric))
            if row.get("status") != "solved_verified":
                continue
        if value is not None and value >= 0:
            groups[str(row["solver_label"])][float(row["target_kappa"])].append(value)
    if not groups:
        return
    width, height = 960, 600
    left, right, top, bottom = 85, 30, 55, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    kappas = sorted({kappa for group in groups.values() for kappa in group})
    x_min, x_max = min(map(math.log10, kappas)), max(map(math.log10, kappas))
    log_y = metric in {"solve_seconds", "iterations", "independent_kkt"}
    raw_values = [value for group in groups.values() for values in group.values() for value in values]
    if log_y:
        transformed = [math.log10(max(value, 1e-18)) for value in raw_values]
    else:
        transformed = raw_values
    y_min, y_max = min(transformed), max(transformed)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    y_pad = 0.06 * (y_max - y_min)
    y_min, y_max = y_min - y_pad, y_max + y_pad
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]

    def sx(kappa: float) -> float:
        value = math.log10(kappa)
        return left + (value - x_min) / max(x_max - x_min, 1) * plot_w

    def sy(value: float) -> float:
        transformed_value = math.log10(max(value, 1e-18)) if log_y else value
        return top + (y_max - transformed_value) / (y_max - y_min) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>',
    ]
    for kappa in kappas:
        x = sx(kappa)
        elements.append(f'<line x1="{x}" y1="{top+plot_h}" x2="{x}" y2="{top+plot_h+5}" stroke="black"/>')
        elements.append(f'<text x="{x}" y="{top+plot_h+25}" text-anchor="middle" font-family="sans-serif" font-size="12">10^{int(round(math.log10(kappa)))}</text>')
    for tick in range(6):
        coordinate = y_min + tick * (y_max - y_min) / 5
        y = top + (y_max - coordinate) / (y_max - y_min) * plot_h
        label = f"1e{coordinate:.1f}" if log_y else f"{coordinate:.2f}"
        elements.append(f'<line x1="{left-5}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#dddddd"/>')
        elements.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{label}</text>')
    for group_index, (label, by_kappa) in enumerate(sorted(groups.items())):
        color = colors[group_index % len(colors)]
        medians = []
        for kappa, values in sorted(by_kappa.items()):
            x = sx(kappa)
            median = statistics.median(values)
            low, high = quantile(values, 0.25), quantile(values, 0.75)
            medians.append((x, sy(median)))
            elements.append(f'<line x1="{x}" y1="{sy(low)}" x2="{x}" y2="{sy(high)}" stroke="{color}" stroke-width="3"/>')
            for point_index, value in enumerate(values):
                jitter = ((point_index % 5) - 2) * 2.2
                elements.append(f'<circle cx="{x+jitter}" cy="{sy(value)}" r="2.6" fill="{color}" fill-opacity="0.55"/>')
        if len(medians) > 1:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in medians)
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = top + 18 * group_index
        elements.append(f'<line x1="{left+plot_w-170}" y1="{legend_y}" x2="{left+plot_w-150}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{left+plot_w-145}" y="{legend_y+4}" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
    elements.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">log10 nonzero spectral condition number</text>')
    elements.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def markdown_report(
    path: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
    audit: dict[str, Any], completeness: dict[str, Any],
    references: dict[str, Any],
) -> None:
    status_counts = Counter(str(row["status"]) for row in rows)
    verified = status_counts["solved_verified"]
    lines = [
        "# AE-11 experiment results",
        "",
        "This file is generated by `analyze_results.py`. It distinguishes completed evidence from configured but unexecuted work.",
        "",
        "## Completeness and correctness audit",
        "",
        f"- Result records: {len(rows)}",
        f"- Configured-grid records present: {completeness['configured_records_present']}/{completeness['expected_records']}",
        f"- Configured grid complete: {'yes' if completeness['complete'] else 'no'}",
        f"- Missing records by solver: `{json.dumps(completeness['missing_by_solver'], sort_keys=True)}`",
        f"- Extra diagnostic/reference records: {completeness['extra_records']}",
        f"- Pending execution incidents: {completeness['pending_execution_incidents']} `{json.dumps(completeness['pending_execution_incident_ids'])}`",
        f"- Independently verified solves: {verified}/{len(rows)}",
        f"- Status taxonomy: `{json.dumps(dict(status_counts), sort_keys=True)}`",
        f"- Pattern/value/response hash audit: {'passed' if audit else 'not run'}",
        f"- Float64 audit: {'passed' if audit.get('all_float64') else 'failed'}",
        f"- Config/common hash consistency: {'passed' if audit.get('config_hash_consistent') and audit.get('common_hash_consistent') else 'failed'}",
        f"- Recorded CPU thread settings: `{json.dumps(audit.get('cpu_thread_settings', {}), sort_keys=True)}`",
        f"- Recorded CPU affinity settings: `{json.dumps(audit.get('cpu_affinity_settings', {}), sort_keys=True)}`",
        f"- High-accuracy CPU reference objectives available (tol. <= {references['required_tolerance']:.0e}): {references['reference_groups']} groups, selected from `{json.dumps(references['reference_solvers'], sort_keys=True)}`",
        "",
        "## Main Panel-A table",
        "",
        "| m | n | K | Solver | Tol. | Verified/Total | SGM time | GM iter. | Median KKT | Peak memory |",
        "|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        if item["panel"] != "A":
            continue
        lines.append(
            f"| {item['m']} | {item['n']} | {item['target_kappa']:.0e} | "
            f"{item['solver']} | {item['tolerance']:.0e} | "
            f"{item['verified']}/{item['total']} | {fmt(item['sgm_time_seconds'])} | "
            f"{fmt(item['gm_iterations'])} | {fmt(item['median_independent_kkt'])} | "
            f"{fmt(item['peak_memory_bytes'])} |"
        )
    lines.extend([
        "",
        "Iteration and KKT summaries include every finite returned iterate, including solver-claimed solutions rejected by the common verifier; availability failures have no numerical iterate and are excluded. This prevents verified-only survivorship bias.",
        "",
        "## Interpretation boundary",
        "",
        "Rows marked `input_or_conversion_failure` or `license_failure` describe environment availability, not an algorithmic failure. `K=1e8` is an extreme Float64 stress test and is reported separately from conclusions for `K<=1e6`. Unsuccessful runs remain in capped-time summaries and are not silently dropped.",
        "",
        "## Artifacts",
        "",
        "- `master_results.csv`: every run and independent-verifier field",
        "- `main_table.csv`: grouped main/appendix table",
        "- `rescaling_ablation.csv`: paired cuPDCS on/off rows",
        "- `conditioning_ratios.json`: K/K=1 paired ratios and bootstrap intervals",
        "- `plots/`: seed scatter plus median/IQR SVG figures",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result_paths = sorted(args.results_dir.rglob("*.toml"))
    incidents_path = args.config.parent / "execution_incidents.toml"
    input_paths = [args.config, *result_paths]
    if incidents_path.is_file():
        input_paths.append(incidents_path)
    documents = load_toml_documents(input_paths)
    config = documents[str(args.config)]
    rows = load_rows(result_paths, documents)
    if not rows:
        raise SystemExit("no AE-11 result TOMLs found")
    incidents = apply_execution_incidents(
        rows, config, documents.get(str(incidents_path)),
    )
    for row in rows:
        row["capped_solve_seconds"] = capped_time(row)
    audit = audit_rows(rows)
    completeness = expected_grid_audit(rows, config, args.allow_incomplete)
    completeness["pending_execution_incidents"] = incidents["pending"]
    completeness["pending_execution_incident_ids"] = incidents["pending_ids"]
    completeness["execution_incident_matched_rows"] = incidents["matched_rows"]
    completeness["complete"] = completeness["complete"] and incidents["pending"] == 0
    if incidents["pending"] and not args.allow_incomplete:
        raise SystemExit(
            "pending execution incidents require replacement runs: "
            f"{incidents['pending_ids']}"
        )
    references = add_reference_errors(
        rows, float(config["execution"]["reference_tolerance"]),
    )
    shift = float(config["statistics"]["runtime_shift_seconds"])
    summaries = grouped_summary(rows, shift)
    rescaling = rescaling_pairs(rows)
    ratios = conditioning_ratios(rows)
    ratio_summary = conditioning_ratio_summary(ratios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "master_results.csv", rows)
    write_csv(args.output_dir / "main_table.csv", summaries)
    write_csv(args.output_dir / "rescaling_ablation.csv", rescaling)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "audit": audit,
            "completeness": completeness,
            "references": references,
            "status_counts": dict(Counter(str(row["status"]) for row in rows)),
            "grouped_results": summaries,
            "rescaling_pairs": rescaling,
        }, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (args.output_dir / "conditioning_ratios.json").open("w", encoding="utf-8") as stream:
        json.dump({"seed_rows": ratios, "summary": ratio_summary}, stream,
                  indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    markdown_report(
        args.output_dir / "results_report.md", rows, summaries, audit,
        completeness, references,
    )
    for metric, title in (
        ("solve_seconds", "Panel A: capped solve time"),
        ("iterations", "Panel A: verified iteration count"),
        ("independent_kkt", "Panel A: independent KKT residual"),
        ("success_rate", "Panel A: verified success indicator"),
    ):
        svg_plot(args.output_dir / "plots" / f"panel_a_{metric}.svg",
                 rows, metric, title, "A")
    svg_plot(args.output_dir / "plots" / "panel_b_solve_seconds.svg",
             rows, "solve_seconds", "Panel B: capped solve time", "B")
    print(
        "AE11_ANALYSIS",
        f"rows={len(rows)} verified={sum(row['status'] == 'solved_verified' for row in rows)}",
        f"complete={completeness['complete']}",
        f"missing={completeness['missing_records']}",
        f"output={args.output_dir}",
    )


if __name__ == "__main__":
    main()
