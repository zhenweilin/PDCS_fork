#!/usr/bin/env python3

"""Aggregate the bounded Lasso timing-readiness suite.

This is deliberately a one-seed engineering check.  It must not be confused
with the ten-seed medium-scale publication experiment in
rebuttal_plan/ill_conditioned_lasso.md.
"""

import csv
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict


FIELDS = [
    "case",
    "seed",
    "K",
    "solver",
    "backend",
    "tolerance",
    "status",
    "termination_status",
    "input_seconds",
    "setup_seconds",
    "optimize_wall_seconds",
    "native_solve_seconds",
    "end_to_end_seconds",
    "iterations",
    "objective",
    "normalized_kkt",
    "x_error",
    "objective_error",
    "precision",
    "recall",
    "matrix_hash",
    "b_hash",
    "xstar_hash",
    "process_exit_status",
]


def pick(row, *names, default=""):
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            return value
    return default


def normalize(path):
    with path.open(newline="") as stream:
        source = next(csv.DictReader(stream))
    result_dir = path.parent
    case = result_dir.name
    exit_path = result_dir / "exit_status.txt"
    process_exit = exit_path.read_text().strip() if exit_path.exists() else ""
    solver = pick(source, "solver")
    termination = pick(source, "termination_status", "message")
    return {
        "case": case,
        "seed": pick(source, "seed"),
        "K": pick(source, "K"),
        "solver": solver,
        "backend": pick(source, "backend", default="cpu_or_pdcs"),
        "tolerance": pick(source, "tolerance"),
        "status": pick(source, "status", default="missing_result"),
        "termination_status": termination,
        "input_seconds": pick(source, "input_seconds", default="NaN"),
        "setup_seconds": pick(source, "setup_seconds", default="NaN"),
        "optimize_wall_seconds": pick(
            source, "optimize_wall_seconds", default="NaN"
        ),
        "native_solve_seconds": pick(
            source, "native_solve_seconds", default="NaN"
        ),
        "end_to_end_seconds": pick(
            source, "solve_seconds", "cold_seconds", default="NaN"
        ),
        "iterations": pick(source, "iterations"),
        "objective": pick(source, "objective", default="NaN"),
        "normalized_kkt": pick(source, "normalized_kkt", default="NaN"),
        "x_error": pick(source, "x_error", default="NaN"),
        "objective_error": pick(source, "objective_error", default="NaN"),
        "precision": pick(source, "precision", default="NaN"),
        "recall": pick(source, "recall", default="NaN"),
        "matrix_hash": pick(source, "matrix_hash"),
        "b_hash": pick(source, "b_hash"),
        "xstar_hash": pick(source, "xstar_hash"),
        "process_exit_status": process_exit,
    }


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def fmt(value, digits=5):
    parsed = number(value)
    return "NA" if parsed is None else f"{parsed:.{digits}g}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_lasso_readiness_comparison.py RUN_DIR")
    root = pathlib.Path(sys.argv[1]).resolve()
    environment = {}
    environment_path = root / "environment.txt"
    if environment_path.exists():
        for line in environment_path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                environment[key] = value
    expected_solvers = [
        value.strip()
        for value in environment.get(
            "solvers", "pdcs_gpu,scs_gpu,cuclarabel"
        ).split(",")
        if value.strip()
    ]
    expected_cases_text = environment.get("cases", "")
    expected_cases = (
        {
            value.strip()
            for value in expected_cases_text.split(",")
            if value.strip()
        }
        if expected_cases_text
        else {
            "tiny_K_1",
            "tiny_K_100",
            "tiny_K_10000",
            "tiny_K_1000000",
            "smoke_K_1",
        }
    )
    paths = []
    for solver in expected_solvers:
        solver_root = root / "results" / solver
        paths += sorted(solver_root.glob("*/seed_level_results.csv"))
        paths += sorted(solver_root.glob("*/cuclarabel_results.csv"))
        paths += sorted(solver_root.glob("*/scs_gpu_results.csv"))
    rows = [normalize(path) for path in paths]

    master = root / "master_results.csv"
    with master.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)
    hash_checks = {}
    for case, case_rows in by_case.items():
        tuples = {
            (row["matrix_hash"], row["b_hash"], row["xstar_hash"])
            for row in case_rows
            if row["matrix_hash"]
        }
        hash_checks[case] = (
            len(tuples) == 1 and len(case_rows) == len(expected_solvers)
        )
    present_pairs = {(row["solver"], row["case"]) for row in rows}
    missing_pairs = sorted(
        (solver, case)
        for solver in expected_solvers
        for case in expected_cases
        if (solver, case) not in present_pairs
    )
    duplicate_pairs = [
        pair
        for pair, count in Counter(
            (row["solver"], row["case"]) for row in rows
        ).items()
        if count != 1
    ]
    bad_exit_pairs = sorted(
        (row["solver"], row["case"], row["process_exit_status"])
        for row in rows
        if row["process_exit_status"] != "0"
    )
    seed_count = len({row["seed"] for row in rows if row["seed"]})
    suite = environment.get("suite", "readiness")

    report = root / "readiness_summary.md"
    with report.open("w") as out:
        out.write("# Ill-Conditioned Lasso GPU Experiment Summary\n\n")
        out.write(
            f"Suite: `{suite}`. Paired workload seeds found: "
            f"`{seed_count}`. This is not the ten-seed medium-scale "
            "experiment specified in the full plan.\n\n"
        )
        out.write(
            f"Expected `{len(expected_solvers) * len(expected_cases)}` rows; "
            f"found `{len(rows)}`. Missing solver/case pairs: "
            f"`{len(missing_pairs)}`; duplicate pairs: "
            f"`{len(duplicate_pairs)}`; nonzero/missing process exits: "
            f"`{len(bad_exit_pairs)}`.\n\n"
        )
        out.write("## Input identity\n\n")
        for case in sorted(by_case):
            status = "PASS" if hash_checks[case] else "FAIL"
            out.write(
                f"- `{case}`: solver-result hash consistency `{status}` "
                f"across {len(by_case[case])} result rows.\n"
            )
        out.write("\n## Per-case results\n\n")
        out.write(
            "| Case | Solver | Native status | Verifier status | Iterations | "
            "Native solve (s) | Optimize wall (s) | End-to-end (s) | "
            "KKT | x error | Objective |\n"
        )
        out.write(
            "|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|\n"
        )
        for row in sorted(rows, key=lambda r: (r["case"], r["solver"])):
            out.write(
                f"| {row['case']} | {row['solver']} | "
                f"{row['termination_status']} | {row['status']} | "
                f"{row['iterations'] or 'NA'} | "
                f"{fmt(row['native_solve_seconds'])} | "
                f"{fmt(row['optimize_wall_seconds'])} | "
                f"{fmt(row['end_to_end_seconds'])} | "
                f"{fmt(row['normalized_kkt'], 4)} | "
                f"{fmt(row['x_error'], 4)} | "
                f"{fmt(row['objective'], 8)} |\n"
            )
        counts = Counter(row["status"] for row in rows)
        out.write("\n## Status counts\n\n")
        for status, count in sorted(counts.items()):
            out.write(f"- `{status}`: {count}\n")
        out.write("\n## Aggregate by solver and K\n\n")
        out.write(
            "| Solver | K | Runs | Native optimal | Verified | "
            "Median native, all (s) | Median native, verified (s) | "
            "Median KKT | Median x error |\n"
        )
        out.write(
            "|:--|--:|--:|--:|--:|--:|--:|--:|--:|\n"
        )
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["solver"], row["K"])].append(row)
        for (solver, k_value), group_rows in sorted(
            grouped.items(),
            key=lambda item: (
                item[0][0],
                number(item[0][1])
                if number(item[0][1]) is not None
                else math.inf,
            ),
        ):
            all_times = [
                parsed
                for parsed in (
                    number(row["native_solve_seconds"]) for row in group_rows
                )
                if parsed is not None
            ]
            verified_rows = [
                row
                for row in group_rows
                if row["status"] == "solved_verified"
            ]
            verified_times = [
                parsed
                for parsed in (
                    number(row["native_solve_seconds"])
                    for row in verified_rows
                )
                if parsed is not None
            ]
            kkt_values = [
                parsed
                for parsed in (
                    number(row["normalized_kkt"]) for row in group_rows
                )
                if parsed is not None
            ]
            x_errors = [
                parsed
                for parsed in (
                    number(row["x_error"]) for row in group_rows
                )
                if parsed is not None
            ]
            native_optimal = sum(
                row["termination_status"] in ("OPTIMAL", "ALMOST_OPTIMAL")
                for row in group_rows
            )
            out.write(
                f"| {solver} | {fmt(k_value, 7)} | {len(group_rows)} | "
                f"{native_optimal} | {len(verified_rows)} | "
                f"{fmt(statistics.median(all_times) if all_times else None)} | "
                f"{fmt(statistics.median(verified_times) if verified_times else None)} | "
                f"{fmt(statistics.median(kkt_values) if kkt_values else None, 4)} | "
                f"{fmt(statistics.median(x_errors) if x_errors else None, 4)} |\n"
            )
        out.write(
            "\nSolver-native time is preferred for numerical comparisons. "
            "Optimize wall time includes wrapper and synchronization costs. "
            "End-to-end time also includes JuMP model construction, result "
            "extraction, and independent verification. Every formal solve "
            "was preceded by a same-process smaller SOCP warm-up.\n"
        )

    failed_hashes = [case for case, passed in hash_checks.items() if not passed]
    print(
        f"LASSO_READINESS_ANALYSIS_COMPLETE rows={len(rows)} "
        f"hash_failures={len(failed_hashes)} "
        f"missing_pairs={len(missing_pairs)} "
        f"duplicate_pairs={len(duplicate_pairs)} "
        f"bad_exit_pairs={len(bad_exit_pairs)} report={report}"
    )
    return (
        1
        if failed_hashes or missing_pairs or duplicate_pairs or bad_exit_pairs
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
