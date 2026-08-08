#!/usr/bin/env python3
"""Validate and summarize disk-free Table 5 Lasso runs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

try:
    import tomllib

    TOMLDecodeError = tomllib.TOMLDecodeError

    def load_toml(path: Path) -> dict[str, Any]:
        with path.open("rb") as stream:
            return tomllib.load(stream)

except ModuleNotFoundError:
    # Ubuntu 22.04 ships Python 3.10. The repository environment already
    # provides the compatible `toml` package, so no install is needed.
    import toml

    TOMLDecodeError = toml.TomlDecodeError

    def load_toml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            return toml.load(stream)


EXPECTED_DIMENSIONS = (
    (10_000, 100_000, 1e-4),
    (70_000, 700_000, 1e-4),
    (400_000, 7_000_000, 1e-4),
    (700_000, 7_000_000, 1e-4),
    (750_000, 7_500_000, 1e-4),
)
DIGEST = re.compile(r"^[0-9a-f]{64}$")
GENERATION_FIELDS = (
    "m",
    "n",
    "density",
    "replicate",
    "seed",
    "nnz",
    "lambda",
    "lambda_bits",
    "numerical_digest",
    "model_digest",
)
SUCCESSFUL_TERMINATIONS = {"OPTIMAL", "ALMOST_OPTIMAL", "LOCALLY_SOLVED"}


def read_toml(path: Path) -> dict[str, Any]:
    return load_toml(path)


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def validate_manifest(path: Path) -> list[dict[str, Any]]:
    document = read_toml(path)
    if document.get("schema_version") != 1:
        fail("manifest schema_version must be 1")
    if document.get("generator_version") != 1:
        fail("manifest generator_version must be 1")
    if document.get("preset") != "table5":
        fail("manifest preset must be table5")
    if document.get("master_seed") != "20260728":
        fail("manifest master_seed must be 20260728")
    if document.get("replicates") != 5:
        fail("manifest replicates must be 5")
    instances = document.get("instances")
    if not isinstance(instances, list) or len(instances) != 25:
        fail(f"manifest must contain 25 instances, found {len(instances or [])}")

    ids: list[str] = []
    seeds: list[str] = []
    dimension_counts: Counter[tuple[int, int, float]] = Counter()
    replicates: dict[tuple[int, int, float], set[int]] = {}
    required = {"id", "preset", "m", "n", "density", "replicate", "seed"}
    for index, instance in enumerate(instances, start=1):
        missing = required - instance.keys()
        if missing:
            fail(f"manifest instance {index} is missing {sorted(missing)}")
        if instance["preset"] != "table5":
            fail(f"{instance['id']}: preset must be table5")
        dimension = (
            int(instance["m"]),
            int(instance["n"]),
            float(instance["density"]),
        )
        if dimension not in EXPECTED_DIMENSIONS:
            fail(f"{instance['id']}: unexpected dimensions {dimension}")
        replicate = int(instance["replicate"])
        if replicate not in range(1, 6):
            fail(f"{instance['id']}: replicate must be in 1:5")
        try:
            seed = int(instance["seed"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{instance['id']}: seed is not an integer") from error
        if seed <= 0:
            fail(f"{instance['id']}: seed must be positive")
        ids.append(str(instance["id"]))
        seeds.append(str(instance["seed"]))
        dimension_counts[dimension] += 1
        replicates.setdefault(dimension, set()).add(replicate)

    if len(set(ids)) != 25:
        fail("manifest instance IDs are not unique")
    if len(set(seeds)) != 25:
        fail("manifest seeds are not unique")
    for dimension in EXPECTED_DIMENSIONS:
        if dimension_counts[dimension] != 5:
            fail(f"dimension {dimension} does not have exactly five instances")
        if replicates[dimension] != set(range(1, 6)):
            fail(f"dimension {dimension} does not have replicates 1 through 5")
    return instances


def command_validate_manifest(arguments: argparse.Namespace) -> int:
    instances = validate_manifest(arguments.manifest)
    if arguments.ids_only:
        for instance in instances:
            print(instance["id"])
        return 0
    print("id\tm\tn\tdensity\treplicate\tseed")
    for instance in instances:
        print(
            instance["id"],
            instance["m"],
            instance["n"],
            instance["density"],
            instance["replicate"],
            instance["seed"],
            sep="\t",
        )
    print("TABLE5_MANIFEST_VALID instances=25 unique_ids=25 unique_seeds=25")
    return 0


def solver_result(
    solver_dir: Path,
    solver: str,
    instance_id: str,
) -> tuple[str, dict[str, Any] | None, str]:
    case_dir = solver_dir / solver / instance_id
    done = case_dir / "DONE"
    if done.is_file():
        attempt_name = done.read_text(encoding="utf-8").strip()
        if attempt_name in {
            "EARLY_STOP_PENDING",
            "HALF_HOUR_PENDING",
            "IN_PROGRESS_GPU0",
        }:
            return "pending", None, str(done)
        result_path = case_dir / attempt_name / "result.toml"
        if not result_path.is_file():
            return "failed", None, str(result_path)
        result = read_toml(result_path)
        if result.get("run_status") == "skipped":
            return "skipped", result, str(result_path)
        if result.get("run_status") != "completed":
            return "failed", result, str(result_path)
        return "completed", result, str(result_path)

    attempts = sorted(case_dir.glob("attempt_*/result.toml"))
    if attempts:
        result = read_toml(attempts[-1])
        return "failed", result, str(attempts[-1])
    return "pending", None, ""


def validate_generation_result(
    instance: dict[str, Any],
    result: dict[str, Any],
) -> None:
    if result.get("instance_id") != instance["id"]:
        fail(f"{instance['id']}: solver result has wrong instance_id")
    for field in ("m", "n", "density", "replicate", "seed"):
        if result.get(field) != instance[field]:
            fail(
                f"{instance['id']}: result {field}={result.get(field)!r}, "
                f"manifest has {instance[field]!r}"
            )
    for field in ("numerical_digest", "model_digest"):
        if not DIGEST.fullmatch(str(result.get(field, ""))):
            fail(f"{instance['id']}: invalid {field}")
    if int(result.get("nnz", -1)) < 0:
        fail(f"{instance['id']}: invalid nnz")


def collect_rows(
    manifest: Path,
    solver_dir: Path,
    solvers: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instances = validate_manifest(manifest)
    case_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    for instance in instances:
        completed_results: list[dict[str, Any]] = []
        statuses: list[str] = []
        solver_labels: list[str] = []
        first_failure_log = ""
        for solver in solvers:
            execution_status, result, result_path = solver_result(
                solver_dir,
                solver,
                instance["id"],
            )
            if execution_status in {"completed", "skipped"} and result is not None:
                termination_status = str(result.get("termination_status", ""))
                if execution_status == "skipped":
                    status = "skipped"
                else:
                    status = (
                        "success"
                        if termination_status in SUCCESSFUL_TERMINATIONS
                        else "failed"
                    )
            else:
                termination_status = ""
                status = execution_status
            statuses.append(status)
            if status == "failed" and not first_failure_log:
                first_failure_log = result_path
            if result is not None:
                solver_rows.append(
                    {
                        "id": instance["id"],
                        "solver": solver,
                        "run_status": result.get("run_status", ""),
                        "outcome": status,
                        "termination_status": termination_status,
                        "primal_status": result.get("primal_status", ""),
                        "iterations": result.get("iterations", ""),
                        "objective_value": result.get("objective_value", ""),
                        "generation_seconds": result.get(
                            "generation_seconds",
                            "",
                        ),
                        "setup_seconds": result.get("setup_seconds", ""),
                        "optimize_wall_seconds": result.get(
                            "optimize_wall_seconds",
                            "",
                        ),
                        "reason": result.get("reason", result.get("error", "")),
                        "result": result_path,
                    }
                )
            if execution_status == "completed" and result is not None:
                validate_generation_result(instance, result)
                completed_results.append(result)
                solver_labels.append(
                    f"{solver}:{termination_status}({status})"
                )
            elif execution_status == "skipped" and result is not None:
                solver_labels.append(
                    f"{solver}:{termination_status}(skipped)"
                )
            else:
                solver_labels.append(f"{solver}:{status}")

        if completed_results:
            reference = completed_results[0]
            for other in completed_results[1:]:
                for field in GENERATION_FIELDS:
                    if other[field] != reference[field]:
                        fail(
                            f"{instance['id']}: solvers regenerated different "
                            f"values for {field}"
                        )
        else:
            reference = {}
        if "failed" in statuses:
            case_status = "failed"
        elif statuses and all(status == "success" for status in statuses):
            case_status = "success"
        elif "skipped" in statuses:
            case_status = "skipped"
        else:
            case_status = "pending"
        case_rows.append(
            {
                "id": instance["id"],
                "m": instance["m"],
                "n": instance["n"],
                "replicate": instance["replicate"],
                "seed": instance["seed"],
                "nnz": reference.get("nnz", ""),
                "lambda": reference.get("lambda", ""),
                "numerical_digest": reference.get("numerical_digest", ""),
                "model_digest": reference.get("model_digest", ""),
                "generation_seconds": reference.get("generation_seconds", ""),
                "status": case_status,
                "solver_statuses": "; ".join(solver_labels),
                "first_failure_result": first_failure_log,
            }
        )
    return case_rows, solver_rows


def command_validate_runs(arguments: argparse.Namespace) -> int:
    solvers = arguments.solvers.split(",")
    rows, _solver_rows = collect_rows(
        arguments.manifest,
        arguments.solver_dir,
        solvers,
    )
    completed = sum(row["status"] == "success" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    pending = len(rows) - completed - failed - skipped
    recorded = len(rows) - pending
    if arguments.require_complete and pending != 0:
        fail(
            f"expected 25 recorded case outcomes, found {recorded}; "
            f"successful={completed} failed={failed} skipped={skipped} "
            f"pending={pending}"
        )
    print(
        f"TABLE5_RUNS_VALID recorded={recorded}/25 successful={completed} "
        f"failed={failed} skipped={skipped} pending={pending}"
    )
    return 0


def command_report(arguments: argparse.Namespace) -> int:
    solvers = arguments.solvers.split(",")
    rows, solver_rows = collect_rows(
        arguments.manifest,
        arguments.solver_dir,
        solvers,
    )
    arguments.csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arguments.solver_csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.solver_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "id",
            "solver",
            "run_status",
            "outcome",
            "termination_status",
            "primal_status",
            "iterations",
            "objective_value",
            "generation_seconds",
            "setup_seconds",
            "optimize_wall_seconds",
            "reason",
            "result",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(solver_rows)

    success_count = sum(row["status"] == "success" for row in rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    skipped_count = sum(row["status"] == "skipped" for row in rows)
    pending_count = sum(row["status"] == "pending" for row in rows)
    recorded_count = len(rows) - pending_count
    expected_solver_records = len(rows) * len(solvers)
    recorded_solver_records = len(solver_rows)
    solver_outcomes = Counter(
        (row["solver"], row["outcome"]) for row in solver_rows
    )
    dimension_success = Counter(
        (int(row["m"]), int(row["n"]))
        for row in rows
        if row["status"] == "success"
    )
    lines = [
        "# Table 5 large-scale Lasso report",
        "",
        "The instances are regenerated deterministically in memory for each solver; no CBF files are stored.",
        "",
        "| ID | m | n | replicate | seed | nnz | lambda | numerical digest | model digest | generation seconds | status | solver statuses |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {m} | {n} | {replicate} | {seed} | {nnz} | "
            "{lambda} | {numerical_digest} | {model_digest} | "
            "{generation_seconds} | {status} | {solver_statuses} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Recorded case outcomes: {recorded_count}/25",
            (
                "- Recorded solver-case results: "
                f"{recorded_solver_records}/{expected_solver_records}"
            ),
            f"- Cases where every requested solver succeeded: {success_count}/25",
            f"- Failed cases: {failed_count}",
            f"- Skipped cases: {skipped_count}",
            f"- Pending cases: {pending_count}",
        ]
    )
    for solver in solvers:
        lines.append(
            f"- {solver}: "
            f"{solver_outcomes[(solver, 'success')]} successful, "
            f"{solver_outcomes[(solver, 'failed')]} failed, "
            f"{solver_outcomes[(solver, 'skipped')]} skipped"
        )
    for m, n, _density in EXPECTED_DIMENSIONS:
        lines.append(
            f"- m={m}, n={n}: "
            f"{dimension_success[(m, n)]}/5 all-solver successes"
        )
    lines.append(
        f"- All requested results recorded: "
        f"{'yes' if pending_count == 0 and recorded_solver_records == expected_solver_records else 'no'}"
    )
    failed_rows = [row for row in rows if row["status"] == "failed"]
    if failed_rows:
        first = failed_rows[0]
        lines.extend(
            [
                f"- First failed case: {first['id']}",
                f"- Failure result: {first['first_failure_result'] or 'not recorded'}",
            ]
        )
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"TABLE5_REPORT_WRITTEN markdown={arguments.markdown} csv={arguments.csv} "
        f"solver_csv={arguments.solver_csv} "
        f"recorded={recorded_solver_records}/{expected_solver_records} "
        f"successful_cases={success_count} failed_cases={failed_count} "
        f"skipped_cases={skipped_count} pending_cases={pending_count}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)

    validate_manifest_parser = subparsers.add_parser("validate-manifest")
    validate_manifest_parser.add_argument("--manifest", type=Path, required=True)
    validate_manifest_parser.add_argument("--ids-only", action="store_true")
    validate_manifest_parser.set_defaults(function=command_validate_manifest)

    validate_runs_parser = subparsers.add_parser("validate-runs")
    validate_runs_parser.add_argument("--manifest", type=Path, required=True)
    validate_runs_parser.add_argument("--solver-dir", type=Path, required=True)
    validate_runs_parser.add_argument("--solvers", required=True)
    validate_runs_parser.add_argument("--require-complete", action="store_true")
    validate_runs_parser.set_defaults(function=command_validate_runs)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--manifest", type=Path, required=True)
    report_parser.add_argument("--solver-dir", type=Path, required=True)
    report_parser.add_argument("--solvers", required=True)
    report_parser.add_argument("--markdown", type=Path, required=True)
    report_parser.add_argument("--csv", type=Path, required=True)
    report_parser.add_argument("--solver-csv", type=Path, required=True)
    report_parser.set_defaults(function=command_report)
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        return arguments.function(arguments)
    except (OSError, ValueError, TOMLDecodeError) as error:
        print(f"TABLE5_VALIDATION_ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
