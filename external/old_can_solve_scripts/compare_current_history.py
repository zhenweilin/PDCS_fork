#!/usr/bin/env python3
"""Create one strict-metric table for current and historical GPU runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CASES = (
    "batch",
    "batchs101006m",
    "batchs121208m",
    "batchs151208m",
    "batchs201210m",
    "enpro56",
)
CURRENT_VARIANTS = ("no_halpern_full", "inline_halpern_full")
FIELDS = (
    "case",
    "method",
    "time_limit_seconds",
    "termination_status",
    "verified_solved",
    "iterations",
    "solver_seconds",
    "wall_seconds",
    "l_inf_rel_primal_res",
    "l_inf_rel_dual_res",
    "relative_gap",
    "max_metric",
    "result_json",
)


def row_from_result(path: Path, case: str, method: str) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result.get("metrics", {})
    values = [
        metrics.get("l_inf_rel_primal_res"),
        metrics.get("l_inf_rel_dual_res"),
        metrics.get("relative_gap"),
    ]
    finite_values = [value for value in values if isinstance(value, (int, float))]
    max_metric = (
        max(finite_values)
        if len(finite_values) == 3
        else result.get(
            "verification_metric_max",
            result.get("verification_metric_max_from_printed_summary", ""),
        )
    )
    return {
        "case": case,
        "method": method,
        "time_limit_seconds": result.get("time_limit_seconds", ""),
        "termination_status": result.get("termination_status", ""),
        "verified_solved": result.get("verified_solved", False),
        "iterations": result.get(
            "iterations",
            result.get("iteration_count", metrics.get("iterations", "")),
        ),
        "solver_seconds": result.get(
            "solver_seconds",
            metrics.get("solve_time_sec", ""),
        ),
        "wall_seconds": result.get("wall_seconds", ""),
        "l_inf_rel_primal_res": values[0] if values[0] is not None else "",
        "l_inf_rel_dual_res": values[1] if values[1] is not None else "",
        "relative_gap": values[2] if values[2] is not None else "",
        "max_metric": max_metric,
        "result_json": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument(
        "--history-run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="May be supplied more than once.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    current_dir = args.current_dir.resolve()
    for case in CASES:
        matches = list(current_dir.glob(f"cases/*__exp_cone__{case}"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one current directory for {case}, got {matches}")
        for variant in CURRENT_VARIANTS:
            path = matches[0] / variant / "attempt_001" / "result.json"
            rows.append(row_from_result(path, case, f"current_{variant}"))

    for specification in args.history_run:
        if "=" not in specification:
            raise ValueError(f"history run must be LABEL=PATH: {specification}")
        label, raw_path = specification.split("=", 1)
        run_dir = Path(raw_path).resolve()
        for case in CASES:
            path = run_dir / "cases" / case / "result.json"
            if path.is_file():
                rows.append(row_from_result(path, case, label))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"GPU_HISTORY_COMPARISON rows={len(rows)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
