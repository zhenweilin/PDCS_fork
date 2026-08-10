#!/usr/bin/env python3
"""Summarize archived-PDCS reruns without changing their raw logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "case",
    "input_sha256",
    "old_source_commit",
    "run_status",
    "termination_status",
    "verified_solved",
    "iterations",
    "solver_seconds",
    "wall_seconds",
    "verification_metric_max_from_printed_summary",
    "l_inf_rel_primal_res",
    "l_inf_rel_dual_res",
    "relative_gap",
    "cuda_visible_devices",
    "raw_log",
    "result_json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=6)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("cases/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result.get("metrics", {})
        row = {key: result.get(key, "") for key in FIELDS}
        for key in (
            "l_inf_rel_primal_res",
            "l_inf_rel_dual_res",
            "relative_gap",
        ):
            row[key] = metrics.get(key, "")
        row["result_json"] = str(path)
        rows.append(row)
    output = run_dir / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    solved = sum(row["verified_solved"] is True for row in rows)
    errors = sum(row["run_status"] != "COMPLETED" for row in rows)
    tolerances = {
        float(json.loads(path.read_text(encoding="utf-8"))["tolerance"])
        for path in sorted(run_dir.glob("cases/*/result.json"))
    }
    tolerance_text = (
        format(next(iter(tolerances)), ".8g") if len(tolerances) == 1 else "the requested tolerance"
    )
    lines = [
        "# Archived PDCS EXP-case rerun",
        "",
        f"Completed records: {len(rows)}/{args.expected_count}; verified solved: "
        f"{solved}/{args.expected_count}; runtime errors: {errors}.",
        "",
        "A case is marked verified only when the archived solver reports `OPTIMAL` "
        "and all three residual/gap values printed in its final summary are at most "
        f"`{tolerance_text}`.",
        "",
        "| Case | Termination | Verified | Iterations | Solver seconds | Max metric |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case']}` | {row['termination_status']} | "
            f"{row['verified_solved']} | {row['iterations']} | "
            f"{row['solver_seconds']} | "
            f"{row['verification_metric_max_from_printed_summary']} |"
        )
    lines.extend(
        [
            "",
            "Complete archived solver output is retained in `cases/<case>/solver.raw.log`.",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"OLD_PDCS_SUMMARY records={len(rows)} solved={solved} errors={errors}")


if __name__ == "__main__":
    main()
