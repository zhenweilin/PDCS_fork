#!/usr/bin/env python3
"""Verify the explicit family/branch/scaling/hierarchy coverage contract."""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path


FAMILY_BRANCHES = {
    "soc": (
        "inside", "polar", "boundary_inside", "boundary_outside",
        "root_decreasing", "root_increasing",
    ),
    "exp": (
        "inside", "polar", "boundary", "root_positive", "root_negative",
        "near_degenerate",
    ),
    "dual_exp": (
        "inside", "polar", "boundary", "root_positive", "root_negative",
        "near_degenerate",
    ),
}
SCALINGS = ("identity", "scalar", "diagonal")
STRATEGIES = ("gridWise", "blockWise", "warpWise", "threadWise")


def truth(value: str) -> bool:
    return value.lower() == "true"


def family(row: dict[str, str]) -> str:
    case_id = row.get("case_id", "")
    for name in ("dual_exp", "soc", "exp"):
        if case_id.startswith(f"branch_{name}_"):
            return name
    return ""


def expected_keys() -> set[tuple[str, str, str, str]]:
    return {
        (name, branch, scaling, strategy)
        for name, branches in FAMILY_BRANCHES.items()
        for branch in branches
        for scaling in SCALINGS
        for strategy in STRATEGIES
    }


def load(patterns: list[str]) -> tuple[list[Path], list[dict[str, str]]]:
    paths = sorted({Path(path).resolve() for pattern in patterns
                    for path in glob.glob(pattern)})
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("category") == "branch":
                    row["source_csv"] = str(path)
                    rows.append(row)
    return paths, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths, rows = load(args.inputs)
    expected = expected_keys()
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            family(row), row.get("branch", ""), row.get("scaling", ""),
            row.get("strategy", ""),
        )
        grouped.setdefault(key, []).append(row)

    observed = set(grouped)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(key for key, values in grouped.items() if len(values) != 1)
    failures: list[tuple[tuple[str, str, str, str], str]] = []
    reference_invalid = 0
    for key in sorted(expected & observed):
        row = grouped[key][0]
        if row.get("status") != "OK":
            failures.append((key, f"status={row.get('status', '')}"))
        elif not truth(row.get("output_finite", "")):
            failures.append((key, "non-finite output"))
        elif truth(row.get("reference_valid", "")) and not truth(
            row.get("correct", "")
        ):
            failures.append((key, "valid-reference correctness failure"))
        if not truth(row.get("reference_valid", "")):
            reference_invalid += 1

    counts = Counter(key[0] for key in expected & observed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "family", "branch", "scaling", "strategy", "case_id", "status",
        "output_finite", "reference_valid", "correct", "validation_scope",
        "source_csv",
    ]
    with (args.output_dir / "branch_rows.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key in sorted(expected & observed):
            row = grouped[key][0]
            writer.writerow({
                "family": key[0], "branch": key[1], "scaling": key[2],
                "strategy": key[3], **{name: row.get(name, "")
                                        for name in columns[4:]},
            })

    passed = not (missing or unexpected or duplicates or failures)
    lines = ["# Projection branch-coverage gate", ""]
    lines.append(
        f"Input CSV files: {len(paths)}; required Cartesian cells: "
        f"{len(expected)}; observed: {len(expected & observed)}."
    )
    lines.append("")
    lines.append(
        "Gate: " + ("PASS" if passed else "FAIL") + ". Every required cell "
        "must execute successfully, return a finite candidate, and pass the "
        "correctness gate whenever the independent reference is valid."
    )
    lines.extend([
        "", "| Family | Required | Observed |", "|---|---:|---:|",
    ])
    for name, branches in FAMILY_BRANCHES.items():
        required = len(branches) * len(SCALINGS) * len(STRATEGIES)
        lines.append(f"| {name} | {required} | {counts[name]} |")
    lines.extend([
        "",
        f"Reference-invalid extreme diagnostics: {reference_invalid}; missing: "
        f"{len(missing)}; unexpected: {len(unexpected)}; duplicate cells: "
        f"{len(duplicates)}; execution/correctness failures: {len(failures)}.",
        "",
        "The contract spans every declared mathematical branch, identity/"
        "scalar/diagonal scaling, and grid/block/warp/thread execution. It "
        "contains no RSOC family.",
        "",
    ])
    if missing or unexpected or duplicates or failures:
        lines.extend(["## Discrepancies", ""])
        lines.extend(f"- missing: `{key}`" for key in missing)
        lines.extend(f"- unexpected: `{key}`" for key in unexpected)
        lines.extend(f"- duplicate: `{key}`" for key in duplicates)
        lines.extend(f"- failure: `{key}` ({reason})" for key, reason in failures)
        lines.append("")
    report = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
