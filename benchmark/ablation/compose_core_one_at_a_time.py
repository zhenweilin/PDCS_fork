#!/usr/bin/env python3
"""Compose the six-config core one-at-a-time ablation without rerunning data.

The completed progressive batch already contains the current full configuration
and the configuration with adaptive eta disabled.  This script links those two
records with the four missing one-at-a-time configurations, verifies that both
source batches use the identical 63-instance manifest, and runs the standard
ablation analyzer on the composed view.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path


CONFIGURATIONS = (
    "full",
    "no_scaling",
    "no_adaptive_step",
    "no_adaptive_primal_weight",
    "no_restart",
    "no_reflection",
)

PROGRESSIVE_SOURCE = {
    "full": "pdhg_restart_scaling_reflection_adaptive",
    "no_adaptive_step": (
        "pdhg_restart_scaling_reflection_adaptive_primal_weight"
    ),
}

MISSING_SOURCE = {
    "no_scaling": "no_scaling",
    "no_adaptive_primal_weight": "no_adaptive_primal_weight",
    "no_restart": "no_restart",
    "no_reflection": "no_reflection",
}

EXPECTED_FLAGS = {
    "full": (True, True, True, True, True, False),
    "no_scaling": (False, True, True, True, True, False),
    "no_adaptive_step": (True, False, True, True, True, False),
    "no_adaptive_primal_weight": (
        True,
        True,
        False,
        True,
        True,
        False,
    ),
    "no_restart": (True, True, True, False, True, False),
    "no_reflection": (True, True, True, True, False, False),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--progressive-run",
        type=Path,
        default=(
            root
            / "benchmark"
            / "results"
            / "rebuttal"
            / "progressive_ablation"
            / "progressive_six_stage_600s_all_idle_20260730"
        ),
    )
    parser.add_argument("--missing-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 63:
        raise ValueError(f"expected 63 manifest rows in {path}, found {len(rows)}")
    return rows


def same_manifest(
    progressive: list[dict[str, str]],
    missing: list[dict[str, str]],
) -> bool:
    keys = ("instance_id", "compressed_sha256", "decompressed_sha256")
    progressive_ids = {
        tuple(row[key] for key in keys)
        for row in progressive
    }
    missing_ids = {
        tuple(row[key] for key in keys)
        for row in missing
    }
    return progressive_ids == missing_ids


def validate_source_flags(source: Path, configuration: str) -> None:
    attempt_name = (source / "DONE").read_text(encoding="utf-8").strip()
    result_path = source / attempt_name / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing selected result: {result_path}")
    with result_path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    flags = result.get("resolved_flags", {})
    observed = (
        flags.get("diagonal_rescaling"),
        flags.get("adaptive_step"),
        flags.get("adaptive_primal_weight"),
        flags.get("restart"),
        flags.get("reflection"),
        flags.get("halpern"),
    )
    expected = EXPECTED_FLAGS[configuration]
    if observed != expected:
        raise ValueError(
            f"flag mismatch for {configuration} at {result_path}: "
            f"expected={expected}, observed={observed}"
        )


def link_configuration(
    source: Path,
    destination: Path,
    configuration: str,
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"missing completed case directory: {source}")
    if not (source / "DONE").is_file():
        raise FileNotFoundError(f"case directory has no DONE marker: {source}")
    validate_source_flags(source, configuration)
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source, start=destination.parent)
    if destination.is_symlink():
        if os.readlink(destination) != relative_source:
            raise FileExistsError(
                f"existing symlink has a different target: {destination}"
            )
        return
    if destination.exists():
        raise FileExistsError(
            f"refusing to replace existing composed path: {destination}"
        )
    destination.symlink_to(relative_source, target_is_directory=True)


def write_source_map(
    path: Path,
    progressive_run: Path,
    missing_run: Path,
) -> None:
    rows = []
    for configuration in CONFIGURATIONS:
        if configuration in PROGRESSIVE_SOURCE:
            source_run = progressive_run
            source_configuration = PROGRESSIVE_SOURCE[configuration]
        else:
            source_run = missing_run
            source_configuration = MISSING_SOURCE[configuration]
        rows.append(
            {
                "configuration": configuration,
                "source_run": str(source_run),
                "source_configuration": source_configuration,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    progressive_run = args.progressive_run.expanduser().resolve()
    missing_run = args.missing_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    progressive_manifest = read_manifest(progressive_run / "manifest.csv")
    missing_manifest = read_manifest(missing_run / "manifest.csv")
    if not same_manifest(progressive_manifest, missing_manifest):
        raise ValueError("source batches do not use the same 63 instances")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(progressive_run / "manifest.csv", output_dir / "manifest.csv")

    for row in progressive_manifest:
        instance_id = row["instance_id"]
        for configuration in CONFIGURATIONS:
            if configuration in PROGRESSIVE_SOURCE:
                source = (
                    progressive_run
                    / "cases"
                    / instance_id
                    / PROGRESSIVE_SOURCE[configuration]
                )
            else:
                source = (
                    missing_run
                    / "cases"
                    / instance_id
                    / MISSING_SOURCE[configuration]
                )
            link_configuration(
                source,
                output_dir / "cases" / instance_id / configuration,
                configuration,
            )

    write_source_map(
        output_dir / "source_map.csv",
        progressive_run,
        missing_run,
    )

    analyzer = root / "benchmark" / "ablation" / "analyze_ablation.py"
    command = [
        "python3",
        str(analyzer),
        "--run-dir",
        str(output_dir),
        "--configs",
        ",".join(CONFIGURATIONS),
        "--tolerance",
        str(args.tolerance),
        "--timeout-value",
        str(args.time_limit),
        "--sgm-shift",
        "10",
        "--bootstrap-samples",
        "10000",
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--report",
        str(output_dir / "report.md"),
    ]
    subprocess.run(command, cwd=root, check=True)
    print(
        "CORE_ONE_AT_A_TIME_COMPOSED "
        f"records={63 * len(CONFIGURATIONS)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
