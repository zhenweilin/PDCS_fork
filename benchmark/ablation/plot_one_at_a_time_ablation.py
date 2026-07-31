#!/usr/bin/env python3
"""Create separate figures for the current core one-at-a-time ablation.

Halpern is disabled in all six configurations.  The full and fixed-eta records
come from the completed progressive batch; the other four configurations are
strict one-at-a-time removals from the same core solver.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

from plot_progressive_ablation import (
    bar_tex,
    compile_figure,
    finite_float,
    index_unique,
    ratio_tex,
    read_csv,
)


CONFIGURATIONS = [
    "full",
    "no_restart",
    "no_scaling",
    "no_reflection",
    "no_adaptive_primal_weight",
    "no_adaptive_step",
]

CONFIGURATION_LABELS = [
    "Full",
    "No restart",
    "No scaling",
    "No reflection",
    r"Fixed $\omega$",
    r"Fixed $\eta$",
]

ABLATIONS = [
    "no_restart",
    "no_scaling",
    "no_reflection",
    "no_adaptive_primal_weight",
    "no_adaptive_step",
]

ABLATION_LABELS = [
    "No restart",
    "No scaling",
    "No reflection",
    r"Fixed $\omega$",
    r"Fixed $\eta$",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_run = (
        root
        / "benchmark"
        / "results"
        / "rebuttal"
        / "ablation"
        / "one_at_a_time_core_600s_20260731"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Plot the completed one-at-a-time cuPDCS ablation. "
            "Writes separate editable TeX, PDF, and PNG figures."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run,
        help="directory containing summary_overall.csv and paired_effects.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "rebuttal_plan" / "figures",
        help="figure output directory",
    )
    parser.add_argument(
        "--prefix",
        default="one_at_a_time_ablation",
        help="output filename prefix",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300)",
    )
    parser.add_argument(
        "--tex-only",
        action="store_true",
        help="write PGFPlots .tex files without compiling PDF/PNG",
    )
    return parser.parse_args()


def validate_inputs(
    overall: Dict[str, Dict[str, str]],
    effects: Dict[str, Dict[str, str]],
) -> None:
    if set(overall) != set(CONFIGURATIONS):
        missing = sorted(set(CONFIGURATIONS) - set(overall))
        extra = sorted(set(overall) - set(CONFIGURATIONS))
        raise ValueError(
            f"unexpected configurations; missing={missing}, extra={extra}"
        )
    if set(effects) != set(ABLATIONS):
        missing = sorted(set(ABLATIONS) - set(effects))
        extra = sorted(set(effects) - set(ABLATIONS))
        raise ValueError(
            f"unexpected paired effects; missing={missing}, extra={extra}"
        )
    for configuration in CONFIGURATIONS:
        completed = int(overall[configuration]["completed_records"])
        expected = int(overall[configuration]["expected_instances"])
        if completed != 63 or expected != 63:
            raise ValueError(
                f"{configuration} is incomplete: {completed}/{expected}"
            )


def solved_tex(overall: Dict[str, Dict[str, str]]) -> str:
    return bar_tex(
        CONFIGURATION_LABELS,
        [
            float(overall[configuration]["verified_solved"])
            for configuration in CONFIGURATIONS
        ],
        ylabel="Verified solves (out of 63)",
        ymax=66,
        yticks=[0, 10, 20, 30, 40, 50, 60],
        color="0,114,178",
        precision=0,
        title="Core one-at-a-time (Halpern candidate disabled)",
    )


def sgm_tex(overall: Dict[str, Dict[str, str]]) -> str:
    return bar_tex(
        CONFIGURATION_LABELS,
        [
            finite_float(overall[configuration], "sgm10_wall_seconds")
            for configuration in CONFIGURATIONS
        ],
        ylabel="SGM(10) wall time (s)",
        ymax=75,
        yticks=[0, 10, 20, 30, 40, 50, 60, 70],
        color="213,94,0",
        precision=1,
        title="Core one-at-a-time (Halpern candidate disabled)",
    )


def effects_tex(effects: Dict[str, Dict[str, str]]) -> str:
    runtime_points = []
    iteration_points = []
    annotations = []
    for index, configuration in enumerate(ABLATIONS, start=1):
        row = effects[configuration]
        jointly_solved = int(row["jointly_solved"])
        runtime_text = row.get("runtime_ratio_geomean", "")
        iteration_text = row.get("iteration_ratio_geomean", "")
        try:
            runtime = float(runtime_text)
            iteration = float(iteration_text)
        except ValueError as error:
            raise ValueError(
                f"invalid paired ratio for {configuration}: {row}"
            ) from error

        if not math.isfinite(runtime) or not math.isfinite(iteration):
            if jointly_solved != 0:
                raise ValueError(
                    f"{configuration} has non-finite ratios despite "
                    f"{jointly_solved} jointly solved instances"
                )
            annotations.append(
                rf"\node[anchor=north,font=\scriptsize,text=black!70] "
                rf"at (axis cs:{index},0.31) "
                rf"{{0 jointly solved}};"
            )
            continue

        lower = finite_float(row, "runtime_ratio_ci95_lower")
        upper = finite_float(row, "runtime_ratio_ci95_upper")
        runtime_points.append((index - 0.08, runtime, lower, upper))
        iteration_points.append((index + 0.08, iteration))

    return ratio_tex(
        ABLATION_LABELS,
        runtime_points,
        iteration_points,
        ylabel="Ablated / Full ratio",
        ymin=0.2,
        ymax=3.6,
        yticks=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        annotations="\n".join(annotations),
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    overall = index_unique(
        read_csv(run_dir / "summary_overall.csv"), "configuration"
    )
    effects = index_unique(
        read_csv(run_dir / "paired_effects.csv"), "configuration"
    )
    validate_inputs(overall, effects)

    outputs = [
        (
            output_dir / f"{args.prefix}_solved.tex",
            solved_tex(overall),
        ),
        (
            output_dir / f"{args.prefix}_sgm.tex",
            sgm_tex(overall),
        ),
        (
            output_dir / f"{args.prefix}_paired_effects.tex",
            effects_tex(effects),
        ),
    ]

    created: List[Path] = []
    for tex_path, contents in outputs:
        tex_path.write_text(contents, encoding="utf-8")
        created.append(tex_path)
        if not args.tex_only:
            created.extend(compile_figure(tex_path, args.dpi))

    for path in created:
        print(f"CREATED {path}")


if __name__ == "__main__":
    main()
