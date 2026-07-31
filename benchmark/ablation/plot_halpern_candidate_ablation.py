#!/usr/bin/env python3
"""Create separate figures for the current Halpern-candidate ablation.

Both configurations use the same reflected main sequence and differ only in
whether the auxiliary Halpern point is available as a restart candidate.
"""

from __future__ import annotations

import argparse
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
    "without_halpern_candidate",
    "with_halpern_candidate",
]

CONFIGURATION_LABELS = [
    "Candidate disabled",
    "Candidate enabled",
]

PLOT_TITLE = "Halpern restart candidate"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_run = (
        root
        / "benchmark"
        / "results"
        / "rebuttal"
        / "halpern_candidate"
        / "halpern_candidate_600s_gpu5_20260730"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Plot the completed candidate-only Halpern ablation. "
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
        default="halpern_candidate_ablation",
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
    expected_effect = {"without_halpern_candidate"}
    if set(effects) != expected_effect:
        missing = sorted(expected_effect - set(effects))
        extra = sorted(set(effects) - expected_effect)
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
    jointly_solved = int(
        effects["without_halpern_candidate"]["jointly_solved"]
    )
    if jointly_solved != 56:
        raise ValueError(
            f"expected 56 jointly solved instances, found {jointly_solved}"
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
        title=PLOT_TITLE,
    )


def sgm_tex(overall: Dict[str, Dict[str, str]]) -> str:
    return bar_tex(
        CONFIGURATION_LABELS,
        [
            finite_float(overall[configuration], "sgm10_wall_seconds")
            for configuration in CONFIGURATIONS
        ],
        ylabel="SGM(10) wall time (s)",
        ymax=31,
        yticks=[0, 5, 10, 15, 20, 25, 30],
        color="213,94,0",
        precision=1,
        title=PLOT_TITLE,
    )


def effects_tex(effects: Dict[str, Dict[str, str]]) -> str:
    row = effects["without_halpern_candidate"]
    runtime = finite_float(row, "runtime_ratio_geomean")
    lower = finite_float(row, "runtime_ratio_ci95_lower")
    upper = finite_float(row, "runtime_ratio_ci95_upper")
    iteration = finite_float(row, "iteration_ratio_geomean")
    return ratio_tex(
        ["Halpern candidate"],
        [(0.94, runtime, lower, upper)],
        [(1.06, iteration)],
        ylabel="Disabled / enabled ratio",
        ymin=0.75,
        ymax=1.12,
        yticks=[0.8, 0.9, 1.0, 1.1],
        annotations=(
            r"\node[anchor=south east,font=\scriptsize,text=black!70] "
            r"at (axis cs:1.50,0.77) {56 jointly solved};"
        ),
        width="9.2cm",
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
