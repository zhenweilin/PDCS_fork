#!/usr/bin/env python3
"""Create publication-ready figures for the six-stage cuPDCS ablation.

The script uses only the Python standard library.  It writes editable PGFPlots
sources and, unless --tex-only is supplied, compiles PDF and PNG versions with
pdflatex plus either Inkscape or pdftoppm.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List


CONFIGURATIONS = [
    "pdhg",
    "pdhg_restart",
    "pdhg_restart_scaling",
    "pdhg_restart_scaling_reflection",
    "pdhg_restart_scaling_reflection_adaptive_primal_weight",
    "pdhg_restart_scaling_reflection_adaptive",
]

STAGE_LABELS = [
    "PDHG",
    "+Restart",
    "+Scaling",
    "+Reflection",
    r"+Adapt. $\omega$",
    r"+Adapt. $\eta$",
]

COMPONENTS = [
    "Restart",
    "Diagonal rescaling",
    "Reflection",
    "Adaptive primal weight",
    "Adaptive step",
]

COMPONENT_LABELS = [
    "Restart",
    "Scaling",
    "Reflection",
    r"Adapt. $\omega$",
    r"Adapt. $\eta$",
]

BAR_TEMPLATE = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\definecolor{pdcsbar}{RGB}{__COLOR__}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  width=9.2cm,
  height=6.35cm,
  xmin=0.45,
  xmax=__XMAX__,
  xtick={__XTICKS__},
  xticklabels={__LABELS__},
  xticklabel style={rotate=27,anchor=east,font=\small},
  tick label style={font=\small},
  label style={font=\small},
  ylabel={__YLABEL__},
  ymin=0,
  ymax=__YMAX__,
  ytick={__YTICKS__},
  ymajorgrids=true,
  grid style={gray!22},
  axis line style={gray!65},
  tick style={gray!65},
  enlarge x limits=false,
]
\addplot[
  ybar,
  bar width=10.5pt,
  fill=pdcsbar!78,
  draw=pdcsbar,
  line width=0.45pt,
  point meta=y,
  nodes near coords={
    \pgfmathprintnumber[fixed,precision=__PRECISION__]
      {\pgfplotspointmeta}
  },
  every node near coord/.append style={font=\scriptsize},
] coordinates {__COORDS__};
\end{axis}
\end{tikzpicture}
\end{document}
"""

EFFECTS_TEMPLATE = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\definecolor{pdcsblue}{RGB}{0,114,178}
\definecolor{pdcsorange}{RGB}{213,94,0}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  width=15.7cm,
  height=7.1cm,
  xmin=0.45,
  xmax=__XMAX__,
  ymin=__YMIN__,
  ymax=__YMAX__,
  xtick={__XTICKS__},
  xticklabels={__COMPONENT_LABELS__},
  xticklabel style={rotate=20,anchor=east,font=\small},
  ytick={__YTICKS__},
  ylabel={__YLABEL__},
  tick label style={font=\small},
  label style={font=\small},
  legend style={
    at={(0.5,1.02)},
    anchor=south,
    legend columns=2,
    draw=none,
    font=\small
  },
  ymajorgrids=true,
  grid style={gray!22},
  axis line style={gray!65},
  tick style={gray!65},
]
\addplot[black!65,dashed,line width=0.8pt,forget plot]
  coordinates {(0.45,1) (__XMAX__,1)};
\node[anchor=south east,font=\scriptsize,text=black!65]
  at (axis cs:__NO_CHANGE_X__,1.01) {no change};

\addplot+[
  only marks,
  mark=*,
  mark size=2.6pt,
  color=pdcsblue,
  mark options={fill=pdcsblue},
  error bars/.cd,
  y dir=both,
  y explicit,
  error bar style={line width=0.85pt},
  error mark options={rotate=90,mark size=3pt,line width=0.85pt},
] coordinates {
__RUNTIME_COORDS__
};
\addlegendentry{Wall time (95\% bootstrap CI)}

\addplot+[
  only marks,
  mark=triangle*,
  mark size=3.1pt,
  color=pdcsorange,
  mark options={fill=pdcsorange},
] coordinates {
__ITERATION_COORDS__
};
\addlegendentry{Iterations}
__ANNOTATIONS__
\end{axis}
\end{tikzpicture}
\end{document}
"""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_run = (
        root
        / "benchmark"
        / "results"
        / "rebuttal"
        / "progressive_ablation"
        / "progressive_six_stage_600s_all_idle_20260730"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Plot the completed six-stage progressive cuPDCS ablation. "
            "Writes editable TeX plus PDF/PNG figures."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run,
        help="directory containing summary_overall.csv and adjacent_effects.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "rebuttal_plan" / "figures",
        help="figure output directory",
    )
    parser.add_argument(
        "--prefix",
        default="progressive_ablation",
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required result file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_float(row: Dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field!r} in row: {row}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field!r} in row: {row}")
    return value


def index_unique(
    rows: Iterable[Dict[str, str]], field: str
) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = row.get(field, "")
        if not key:
            raise ValueError(f"missing {field!r} in row: {row}")
        if key in result:
            raise ValueError(f"duplicate {field}={key!r}")
        result[key] = row
    return result


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
    if set(effects) != set(COMPONENTS):
        missing = sorted(set(COMPONENTS) - set(effects))
        extra = sorted(set(effects) - set(COMPONENTS))
        raise ValueError(
            f"unexpected adjacent effects; missing={missing}, extra={extra}"
        )
    for configuration in CONFIGURATIONS:
        completed = int(overall[configuration]["completed_records"])
        expected = int(overall[configuration]["expected_instances"])
        if completed != 63 or expected != 63:
            raise ValueError(
                f"{configuration} is incomplete: {completed}/{expected}"
            )


def bar_tex(
    labels: List[str],
    values: List[float],
    *,
    ylabel: str,
    ymax: float,
    yticks: List[float],
    color: str,
    precision: int,
) -> str:
    if len(labels) != len(values) or not labels:
        raise ValueError("bar labels and values must have the same nonzero length")
    coordinates = " ".join(
        f"({index},{value:.8g})"
        for index, value in enumerate(values, start=1)
    )
    xticks = ",".join(str(index) for index in range(1, len(labels) + 1))
    return (
        BAR_TEMPLATE.replace("__COLOR__", color)
        .replace("__XMAX__", f"{len(labels) + 0.55:.2f}")
        .replace("__XTICKS__", xticks)
        .replace("__LABELS__", ",".join(labels))
        .replace("__YLABEL__", ylabel)
        .replace("__YMAX__", f"{ymax:.8g}")
        .replace("__YTICKS__", ",".join(f"{tick:g}" for tick in yticks))
        .replace("__PRECISION__", str(precision))
        .replace("__COORDS__", coordinates)
    )


def ratio_tex(
    labels: List[str],
    runtime_points: List[tuple[float, float, float, float]],
    iteration_points: List[tuple[float, float]],
    *,
    ylabel: str,
    ymin: float,
    ymax: float,
    yticks: List[float],
    annotations: str = "",
) -> str:
    if not labels:
        raise ValueError("ratio plot needs at least one x-axis label")
    runtime_coordinates = []
    iteration_coordinates = []
    for x_value, runtime, lower, upper in runtime_points:
        if not lower <= runtime <= upper:
            raise ValueError(
                f"runtime ratio outside CI: "
                f"{lower} <= {runtime} <= {upper}"
            )
        runtime_coordinates.append(
            f"  ({x_value:.2f},{runtime:.8g}) "
            f"+= (0,{upper - runtime:.8g}) "
            f"-= (0,{runtime - lower:.8g})"
        )
    for x_value, iteration in iteration_points:
        iteration_coordinates.append(f"  ({x_value:.2f},{iteration:.8g})")
    xticks = ",".join(str(index) for index in range(1, len(labels) + 1))
    xmax = len(labels) + 0.55
    return (
        EFFECTS_TEMPLATE.replace("__XMAX__", f"{xmax:.2f}")
        .replace("__YMIN__", f"{ymin:.8g}")
        .replace("__YMAX__", f"{ymax:.8g}")
        .replace("__XTICKS__", xticks)
        .replace("__COMPONENT_LABELS__", ",".join(labels))
        .replace("__YTICKS__", ",".join(f"{tick:g}" for tick in yticks))
        .replace("__YLABEL__", ylabel)
        .replace("__NO_CHANGE_X__", f"{xmax - 0.03:.2f}")
        .replace("__RUNTIME_COORDS__", "\n".join(runtime_coordinates))
        .replace("__ITERATION_COORDS__", "\n".join(iteration_coordinates))
        .replace("__ANNOTATIONS__", annotations)
    )


def effects_tex(effects: Dict[str, Dict[str, str]]) -> str:
    runtime_points = []
    iteration_points = []
    for index, component in enumerate(COMPONENTS, start=1):
        row = effects[component]
        runtime = finite_float(row, "runtime_ratio_after_over_before")
        lower = finite_float(row, "runtime_ratio_ci95_lower")
        upper = finite_float(row, "runtime_ratio_ci95_upper")
        iteration = finite_float(row, "iteration_ratio_after_over_before")
        runtime_points.append((index - 0.08, runtime, lower, upper))
        iteration_points.append((index + 0.08, iteration))
    return ratio_tex(
        COMPONENT_LABELS,
        runtime_points,
        iteration_points,
        ylabel="Ratio after / before adding component",
        ymin=0.2,
        ymax=1.58,
        yticks=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    )


def solved_tex(overall: Dict[str, Dict[str, str]]) -> str:
    return bar_tex(
        STAGE_LABELS,
        [
            float(overall[configuration]["verified_solved"])
            for configuration in CONFIGURATIONS
        ],
        ylabel="Verified solves (out of 63)",
        ymax=66,
        yticks=[0, 10, 20, 30, 40, 50, 60],
        color="0,114,178",
        precision=0,
    )


def sgm_tex(overall: Dict[str, Dict[str, str]]) -> str:
    return bar_tex(
        STAGE_LABELS,
        [
            finite_float(overall[configuration], "sgm10_wall_seconds")
            for configuration in CONFIGURATIONS
        ],
        ylabel="SGM(10) wall time (s)",
        ymax=100,
        yticks=[0, 20, 40, 60, 80, 100],
        color="213,94,0",
        precision=1,
    )


def run_checked(command: List[str], cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{tail}"
        )


def compile_figure(tex_path: Path, dpi: int) -> tuple[Path, Path]:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError(
            "pdflatex was not found; install TeX/PGFPlots or use --tex-only"
        )
    pdf_path = tex_path.with_suffix(".pdf")
    png_path = tex_path.with_suffix(".png")
    with tempfile.TemporaryDirectory(prefix="pdcs_ablation_plot_") as temp_name:
        temp_dir = Path(temp_name)
        run_checked(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={temp_dir}",
                str(tex_path),
            ],
            cwd=tex_path.parent,
        )
        built_pdf = temp_dir / pdf_path.name
        if not built_pdf.is_file():
            raise RuntimeError(f"pdflatex did not create {built_pdf}")
        shutil.copy2(built_pdf, pdf_path)

    inkscape = shutil.which("inkscape")
    if inkscape is not None:
        run_checked(
            [
                inkscape,
                str(pdf_path),
                "--export-type=png",
                f"--export-filename={png_path}",
                f"--export-dpi={dpi}",
                "--export-background=white",
                "--export-background-opacity=1",
            ],
            cwd=tex_path.parent,
        )
    else:
        pdftoppm = shutil.which("pdftoppm")
        if pdftoppm is None:
            raise RuntimeError(
                "neither inkscape nor pdftoppm was found; PDF was created "
                "but PNG conversion is unavailable"
            )
        run_checked(
            [
                pdftoppm,
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                str(pdf_path),
                str(png_path.with_suffix("")),
            ],
            cwd=tex_path.parent,
        )
    if not png_path.is_file():
        raise RuntimeError(f"PNG conversion did not create {png_path}")
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    overall_rows = read_csv(run_dir / "summary_overall.csv")
    effect_rows = read_csv(run_dir / "adjacent_effects.csv")
    overall = index_unique(overall_rows, "configuration")
    effects = index_unique(effect_rows, "component_added")
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
