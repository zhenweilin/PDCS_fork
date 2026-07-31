#!/usr/bin/env python3
"""Inspect the fixed CBF ablation dataset and create reproducible run order."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import random
from pathlib import Path


CONE_NAMES = {"F", "L+", "L-", "L=", "Q", "QR", "EXP", "EXP*"}
CONFIGS = (
    "full",
    "no_scaling",
    "no_adaptive_step",
    "no_adaptive_primal_weight",
    "no_restart",
    "no_reflection",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_lines(raw: bytes) -> list[str]:
    text = raw.decode("utf-8")
    lines = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def section_index(lines: list[str], name: str) -> int | None:
    for index, line in enumerate(lines):
        if line == name:
            return index
    return None


def parse_ints(line: str) -> list[int]:
    return [int(value) for value in line.split()]


def parse_cone_section(lines: list[str], name: str) -> tuple[int, list[tuple[str, int]]]:
    index = section_index(lines, name)
    if index is None:
        return 0, []
    header = parse_ints(lines[index + 1])
    if len(header) != 2:
        raise ValueError(f"{name} header must contain total dimension and block count")
    total_dimension, block_count = header
    blocks: list[tuple[str, int]] = []
    for offset in range(block_count):
        tokens = lines[index + 2 + offset].split()
        if len(tokens) != 2 or tokens[0] not in CONE_NAMES:
            raise ValueError(f"invalid {name} cone block: {lines[index + 2 + offset]}")
        blocks.append((tokens[0], int(tokens[1])))
    if sum(dimension for _, dimension in blocks) != total_dimension:
        raise ValueError(f"{name} block dimensions do not sum to {total_dimension}")
    return total_dimension, blocks


def parse_count_section(lines: list[str], name: str) -> int:
    index = section_index(lines, name)
    if index is None:
        return 0
    values = parse_ints(lines[index + 1])
    if len(values) != 1:
        raise ValueError(f"{name} count line must contain one integer")
    return values[0]


def parse_version(lines: list[str]) -> int:
    index = section_index(lines, "VER")
    if index is None:
        raise ValueError("missing VER section")
    return int(lines[index + 1])


def cone_summary(blocks: list[tuple[str, int]]) -> tuple[dict[str, int], dict[str, int]]:
    block_counts = {name: 0 for name in sorted(CONE_NAMES)}
    dimensions = {name: 0 for name in sorted(CONE_NAMES)}
    for name, dimension in blocks:
        block_counts[name] += 1
        dimensions[name] += dimension
    return block_counts, dimensions


def inspect(path: Path) -> dict[str, object]:
    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    lines = clean_lines(raw)
    var_dimension, var_blocks = parse_cone_section(lines, "VAR")
    con_dimension, con_blocks = parse_cone_section(lines, "CON")
    all_blocks = var_blocks + con_blocks
    block_counts, dimensions = cone_summary(all_blocks)
    has_soc = block_counts["Q"] > 0 or block_counts["QR"] > 0
    has_exp = block_counts["EXP"] > 0 or block_counts["EXP*"] > 0
    if has_soc and has_exp:
        cone_mix = "soc_exp"
    elif has_soc:
        cone_mix = "soc_without_exp"
    elif has_exp:
        cone_mix = "exp_without_soc"
    else:
        cone_mix = "linear_only"
    nnz_a = parse_count_section(lines, "ACOORD")
    if nnz_a < 50_000:
        size_class = "small"
    elif nnz_a <= 500_000:
        size_class = "medium"
    else:
        size_class = "large"
    result: dict[str, object] = {
        "instance_id": path.name.removesuffix(".cbf.gz"),
        "filename": path.name,
        "compressed_sha256": sha256_bytes(compressed),
        "decompressed_sha256": sha256_bytes(raw),
        "cbf_version": parse_version(lines),
        "num_variables": var_dimension,
        "num_constraints": con_dimension,
        "nnz_a": nnz_a,
        "cone_mix": cone_mix,
        "size_class": size_class,
        "parse_status": "PASS",
    }
    for cone in sorted(CONE_NAMES):
        safe = (
            cone.replace("+", "pos")
            .replace("-", "neg")
            .replace("=", "zero")
            .replace("*", "dual")
        )
        result[f"{safe}_blocks"] = block_counts[cone]
        result[f"{safe}_dimension"] = dimensions[cone]
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-order", type=Path)
    parser.add_argument("--order-seed", type=int, default=20260728)
    parser.add_argument("--expected-count", type=int, default=63)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.cbf.gz"))
    if len(files) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} .cbf.gz files, found {len(files)}"
        )
    rows = [inspect(path) for path in files]
    write_csv(args.output, rows)

    if args.run_order is not None:
        rng = random.Random(args.order_seed)
        order_rows: list[dict[str, object]] = []
        task_index = 0
        for row in rows:
            configs = list(CONFIGS)
            rng.shuffle(configs)
            for within_instance_order, config in enumerate(configs, start=1):
                task_index += 1
                order_rows.append(
                    {
                        "task_index": task_index,
                        "instance_id": row["instance_id"],
                        "filename": row["filename"],
                        "configuration": config,
                        "within_instance_order": within_instance_order,
                        "order_seed": args.order_seed,
                    }
                )
        write_csv(args.run_order, order_rows)

    by_size: dict[str, int] = {}
    by_mix: dict[str, int] = {}
    for row in rows:
        by_size[str(row["size_class"])] = by_size.get(str(row["size_class"]), 0) + 1
        by_mix[str(row["cone_mix"])] = by_mix.get(str(row["cone_mix"]), 0) + 1
    print(f"MANIFEST_COMPLETE instances={len(rows)}")
    print(f"SIZE_COUNTS {by_size}")
    print(f"CONE_MIX_COUNTS {by_mix}")
    if args.run_order is not None:
        print(f"RUN_ORDER_COMPLETE tasks={len(rows) * len(CONFIGS)}")


if __name__ == "__main__":
    main()
