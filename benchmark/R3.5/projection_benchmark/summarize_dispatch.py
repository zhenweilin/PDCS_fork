#!/usr/bin/env python3
"""Pair heterogeneous-dispatch timings on the same GPU and artifact."""

import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict


def truth(value):
    return str(value).lower() == "true"


def geometric_mean(values):
    values = [value for value in values if value > 0 and math.isfinite(value)]
    return math.exp(statistics.mean(map(math.log, values))) if values else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = sorted({path for pattern in args.inputs for path in glob.glob(pattern)})
    groups = defaultdict(dict)
    for path in paths:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") != "OK" or not truth(row.get("correct")):
                    continue
                key = (row.get("device"), row.get("case_id"),
                       row.get("artifact_sha256"))
                groups[key][truth(row.get("heterogeneous"))] = row

    paired = []
    for (device, case_id, artifact), rows in sorted(groups.items()):
        if True not in rows or False not in rows:
            continue
        on, off = rows[True], rows[False]
        on_us = float(on["latency_median_us"])
        off_us = float(off["latency_median_us"])
        paired.append({
            "device": device, "case_id": case_id,
            "category": on.get("category", ""),
            "strategy": on.get("strategy", ""),
            "artifact_sha256": artifact,
            "off_us": off_us, "on_us": on_us,
            "speedup_off_over_on": off_us / on_us,
            "plan_compacted": on.get("plan_compacted", ""),
            "plan_serial": on.get("plan_serial", ""),
            "plan_thread_soc": on.get("plan_thread_soc", ""),
            "plan_warp_soc": on.get("plan_warp_soc", ""),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "dispatch_ab.csv")
    columns = list(paired[0]) if paired else ["device", "case_id"]
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(paired)

    speedups = [row["speedup_off_over_on"] for row in paired]
    wins = sum(value > 1.0 for value in speedups)
    md_path = os.path.join(args.output_dir, "summary.md")
    with open(md_path, "w") as stream:
        stream.write("# Same-GPU heterogeneous dispatch A/B\n\n")
        stream.write(f"Paired rows: {len(paired)}; dispatch wins: {wins}; ")
        stream.write(f"geometric-mean speedup: {geometric_mean(speedups):.3f}x.\n\n")
        stream.write("| GPU | Case | Off us | On us | Speedup | Compact/thread/warp/serial |\n")
        stream.write("|---:|---|---:|---:|---:|---|\n")
        for row in sorted(paired, key=lambda item: item["speedup_off_over_on"], reverse=True):
            layout = "/".join(row[name] for name in (
                "plan_compacted", "plan_thread_soc", "plan_warp_soc", "plan_serial"))
            stream.write(f"| {row['device']} | {row['case_id']} | {row['off_us']:.3f} | ")
            stream.write(f"{row['on_us']:.3f} | {row['speedup_off_over_on']:.3f}x | {layout} |\n")
    print(md_path)


if __name__ == "__main__":
    main()
