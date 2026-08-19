#!/usr/bin/env python3
"""Compare two projection artifacts using same-case, same-GPU rows."""

import argparse
import csv
import glob
import math
import os
import statistics


def truth(value):
    return str(value).lower() == "true"


def read_rows(patterns, label):
    rows = {}
    bad = []
    diagnostics = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("variant") != label:
                        continue
                    key = (row.get("device"), row.get("case_id"))
                    rows[key] = row
                    if row.get("status") != "OK":
                        bad.append((key, row.get("status"), row.get("error", "")))
                    elif row.get("reference_valid", "").lower() == "false":
                        diagnostics.append(key)
                    elif not truth(row.get("correct")):
                        bad.append((key, row.get("status"), row.get("error", "")))
    return rows, bad, diagnostics


def geometric_mean(values):
    values = [x for x in values if x > 0 and math.isfinite(x)]
    return math.exp(statistics.mean(map(math.log, values))) if values else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    baseline, baseline_bad, baseline_diagnostics = read_rows(
        args.baseline, args.baseline_label,
    )
    candidate, candidate_bad, candidate_diagnostics = read_rows(
        args.candidate, args.candidate_label,
    )
    pairs = []
    for key in sorted(baseline.keys() & candidate.keys()):
        old, new = baseline[key], candidate[key]
        if (old.get("status") != "OK" or new.get("status") != "OK" or
                not truth(old.get("correct")) or not truth(new.get("correct"))):
            continue
        old_us = float(old["latency_median_us"])
        new_us = float(new["latency_median_us"])
        pairs.append({
            "device": key[0], "case_id": key[1],
            "category": new.get("category", ""),
            "strategy": new.get("strategy", ""),
            "baseline_us": old_us, "candidate_us": new_us,
            "speedup": old_us / new_us,
            "baseline_reductions": old.get("profile_reductions", ""),
            "candidate_reductions": new.get("profile_reductions", ""),
            "baseline_bisections": old.get("profile_bisections", ""),
            "candidate_bisections": new.get("profile_bisections", ""),
            "baseline_newton_accepts": old.get("profile_newton_accepts", ""),
            "candidate_newton_accepts": new.get("profile_newton_accepts", ""),
        })
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    columns = list(pairs[0]) if pairs else ["device", "case_id"]
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(pairs)
    speedups = [row["speedup"] for row in pairs]
    report = os.path.join(args.output_dir, "summary.md")
    with open(report, "w") as stream:
        stream.write(f"# {args.candidate_label} vs {args.baseline_label}\n\n")
        stream.write(f"Same-GPU correct pairs: {len(pairs)}; candidate wins: ")
        stream.write(f"{sum(x > 1 for x in speedups)}; geometric-mean speedup: ")
        stream.write(f"{geometric_mean(speedups):.3f}x; median: ")
        stream.write(f"{statistics.median(speedups) if speedups else math.nan:.3f}x.\n\n")
        stream.write(f"Verification failures: baseline={len(baseline_bad)}, ")
        stream.write(f"candidate={len(candidate_bad)}. Reference-invalid diagnostics: ")
        stream.write(f"baseline={len(baseline_diagnostics)}, ")
        stream.write(f"candidate={len(candidate_diagnostics)}.\n\n")
        stream.write("| Case | Baseline us | Candidate us | Speedup | Bisections old->new | Reductions old->new |\n")
        stream.write("|---|---:|---:|---:|---:|---:|\n")
        for row in sorted(pairs, key=lambda item: item["speedup"], reverse=True):
            stream.write(f"| {row['case_id']} | {row['baseline_us']:.3f} | ")
            stream.write(f"{row['candidate_us']:.3f} | {row['speedup']:.3f}x | ")
            stream.write(f"{row['baseline_bisections']}->{row['candidate_bisections']} | ")
            stream.write(f"{row['baseline_reductions']}->{row['candidate_reductions']} |\n")
    print(report)


if __name__ == "__main__":
    main()
