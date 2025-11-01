# plot_results.py
# -----------------------------------------------------------
# Generate VLC assignment plots from result CSVs.
# Expects files named like: "<distance>cm-<payload>B-results.csv"
# Each CSV:
#   Line 1: success_rate, throughput, mean_delay_s, std_delay_s, cl, cr
#   Line 2+: per-packet delays (seconds)
#
# Outputs in ./plots:
#  - throughput_vs_distance_<payload>B.png         (per payload)
#  - rtt_vs_distance_<payload>B.png                (per payload; CI if given, else 1.96*sd/sqrt(n))
#  - delay_cdf_<payload>B_<distance>cm.png         (per payload & distance)
#  - throughput_vs_distance_ALL.png                (1B,100B,180B in one figure)
#  - rtt_vs_distance_ALL.png                       (1B,100B,180B in one figure)
#  - summary_metrics.csv
#
# Usage:
#   Place all CSVs in ./results next to this script, then run:
#       python plot_results.py
# -----------------------------------------------------------

import os
import re
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ CONFIG (edit if needed) ------------------
RESULTS_DIRS = ["./results", "."]   # searched in order
DISTANCES_CM = [0, 1, 3, 5, 6, 7, 10]
PAYLOADS_B   = [1, 100, 180]
OUT_DIR      = Path("./plots")
# -------------------------------------------------------------


def find_result_files() -> List[Path]:
    """Find files matching '<d>cm-<p>B-results.csv' in RESULTS_DIRS."""
    files: List[Path] = []
    pat = re.compile(r"^\s*(\d+)\s*cm-(\d+)\s*B-results\.csv$", re.IGNORECASE)
    for root in RESULTS_DIRS:
        p = Path(root)
        if not p.exists():
            continue
        for f in p.glob("**/*-results.csv"):
            if pat.match(f.name):
                files.append(f.resolve())
    # Deduplicate & sort by payload, then distance
    files = sorted(
        set(files),
        key=lambda x: (
            int(re.search(r"(\d+)\s*B-", x.name, re.I).group(1)),
            int(re.search(r"(\d+)\s*cm-", x.name, re.I).group(1)),
        ),
    )
    return files


def parse_filename(path: Path) -> Tuple[int, int]:
    m1 = re.search(r"(\d+)\s*cm-", path.name, re.IGNORECASE)
    m2 = re.search(r"(\d+)\s*B-", path.name, re.IGNORECASE)
    if not (m1 and m2):
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(m1.group(1)), int(m2.group(1))


def parse_csv(path: Path):
    """
    First line: success_rate, throughput, mean_delay_s, std_delay_s, cl, cr
    Subsequent lines: delays (seconds), any number per line.
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    # Robust parse of first line
    first_line = ",".join(rows[0])
    parts = [x.strip() for x in first_line.split(",")]
    if len(parts) < 6:
        raise ValueError(f"Header row has fewer than 6 fields: {path}")

    def to_float(s: str) -> float:
        try:
            return float(s)
        except:
            return float("nan")

    success_rate   = to_float(parts[0])
    throughput     = to_float(parts[1])
    mean_delay_s   = to_float(parts[2])
    std_delay_s    = to_float(parts[3])
    cl             = to_float(parts[4])
    cr             = to_float(parts[5])

    delays: List[float] = []
    if len(rows) > 1:
        for r in rows[1:]:
            for cell in r:
                cell = cell.strip()
                if cell != "":
                    try:
                        delays.append(float(cell))
                    except:
                        pass

    return {
        "success_rate": success_rate,
        "throughput": throughput,
        "mean_delay_s": mean_delay_s,
        "std_delay_s": std_delay_s,
        "cl": cl,
        "cr": cr,
        "delays": delays,
    }


def ci_half_width_from_delays(delays_s: List[float]) -> float:
    """Return 95% CI half-width (ms) using normal approx (1.96*sd/sqrt(n))."""
    n = len(delays_s)
    if n <= 1:
        return 0.0
    arr = np.array(delays_s, dtype=float)
    sd = float(np.std(arr, ddof=1))  # sample std
    hw = 1.96 * sd / math.sqrt(n)
    return hw * 1000.0  # ms


def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def ecdf(values: List[float]):
    if len(values) == 0:
        return np.array([]), np.array([])
    xs = np.sort(np.array(values, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def main():
    ensure_outdir()
    files = find_result_files()

    # Load all
    rows_summary: List[dict] = []
    by_payload: Dict[int, List[dict]] = {p: [] for p in PAYLOADS_B}

    for file in files:
        dist_cm, payload_B = parse_filename(file)
        if (dist_cm not in DISTANCES_CM) or (payload_B not in PAYLOADS_B):
            continue
        data = parse_csv(file)

        # Convert to ms for plotting
        mean_ms = data["mean_delay_s"] * 1000.0 if not math.isnan(data["mean_delay_s"]) else float("nan")
        std_ms  = data["std_delay_s"]  * 1000.0 if not math.isnan(data["std_delay_s"])  else float("nan")
        cl_ms   = data["cl"]           * 1000.0 if not math.isnan(data["cl"])           else float("nan")
        cr_ms   = data["cr"]           * 1000.0 if not math.isnan(data["cr"])           else float("nan")

        # CI half-width: prefer (cr - mean), else compute from raw delays
        if not math.isnan(cl_ms) and not math.isnan(cr_ms) and not math.isnan(mean_ms):
            ci_hw_ms = max(0.0, cr_ms - mean_ms)
        else:
            ci_hw_ms = ci_half_width_from_delays(data["delays"]) if data["delays"] else 0.0

        row = {
            "file": str(file),
            "distance_cm": dist_cm,
            "payload_B": payload_B,
            "success_rate": data["success_rate"],
            "throughput": data["throughput"],
            "mean_delay_ms": mean_ms,
            "std_delay_ms": std_ms,
            "ci_half_width_ms": ci_hw_ms,
            "n_delays": len(data["delays"]),
        }
        rows_summary.append(row)
        by_payload[payload_B].append({**row, "delays_s": data["delays"]})

    # Write summary CSV
    df = pd.DataFrame(rows_summary).sort_values(["payload_B", "distance_cm"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    # ---------- Per-payload plots ----------
    # 1) Throughput vs Distance (per payload)
    for payload_B in PAYLOADS_B:
        rows = sorted(by_payload[payload_B], key=lambda r: r["distance_cm"])
        if not rows:
            continue
        x = [r["distance_cm"] for r in rows]
        y = [r["throughput"] for r in rows]
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Throughput")
        plt.title(f"Throughput vs Distance — payload {payload_B} B")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"throughput_vs_distance_{payload_B}B.png", dpi=150)
        plt.close()

    # 2) Mean RTT vs Distance with error bars (per payload)
    for payload_B in PAYLOADS_B:
        rows = sorted(by_payload[payload_B], key=lambda r: r["distance_cm"])
        if not rows:
            continue
        x = [r["distance_cm"] for r in rows]
        y = [r["mean_delay_ms"] for r in rows]
        yerr = []
        for r in rows:
            hw = r["ci_half_width_ms"]
            if (math.isnan(hw) or hw == 0.0) and r["n_delays"] > 1 and not math.isnan(r["std_delay_ms"]):
                hw = 1.96 * (r["std_delay_ms"] / math.sqrt(r["n_delays"]))
            if math.isnan(hw):
                hw = 0.0
            yerr.append(hw)
        plt.figure()
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
        plt.xlabel("Distance (cm)")
        plt.ylabel("Mean RTT (ms)")
        plt.title(f"Mean RTT vs Distance — payload {payload_B} B")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"rtt_vs_distance_{payload_B}B.png", dpi=150)
        plt.close()

    # 3) Delay CDFs for each (payload, distance)
    for payload_B in PAYLOADS_B:
        rows = sorted(by_payload[payload_B], key=lambda r: r["distance_cm"])
        for r in rows:
            delays_ms = [d * 1000.0 for d in r.get("delays_s", [])]
            if len(delays_ms) == 0:
                continue
            xs, ys = ecdf(delays_ms)
            plt.figure()
            plt.plot(xs, ys)
            plt.xlabel("RTT (ms)")
            plt.ylabel("CDF")
            plt.title(f"Delay CDF — payload {payload_B} B, {r['distance_cm']} cm")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"delay_cdf_{payload_B}B_{r['distance_cm']}cm.png", dpi=150)
            plt.close()

    # ---------- Combined plots (all payloads on one figure) ----------
    # Helper: build vectors aligned to DISTANCES_CM (NaN for missing points)
    def series_for_payload(metric_key: str, payload_B: int):
        rows = {r["distance_cm"]: r for r in by_payload[payload_B]}
        xs = []
        ys = []
        for d in DISTANCES_CM:
            xs.append(d)
            val = rows[d][metric_key] if d in rows else float("nan")
            ys.append(val)
        return xs, ys

    # Combined Throughput
    plt.figure()
    for payload_B in PAYLOADS_B:
        x, y = series_for_payload("throughput", payload_B)
        plt.plot(x, y, marker="o", label=f"{payload_B} B")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Throughput")
    plt.title("Throughput vs Distance — all payloads")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "throughput_vs_distance_ALL.png", dpi=150)
    plt.close()

    # Combined Mean RTT
    plt.figure()
    for payload_B in PAYLOADS_B:
        x, y = series_for_payload("mean_delay_ms", payload_B)
        plt.plot(x, y, marker="o", label=f"{payload_B} B")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Mean RTT (ms)")
    plt.title("Mean RTT vs Distance — all payloads")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rtt_vs_distance_ALL.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
