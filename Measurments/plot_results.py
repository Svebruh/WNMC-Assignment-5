# plot_results.py
# -----------------------------------------------------------
# Report-ready VLC plots from result CSVs:
#  - throughput_vs_distance_<payload>B.png        (per payload)
#  - rtt_mean_vs_distance_<payload>B.png          (per payload; error bars = CI or 1.96*sd/sqrt(n))
#  - rtt_std_vs_distance_<payload>B.png           (per payload)
#  - throughput_vs_distance_ALL.png               (all payloads, all distances found)
#  - rtt_mean_vs_distance_ALL.png                 (all payloads, with CI error bars)
#  - rtt_std_vs_distance_ALL.png                  (all payloads)
#  - rtt_boxplot_ALL.png                          (summary boxplot across distances per payload)
#  - summary_metrics.csv
#
# Expects files named: "<distance>cm-<payload>B-results.csv"
# CSV format:
#   Line 1: success_rate, throughput, mean_delay_s, std_delay_s, cl, cr
#   Line 2+: per-packet RTTs (seconds). NaNs allowed (loss).
# -----------------------------------------------------------

import os
import re
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ CONFIG (edit if needed) ------------------
RESULTS_DIRS = ["./results", "."]   # directories to scan (recursive)
PAYLOADS_B   = [1, 100, 180]        # payload sizes you used
OUT_DIR      = Path("./plots")      # output directory
CI_FALLBACK_Z = 1.96                # 95% CI fallback when cl/cr not provided
# -------------------------------------------------------------


# --------- helpers ---------
def find_result_files() -> List[Path]:
    pat = re.compile(r"^\s*(\d+)\s*cm-(\d+)\s*B-results\.csv$", re.IGNORECASE)
    files: List[Path] = []
    for root in RESULTS_DIRS:
        p = Path(root)
        if not p.exists():
            continue
        for f in p.rglob("*-results.csv"):
            if pat.match(f.name):
                files.append(f.resolve())
    # unique + stable sort by payload then distance
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


def to_float(s: str) -> float:
    try:
        v = float(s)
        return v
    except Exception:
        return float("nan")


def parse_csv(path: Path):
    """
    First line: success_rate, throughput, mean_delay_s, std_delay_s, cl, cr
    Subsequent lines: RTT samples (seconds), may include 'nan' for lost packets.
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    first_line = ",".join(rows[0])
    parts = [x.strip() for x in first_line.split(",")]
    if len(parts) < 6:
        raise ValueError(f"Header row has fewer than 6 fields: {path}")

    success_rate   = to_float(parts[0])
    throughput     = to_float(parts[1])
    mean_delay_s   = to_float(parts[2])
    std_delay_s    = to_float(parts[3])
    cl             = to_float(parts[4])
    cr             = to_float(parts[5])

    delays: List[float] = []
    for r in rows[1:]:
        for cell in r:
            cell = cell.strip()
            if cell == "":
                continue
            v = to_float(cell)
            # keep all numbers incl. NaN; we'll filter NaN/inf for stats
            delays.append(v)

    return {
        "success_rate": success_rate,
        "throughput": throughput,
        "mean_delay_s": mean_delay_s,
        "std_delay_s": std_delay_s,
        "cl": cl,
        "cr": cr,
        "delays": delays,
    }


def valid_numbers(vals: List[float]) -> List[float]:
    out = []
    for v in vals:
        if v == v and math.isfinite(v):  # filters NaN and +/-inf
            out.append(v)
    return out


def ci_half_width_from_delays(delays_s: List[float], z=CI_FALLBACK_Z) -> float:
    """
    Return 95% CI half-width (ms) using normal approx (z * sd / sqrt(n)).
    """
    xs = valid_numbers(delays_s)
    n = len(xs)
    if n <= 1:
        return 0.0
    arr = np.array(xs, dtype=float)
    sd = float(np.std(arr, ddof=1))
    hw = z * sd / math.sqrt(n)
    return hw * 1000.0  # ms


def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def ecdf(values: List[float]):
    xs = np.sort(np.array(values, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


# --------- main pipeline ---------
def main():
    ensure_outdir()
    files = find_result_files()

    rows_summary: List[dict] = []
    by_payload: Dict[int, List[dict]] = {p: [] for p in PAYLOADS_B}
    distances_all: Set[int] = set()

    for file in files:
        dist_cm, payload_B = parse_filename(file)
        if payload_B not in PAYLOADS_B:
            continue
        data = parse_csv(file)

        distances_all.add(dist_cm)

        # Convert to ms for plotting
        mean_ms = data["mean_delay_s"] * 1000.0 if not math.isnan(data["mean_delay_s"]) else float("nan")
        std_ms  = data["std_delay_s"]  * 1000.0 if not math.isnan(data["std_delay_s"])  else float("nan")
        cl_ms   = data["cl"]           * 1000.0 if not math.isnan(data["cl"])           else float("nan")
        cr_ms   = data["cr"]           * 1000.0 if not math.isnan(data["cr"])           else float("nan")

        # CI half-width for RTT mean: prefer (cr - mean), else compute from delays (ignoring NaNs)
        if not math.isnan(mean_ms) and not math.isnan(cr_ms):
            ci_hw_ms = max(0.0, cr_ms - mean_ms)
        else:
            ci_hw_ms = ci_half_width_from_delays(data["delays"]) if data["delays"] else 0.0

        # Std dev from header (ms); if missing, compute from raw
        if math.isnan(std_ms) and data["delays"]:
            xs = [d * 1000.0 for d in valid_numbers(data["delays"])]
            if len(xs) > 1:
                std_ms = float(np.std(np.array(xs), ddof=1))
            elif len(xs) == 1:
                std_ms = 0.0

        row = {
            "file": str(file),
            "distance_cm": dist_cm,
            "payload_B": payload_B,
            "success_rate": data["success_rate"],
            "throughput": data["throughput"],
            "mean_delay_ms": mean_ms,
            "std_delay_ms": std_ms,
            "ci_half_width_ms": ci_hw_ms,
            "n_delays": len(valid_numbers(data["delays"])),
            "delays_ms_list": [d * 1000.0 for d in valid_numbers(data["delays"])],
        }
        rows_summary.append(row)
        by_payload[payload_B].append(row)

    # Distances used in plots = every distance we actually found (this includes 8 cm with throughput 0)
    DISTANCES = sorted(distances_all)

    # Write summary CSV
    df = pd.DataFrame(rows_summary).sort_values(["payload_B", "distance_cm"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    # ---------- Per-payload plots ----------
    for payload_B in PAYLOADS_B:
        rows = sorted(by_payload[payload_B], key=lambda r: r["distance_cm"])
        if not rows:
            continue

        # Build aligned series for all observed distances
        idx = {r["distance_cm"]: r for r in rows}
        xs = DISTANCES
        thr = [idx[d]["throughput"] if d in idx else np.nan for d in xs]
        mean = [idx[d]["mean_delay_ms"] if d in idx else np.nan for d in xs]
        stdv = [idx[d]["std_delay_ms"] if d in idx else np.nan for d in xs]
        ci   = [idx[d]["ci_half_width_ms"] if d in idx else 0.0 for d in xs]

        # Throughput vs distance (per payload)
        plt.figure()
        plt.plot(xs, thr, marker="o")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Throughput")
        plt.title(f"Throughput vs Distance — payload {payload_B} B")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"throughput_vs_distance_{payload_B}B.png", dpi=150)
        plt.close()

        # Mean RTT vs distance with error bars (per payload)
        plt.figure()
        plt.errorbar(xs, mean, yerr=ci, fmt="o-", capsize=3)
        plt.xlabel("Distance (cm)")
        plt.ylabel("Mean RTT (ms)")
        plt.title(f"Mean RTT vs Distance — payload {payload_B} B")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"rtt_mean_vs_distance_{payload_B}B.png", dpi=150)
        plt.close()

        # Std RTT vs distance (per payload)
        plt.figure()
        plt.plot(xs, stdv, marker="o")
        plt.xlabel("Distance (cm)")
        plt.ylabel("RTT Std Dev (ms)")
        plt.title(f"RTT Standard Deviation vs Distance — payload {payload_B} B")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"rtt_std_vs_distance_{payload_B}B.png", dpi=150)
        plt.close()

    # ---------- Combined plots (all payloads) ----------
    def series(payload_B: int, key: str):
        rows = {r["distance_cm"]: r for r in by_payload[payload_B]}
        y = [rows[d][key] if d in rows else np.nan for d in DISTANCES]
        return DISTANCES, y

    # Combined Throughput (include zeros and gaps)
    plt.figure()
    for payload_B in PAYLOADS_B:
        x, y = series(payload_B, "throughput")
        plt.plot(x, y, marker="o", label=f"{payload_B} B")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Throughput")
    plt.title("Throughput vs Distance — all payloads")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "throughput_vs_distance_ALL.png", dpi=150)
    plt.close()

    # Combined Mean RTT with CI error bars
    plt.figure()
    for payload_B in PAYLOADS_B:
        rows = {r["distance_cm"]: r for r in by_payload[payload_B]}
        y  = [rows[d]["mean_delay_ms"] if d in rows else np.nan for d in DISTANCES]
        ye = [rows[d]["ci_half_width_ms"] if d in rows else 0.0 for d in DISTANCES]
        plt.errorbar(DISTANCES, y, yerr=ye, fmt="o-", capsize=3, label=f"{payload_B} B")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Mean RTT (ms)")
    plt.title("Mean RTT vs Distance — all payloads (with CI)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rtt_mean_vs_distance_ALL.png", dpi=150)
    plt.close()

    # Combined Std RTT
    plt.figure()
    for payload_B in PAYLOADS_B:
        x, y = series(payload_B, "std_delay_ms")
        plt.plot(x, y, marker="o", label=f"{payload_B} B")
    plt.xlabel("Distance (cm)")
    plt.ylabel("RTT Std Dev (ms)")
    plt.title("RTT Standard Deviation vs Distance — all payloads")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rtt_std_vs_distance_ALL.png", dpi=150)
    plt.close()

    # ---------- Concise distribution view (boxplot) ----------
    # One figure: three groups (payloads). Each group shows RTT distributions across distances (labels like '1B@0','1B@1',...)
    labels = []
    data   = []
    for payload_B in PAYLOADS_B:
        for d in DISTANCES:
            recs = [r for r in by_payload[payload_B] if r["distance_cm"] == d]
            if not recs:
                continue
            # concatenate RTTs (ms) for that (payload, distance)
            samples = []
            for r in recs:
                samples.extend(r["delays_ms_list"])
            if len(samples) == 0:
                continue
            labels.append(f"{payload_B}B@{d}")
            data.append(samples)

    if data:
        plt.figure(figsize=(max(8, len(labels) * 0.6), 5))
        plt.boxplot(data, showmeans=True)
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
        plt.ylabel("RTT (ms)")
        plt.title("RTT distribution summary (boxplot)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "rtt_boxplot_ALL.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
