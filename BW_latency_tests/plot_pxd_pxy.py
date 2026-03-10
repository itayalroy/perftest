#!/usr/bin/env python3
"""
Plot RDMA BW and Latency vs transport buffer from merged result files.
- Figure 1 (PxDx): source=target — direct, allow_nvlink_reassembly
- Figure 2 (PxDy): source<=target — direct_alltoall, allow_nvlink, allow_nvlink_reassembly_alltoall

Sources: 1 GPU (0), 2 GPUs (0,1), 8 GPUs (0,1,2,3,4,5,6,7)
X-axis: transport buffer (8M, 16M, 32M, 64M, 128M, 256M, 1G)
Note: full message size = 1 GB
"""

import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Result file paths
RESULT_FILES = [
    "BW_latency_tests/9905078_2026-03-10_02-50-45_subset_transport_double_buffer_clc_1GB_pool0-01413_pool0-01436_/rdma_tests_results_9905078_2026-03-10_02-50-45_subset_transport_double_buffer_clc_1GB_pool0-01413_pool0-01436_.txt",
    "BW_latency_tests/9795714_2026-03-04_06-10-49_transport_buffer_clc_1GB_pool0-00677_pool0-01577_/rdma_tests_results_9795714_2026-03-04_06-10-49_transport_buffer_clc_1GB_pool0-00677_pool0-01577_.txt",
    "BW_latency_tests/9801670_2026-03-04_12-22-30_transport_buffer_clc_1GB_pool0-01635_pool0-01648_/rdma_tests_results_9801670_2026-03-04_12-22-30_transport_buffer_clc_1GB_pool0-01635_pool0-01648_.txt",
]

TRANSPORT_BUFFERS = ["8M", "16M", "32M", "64M", "128M", "256M", "1G"]
TB_TO_MB = {"8M": 8, "16M": 16, "32M": 32, "64M": 64, "128M": 128, "256M": 256, "1G": 1024}


def parse_float(s):
    s = str(s).strip().replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_subset_format(path):
    """Parse file 1 format (pipe-separated, config | col1 | col2 | ...)."""
    data = {"bw": {}, "lat": {}}
    content = Path(path).read_text()
    lines = content.split("\n")

    # Find header and parse column indices
    in_bw, in_lat = False, False
    hdr_line, cols = None, []

    for i, line in enumerate(lines):
        if "=== Bandwidth" in line:
            in_bw, in_lat = True, False
            continue
        if "=== Latency" in line:
            in_bw, in_lat = False, True
            continue
        if "All values" in line:
            in_bw, in_lat = False, False
            continue

        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue

        if in_bw or in_lat:
            if parts[0] == "config" or parts[0].startswith("-"):
                if parts[0] == "config":
                    cols = parts[1:]
                continue
            cfg = parts[0].strip()
            if not cfg.startswith("src"):
                continue
            kind = "bw" if in_bw else "lat"
            for j, v in enumerate(parts[1:]):
                if j < len(cols) and cols[j] and cols[j] != "-":
                    val = parse_float(v)
                    if val is not None:
                        key = (cfg, cols[j])
                        if key not in data[kind]:
                            data[kind][key] = []
                        data[kind][key].append(val)

    return data


def parse_markdown_format(path):
    """Parse file 2/3 format (markdown tables)."""
    data = {"bw": {}, "lat": {}}
    content = Path(path).read_text()
    # Split by ## sections
    sections = re.split(r"\n## ", content)
    for sec in sections:
        if "Bandwidth" in sec:
            kind = "bw"
        elif "Latency" in sec:
            kind = "lat"
        else:
            continue

        lines = [l.strip() for l in sec.split("\n") if l.strip()]
        hdr = None
        for line in lines:
            if line.startswith("|") and "source_gpus" in line:
                hdr = [c.strip() for c in line.split("|")[1:-1]]
                continue
            if line.startswith("|") and "---" not in line and hdr:
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) < 2:
                    continue
                src = cells[0]
                n_src = len(src.split(","))
                for j, col in enumerate(hdr[1:], 1):
                    if j >= len(cells):
                        break
                    val = parse_float(cells[j])
                    if val is not None and col:
                        key = (f"src{src.replace(',', '_')}", col)
                        if key not in data[kind]:
                            data[kind][key] = []
                        data[kind][key].append(val)

    return data


def detect_format(path):
    with open(path) as f:
        first = f.read(200)
    return "subset" if "config" in first and "|" in first and "nvlink_" in first else "markdown"


def load_all(base_dir):
    """Load and merge data from all result files."""
    merged_bw = defaultdict(list)
    merged_lat = defaultdict(list)

    for rel_path in RESULT_FILES:
        path = Path(base_dir) / rel_path
        if not path.exists():
            print(f"Warning: {path} not found")
            continue
        fmt = detect_format(path)
        if fmt == "subset":
            d = parse_subset_format(path)
        else:
            d = parse_markdown_format(path)
        for k, vals in d["bw"].items():
            for v in vals:
                merged_bw[k].append(v)
        for k, vals in d["lat"].items():
            for v in vals:
                merged_lat[k].append(v)

    # Average merged values
    def avg(vals):
        return sum(vals) / len(vals) if vals else None

    return (
        {k: avg(v) for k, v in merged_bw.items() if v},
        {k: avg(v) for k, v in merged_lat.items() if v},
    )


# Column name mappings
# PxDx: direct, allow_nvlink_reassembly (no alltoall)
# PxDy: direct_alltoall, allow_nvlink, allow_nvlink_reassembly_alltoall

# Subset format: config -> (n_src, n_tgt)
SUBSET_CONFIG = {
    "src0_tc8": (1, 8),
    "src0_1_tc8": (2, 8),
    "src0_tc4": (1, 4),
    "src0_1_tc4": (2, 4),
    "src0_1_2_3_4_5_6_7": (8, 8),
}

# Markdown: source_gpus -> n_src (assume D8 for allow_nvlink, Dx for direct where x=src)
MD_SRC_TO_N = {"0": 1, "0,1": 2, "0,1,2": 3, "0,1,2,3": 4, "0,1,2,3,4": 5, "0,1,2,3,4,5": 6, "0,1,2,3,4,5,6": 7, "0,1,2,3,4,5,6,7": 8}


def get_value(data, cfg_key, col_patterns):
    """Get value from data, trying multiple column patterns."""
    for pat in col_patterns:
        for k, v in data.items():
            if k[0] == cfg_key and pat in k[1]:
                return v
    return None


def extract_pxdx_data(merged_bw, merged_lat):
    """PxDx: source=target. direct + allow_nvlink_reassembly (no alltoall). From markdown files."""
    # Markdown keys: src0, src0_1, src0_1_2_3_4_5_6_7
    cfg_keys = {"P1D1": "src0", "P2D2": "src0_1", "P8D8": "src0_1_2_3_4_5_6_7"}

    bw = defaultdict(dict)
    lat = defaultdict(dict)

    for label in ["P1D1", "P2D2", "P8D8"]:
        cfg = cfg_keys[label]
        # direct (no transport buffer)
        for k, v in merged_bw.items():
            if k[0] == cfg and k[1] == "direct":
                bw[label]["direct"] = v
                break
        for k, v in merged_lat.items():
            if k[0] == cfg and k[1] == "direct":
                lat[label]["direct"] = v
                break

        # allow_nvlink_reassembly (not alltoall) by transport buffer
        for tb in ["8M", "16M", "32M", "64M", "128M", "1G"]:
            col = f"allow_nvlink_reassembly_{tb}" if tb != "1G" else "allow_nvlink_reassembly_full"
            for k, v in merged_bw.items():
                if k[0] == cfg and k[1] == col:
                    bw[label][tb] = v
                    break
            for k, v in merged_lat.items():
                if k[0] == cfg and k[1] == col:
                    lat[label][tb] = v
                    break

    return bw, lat


def extract_pxdy_data(merged_bw, merged_lat):
    """PxDy: source<=target. direct_alltoall, allow_nvlink, allow_nvlink_reassembly_alltoall."""
    # Subset format (file 1): src0_tc4, src0_tc8, src0_1_tc4, src0_1_tc8, src0_1_2_3_4_5_6_7
    # Markdown format (files 2,3): src0, src0_1, src0_1_2_3_4_5_6_7 (all PxD8)
    cfg_sources = {
        "P1D4": ["src0_tc4"],
        "P1D8": ["src0_tc8", "src0"],
        "P2D4": ["src0_1_tc4"],
        "P2D8": ["src0_1_tc8", "src0_1"],
        "P8D8": ["src0_1_2_3_4_5_6_7"],
    }

    def get_vals(data, cfgs, col_match):
        vals = []
        for cfg in cfgs:
            for k, v in data.items():
                if k[0] == cfg and col_match(k[1]):
                    vals.append(v)
                    break
        return vals

    bw = defaultdict(dict)
    lat = defaultdict(dict)

    for label in ["P1D4", "P1D8", "P2D4", "P2D8", "P8D8"]:
        cfgs = cfg_sources[label]

        # direct_alltoall
        vb = get_vals(merged_bw, cfgs, lambda c: c == "direct_alltoall")
        vl = get_vals(merged_lat, cfgs, lambda c: c == "direct_alltoall")
        if vb:
            bw[label]["direct_alltoall"] = sum(vb) / len(vb)
        if vl:
            lat[label]["direct_alltoall"] = sum(vl) / len(vl)

        for tb in TRANSPORT_BUFFERS:
            # allow_nvlink: subset has nvlink_XM_dbl, nvlink_XM_nodbl; markdown has allow_nvlink_XM
            col_tb = tb if tb != "1G" else "full"

            def match_nvlink(c):
                if "nvlink_" in c and tb in c and "ra_" not in c:
                    return True
                return c == f"allow_nvlink_{col_tb}"

            vb = get_vals(merged_bw, cfgs, match_nvlink)
            vl = get_vals(merged_lat, cfgs, match_nvlink)
            if vb:
                bw[label][f"allow_nvlink_{tb}"] = sum(vb) / len(vb)
            if vl:
                lat[label][f"allow_nvlink_{tb}"] = sum(vl) / len(vl)

            # allow_nvlink_reassembly_alltoall
            def match_ra_alltoall(c):
                if f"nvlink_ra_" in c and tb in c:
                    return True
                return c == f"allow_nvlink_reassembly_alltoall_{col_tb}"

            vb = get_vals(merged_bw, cfgs, match_ra_alltoall)
            vl = get_vals(merged_lat, cfgs, match_ra_alltoall)
            if vb:
                bw[label][f"allow_nvlink_ra_alltoall_{tb}"] = sum(vb) / len(vb)
            if vl:
                lat[label][f"allow_nvlink_ra_alltoall_{tb}"] = sum(vl) / len(vl)

    return bw, lat


def build_pxdx_series(bw, lat):
    """Build series for PxDx: x = [direct, 8M, 16M, ..., 1G], one series per config."""
    x_labels = ["direct", "8M", "16M", "32M", "64M", "128M", "1G"]

    series_bw = {}
    series_lat = {}
    for label in ["P1D1", "P2D2", "P8D8"]:
        ys_bw, ys_lat = [], []
        for x in x_labels:
            if x == "direct":
                v_bw = bw.get(label, {}).get("direct")
                v_lat = lat.get(label, {}).get("direct")
            else:
                v_bw = bw.get(label, {}).get(x)
                v_lat = lat.get(label, {}).get(x)
            ys_bw.append(v_bw)
            ys_lat.append(v_lat)
        series_bw[label] = ys_bw
        series_lat[label] = ys_lat

    return x_labels, series_bw, series_lat




def main():
    base = Path(__file__).resolve().parent.parent
    merged_bw, merged_lat = load_all(base)

    # Debug: print what we loaded
    # print("BW keys sample:", list(merged_bw.keys())[:10])
    # print("LAT keys sample:", list(merged_lat.keys())[:10])

    colors = {"P1D1": "tab:blue", "P2D2": "tab:orange", "P8D8": "tab:green",
              "P1D4": "tab:blue", "P1D8": "tab:cyan", "P2D4": "tab:orange", "P2D8": "tab:red", "P8D8": "tab:green"}

    # --- Figure 1: PxDx ---
    bw_pxdx, lat_pxdx = extract_pxdx_data(merged_bw, merged_lat)
    x_labels, series_bw, series_lat = build_pxdx_series(bw_pxdx, lat_pxdx)
    x_pos = list(range(len(x_labels)))

    fig1, (ax1_bw, ax1_lat) = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("PxDx (source = target) — direct, allow_nvlink_reassembly (full = 1 GB)")

    for label in ["P1D1", "P2D2", "P8D8"]:
        ys = [y for y in series_bw[label] if y is not None]
        xs = [x_pos[i] for i in range(len(series_bw[label])) if series_bw[label][i] is not None]
        if xs and ys:
            ax1_bw.plot(xs, ys, "o-", color=colors[label], label=label)
    ax1_bw.set_xticks(x_pos)
    ax1_bw.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1_bw.set_ylabel("Bandwidth (GB/s)")
    ax1_bw.set_xlabel("Transport buffer")
    ax1_bw.legend()
    ax1_bw.grid(True, alpha=0.3)

    for label in ["P1D1", "P2D2", "P8D8"]:
        ys = [y for y in series_lat[label] if y is not None]
        xs = [x_pos[i] for i in range(len(series_lat[label])) if series_lat[label][i] is not None]
        if xs and ys:
            ax1_lat.plot(xs, ys, "o-", color=colors[label], label=label)
    ax1_lat.set_xticks(x_pos)
    ax1_lat.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1_lat.set_ylabel("Latency (ms/iteration)")
    ax1_lat.set_xlabel("Transport buffer")
    ax1_lat.legend()
    ax1_lat.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(base / "BW_latency_tests" / "plot_pxdx.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved plot_pxdx.png")

    # --- Figure 2: PxDy ---
    bw_pxdy, lat_pxdy = extract_pxdy_data(merged_bw, merged_lat)
    x_labels = ["direct"] + list(TRANSPORT_BUFFERS)
    x_pos = list(range(len(x_labels)))

    fig2, (ax2_bw, ax2_lat) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("PxDy (source ≤ target) — direct_alltoall, allow_nvlink, allow_nvlink_ra_alltoall (full = 1 GB)")

    for cfg in ["P1D4", "P1D8", "P2D4", "P2D8", "P8D8"]:
        vals_bw = []
        v_direct = bw_pxdy.get(cfg, {}).get("direct_alltoall")
        vals_bw.append(v_direct)
        for tb in TRANSPORT_BUFFERS:
            vn = bw_pxdy.get(cfg, {}).get(f"allow_nvlink_{tb}")
            vr = bw_pxdy.get(cfg, {}).get(f"allow_nvlink_ra_alltoall_{tb}")
            vs = [x for x in (vn, vr) if x is not None]
            vals_bw.append(sum(vs) / len(vs) if vs else None)
        xs = [i for i, v in enumerate(vals_bw) if v is not None]
        ys = [v for v in vals_bw if v is not None]
        if xs and ys:
            ax2_bw.plot(xs, ys, "o-", color=colors.get(cfg, "gray"), label=cfg)

    ax2_bw.set_xticks(x_pos)
    ax2_bw.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2_bw.set_ylabel("Bandwidth (GB/s)")
    ax2_bw.set_xlabel("Transport buffer")
    ax2_bw.legend()
    ax2_bw.grid(True, alpha=0.3)

    for cfg in ["P1D4", "P1D8", "P2D4", "P2D8", "P8D8"]:
        vals_lat = []
        v_direct = lat_pxdy.get(cfg, {}).get("direct_alltoall")
        vals_lat.append(v_direct)
        for tb in TRANSPORT_BUFFERS:
            vn = lat_pxdy.get(cfg, {}).get(f"allow_nvlink_{tb}")
            vr = lat_pxdy.get(cfg, {}).get(f"allow_nvlink_ra_alltoall_{tb}")
            vs = [x for x in (vn, vr) if x is not None]
            vals_lat.append(sum(vs) / len(vs) if vs else None)
        xs = [i for i, v in enumerate(vals_lat) if v is not None]
        ys = [v for v in vals_lat if v is not None]
        if xs and ys:
            ax2_lat.plot(xs, ys, "o-", color=colors.get(cfg, "gray"), label=cfg)

    ax2_lat.set_xticks(x_pos)
    ax2_lat.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2_lat.set_ylabel("Latency (ms/iteration)")
    ax2_lat.set_xlabel("Transport buffer")
    ax2_lat.legend()
    ax2_lat.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(base / "BW_latency_tests" / "plot_pxdy.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved plot_pxdy.png")


if __name__ == "__main__":
    main()
