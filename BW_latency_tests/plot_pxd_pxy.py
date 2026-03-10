#!/usr/bin/env python3
"""
Plot RDMA BW and Latency vs transport buffer from merged result files.
- PxDx (source=target): One figure per P1D1, P2D2, P8D8. Lines: direct, PXN + reassembly.
- PxDy (source<=target): One figure per P1D4, P1D8, P2D4, P2D8, P8D8. Lines: direct --all-to-all, PXN, PXN + reassembly.

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


def build_pxdx_series_for_label(bw, lat, label):
    """Build series for one PxDx config: direct (1 pt) + allow_nvlink_reassembly (per tb)."""
    x_labels = ["direct", "8M", "16M", "32M", "64M", "128M", "1G"]
    direct_bw = bw.get(label, {}).get("direct")
    direct_lat = lat.get(label, {}).get("direct")
    ra_bw = [bw.get(label, {}).get(tb) for tb in x_labels[1:]]
    ra_lat = [lat.get(label, {}).get(tb) for tb in x_labels[1:]]
    return x_labels, direct_bw, direct_lat, ra_bw, ra_lat




def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "BW_latency_tests"
    merged_bw, merged_lat = load_all(base)

    mode_colors = {"direct": "tab:blue", "PXN": "tab:green", "PXN + reassembly": "tab:orange"}
    mode_colors_pxdx = {"direct": "tab:blue", "PXN + reassembly": "tab:orange"}

    x_labels_pxdx = ["direct", "8M", "16M", "32M", "64M", "128M", "1G"]
    x_labels_pxdy = ["direct --all-to-all"] + list(TRANSPORT_BUFFERS)

    # --- PxDx: one figure per P1D1, P2D2, P8D8. Lines: direct, PXN + reassembly ---
    bw_pxdx, lat_pxdx = extract_pxdx_data(merged_bw, merged_lat)
    for label in ["P1D1", "P2D2", "P8D8"]:
        x_labels, direct_bw, direct_lat, ra_bw, ra_lat = build_pxdx_series_for_label(bw_pxdx, lat_pxdx, label)
        x_pos = list(range(len(x_labels)))

        fig, (ax_bw, ax_lat) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{label} (source = target) — direct, PXN + reassembly (full = 1 GB)")

        # BW: direct (single point at 0), PXN + reassembly (points 1..6)
        if direct_bw is not None:
            ax_bw.plot(0, direct_bw, "o", color=mode_colors_pxdx["direct"], label="direct")
        ra_x = [i for i, v in enumerate(ra_bw, 1) if v is not None]
        ra_y = [v for v in ra_bw if v is not None]
        if ra_x and ra_y:
            ax_bw.plot(ra_x, ra_y, "o-", color=mode_colors_pxdx["PXN + reassembly"], label="PXN + reassembly")

        ax_bw.set_xticks(x_pos)
        ax_bw.set_xticklabels(x_labels, rotation=45, ha="right")
        ax_bw.set_ylabel("Bandwidth (GB/s)")
        ax_bw.set_xlabel("Transport buffer")
        ax_bw.legend()
        ax_bw.grid(True, alpha=0.3)

        # Latency
        if direct_lat is not None:
            ax_lat.plot(0, direct_lat, "o", color=mode_colors_pxdx["direct"], label="direct")
        ra_x = [i for i, v in enumerate(ra_lat, 1) if v is not None]
        ra_y = [v for v in ra_lat if v is not None]
        if ra_x and ra_y:
            ax_lat.plot(ra_x, ra_y, "o-", color=mode_colors_pxdx["PXN + reassembly"], label="PXN + reassembly")

        ax_lat.set_xticks(x_pos)
        ax_lat.set_xticklabels(x_labels, rotation=45, ha="right")
        ax_lat.set_ylabel("Latency (ms/iteration)")
        ax_lat.set_xlabel("Transport buffer")
        ax_lat.legend()
        ax_lat.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"plot_{label.lower()}_pxdx.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")

    # --- PxDy: one figure per P1D4, P1D8, P2D4, P2D8, P8D8. Lines: direct --all-to-all, PXN, PXN + reassembly ---
    bw_pxdy, lat_pxdy = extract_pxdy_data(merged_bw, merged_lat)
    for label in ["P1D4", "P1D8", "P2D4", "P2D8", "P8D8"]:
        x_pos = list(range(len(x_labels_pxdy)))

        fig, (ax_bw, ax_lat) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{label} (source ≤ target) — direct --all-to-all, PXN, PXN + reassembly (full = 1 GB)")

        d = bw_pxdy.get(label, {})
        # BW: direct_alltoall at 0, allow_nvlink (PXN) and allow_nvlink_ra_alltoall (PXN + reassembly) per tb
        v_direct = d.get("direct_alltoall")
        if v_direct is not None:
            ax_bw.plot(0, v_direct, "o", color=mode_colors["direct"], label="direct --all-to-all")

        pxn_x, pxn_y = [], []
        pra_x, pra_y = [], []
        for i, tb in enumerate(TRANSPORT_BUFFERS, 1):
            vn = d.get(f"allow_nvlink_{tb}")
            vr = d.get(f"allow_nvlink_ra_alltoall_{tb}")
            if vn is not None:
                pxn_x.append(i)
                pxn_y.append(vn)
            if vr is not None:
                pra_x.append(i)
                pra_y.append(vr)
        if pxn_x and pxn_y:
            ax_bw.plot(pxn_x, pxn_y, "o-", color=mode_colors["PXN"], label="PXN")
        if pra_x and pra_y:
            ax_bw.plot(pra_x, pra_y, "o-", color=mode_colors["PXN + reassembly"], label="PXN + reassembly")

        ax_bw.set_xticks(x_pos)
        ax_bw.set_xticklabels(x_labels_pxdy, rotation=45, ha="right")
        ax_bw.set_ylabel("Bandwidth (GB/s)")
        ax_bw.set_xlabel("Transport buffer")
        ax_bw.legend()
        ax_bw.grid(True, alpha=0.3)

        # Latency
        d = lat_pxdy.get(label, {})
        v_direct = d.get("direct_alltoall")
        if v_direct is not None:
            ax_lat.plot(0, v_direct, "o", color=mode_colors["direct"], label="direct --all-to-all")

        pxn_x, pxn_y = [], []
        pra_x, pra_y = [], []
        for i, tb in enumerate(TRANSPORT_BUFFERS, 1):
            vn = d.get(f"allow_nvlink_{tb}")
            vr = d.get(f"allow_nvlink_ra_alltoall_{tb}")
            if vn is not None:
                pxn_x.append(i)
                pxn_y.append(vn)
            if vr is not None:
                pra_x.append(i)
                pra_y.append(vr)
        if pxn_x and pxn_y:
            ax_lat.plot(pxn_x, pxn_y, "o-", color=mode_colors["PXN"], label="PXN")
        if pra_x and pra_y:
            ax_lat.plot(pra_x, pra_y, "o-", color=mode_colors["PXN + reassembly"], label="PXN + reassembly")

        ax_lat.set_xticks(x_pos)
        ax_lat.set_xticklabels(x_labels_pxdy, rotation=45, ha="right")
        ax_lat.set_ylabel("Latency (ms/iteration)")
        ax_lat.set_xlabel("Transport buffer")
        ax_lat.legend()
        ax_lat.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"plot_{label.lower()}_pxdy.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()
