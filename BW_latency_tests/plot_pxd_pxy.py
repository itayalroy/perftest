#!/usr/bin/env python3
"""
Plot RDMA BW and Latency vs transport buffer from merged result files.
- PxDx (source=target): One figure per P1D1, P2D2, P8D8. Lines: direct, PXN + reassembly.
- PxDy (source<=target): One figure per P1D4, P1D8, P2D4, P2D8, P8D8. Lines: direct --all-to-all, PXN, PXN + reassembly.

Sources: 1 GPU (0), 2 GPUs (0,1), 8 GPUs (0,1,2,3,4,5,6,7)
X-axis: transport buffer — auto-detected from result file (1GB: 8M..1G; 128MB: 8M..128M)

Usage:
  python plot_pxd_pxy.py [--result-files FILE1 [FILE2 ...]] [--message-size 1GB|128MB] [--output-dir DIR]
  Default: uses RESULT_FILES below. Message size inferred from path (1GB/128MB) or --message-size.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Default result file paths (1GB)
RESULT_FILES_1GB = [
    "BW_latency_tests/9940605_2026-03-12_02-52-12_subset_transport_double_buffer_clc_1GB_pool0-01707_pool0-01711_/rdma_tests_results_9940605_2026-03-12_02-52-12_subset_transport_double_buffer_clc_1GB_pool0-01707_pool0-01711_.txt",
    "BW_latency_tests/9795714_2026-03-04_06-10-49_transport_buffer_clc_1GB_pool0-00677_pool0-01577_/rdma_tests_results_9795714_2026-03-04_06-10-49_transport_buffer_clc_1GB_pool0-00677_pool0-01577_.txt",
    "BW_latency_tests/9801670_2026-03-04_12-22-30_transport_buffer_clc_1GB_pool0-01635_pool0-01648_/rdma_tests_results_9801670_2026-03-04_12-22-30_transport_buffer_clc_1GB_pool0-01635_pool0-01648_.txt",
]

# Default result file paths (128MB — no 256M, 1G transport buffers)
RESULT_FILES_128MB = [
    "BW_latency_tests/9987673_2026-03-14_12-07-31_subset_transport_double_buffer_clc_128MB_pool0-01343_pool0-01417_/rdma_tests_results_9987673_2026-03-14_12-07-31_subset_transport_double_buffer_clc_128MB_pool0-01343_pool0-01417_.txt",
]

TB_TO_MB = {"8M": 8, "16M": 16, "32M": 32, "64M": 64, "128M": 128, "256M": 256, "1G": 1024}

# Speed-of-light reference bandwidth (GB/s) for InfiniBand HDR
SPEED_OF_LIGHT_BW = 380.0


def parse_float(s):
    s = str(s).strip().replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_transport_buffers_from_cols(cols_or_keys):
    """Extract sorted transport buffer names from column names (e.g. nvlink_8M_dbl -> 8M)."""
    seen = set()
    for c in cols_or_keys:
        col = c[1] if isinstance(c, tuple) else c
        m = re.search(r"nvlink(?:_ra)?_([0-9]+[MG])(?:_dbl|_nodbl)?", col)
        if m:
            seen.add(m.group(1))
    order = ["8M", "16M", "32M", "64M", "128M", "256M", "1G"]
    return [tb for tb in order if tb in seen]


def parse_subset_format(path):
    """Parse file 1 format (pipe-separated, config | col1 | col2 | ...). Returns (data, cols)."""
    data = {"bw": {}, "lat": {}}
    content = Path(path).read_text()
    lines = content.split("\n")

    in_bw, in_lat = False, False
    cols = []

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

    return data, cols


def parse_markdown_format(path):
    """Parse file 2/3 format (markdown tables). Returns (data, cols)."""
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

    cols = list(hdr[1:]) if hdr else []
    return data, cols


def detect_format(path):
    with open(path) as f:
        first = f.read(200)
    return "subset" if "config" in first and "|" in first and "nvlink_" in first else "markdown"


def load_all(base_dir, result_files):
    """Load and merge data from result files. Returns (merged_bw, merged_lat, transport_buffers)."""
    merged_bw = defaultdict(list)
    merged_lat = defaultdict(list)
    all_cols = []

    for rel_path in result_files:
        path = Path(base_dir) / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        if not path.exists():
            print(f"Warning: {path} not found")
            continue
        fmt = detect_format(path)
        if fmt == "subset":
            d, cols = parse_subset_format(path)
        else:
            d, cols = parse_markdown_format(path)
        all_cols.extend(cols)
        for k, vals in d["bw"].items():
            for v in vals:
                merged_bw[k].append(v)
        for k, vals in d["lat"].items():
            for v in vals:
                merged_lat[k].append(v)

    # Average merged values
    def avg(vals):
        return sum(vals) / len(vals) if vals else None

    merged_bw_f = {k: avg(v) for k, v in merged_bw.items() if v}
    merged_lat_f = {k: avg(v) for k, v in merged_lat.items() if v}

    # Auto-detect transport buffers from column names
    transport_buffers = extract_transport_buffers_from_cols(merged_bw_f.keys())
    if not transport_buffers:
        transport_buffers = extract_transport_buffers_from_cols(all_cols)
    if not transport_buffers:
        transport_buffers = ["8M", "16M", "32M", "64M", "128M", "256M", "1G"]  # fallback

    return merged_bw_f, merged_lat_f, transport_buffers


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


def extract_pxdx_data(merged_bw, merged_lat, transport_buffers):
    """PxDx: source=target. direct + allow_nvlink_reassembly (no alltoall)."""
    cfg_keys = {"P1D1": "src0", "P2D2": "src0_1", "P8D8": "src0_1_2_3_4_5_6_7"}

    bw = defaultdict(dict)
    lat = defaultdict(dict)

    for label in ["P1D1", "P2D2", "P8D8"]:
        cfg = cfg_keys[label]
        for k, v in merged_bw.items():
            if k[0] == cfg and k[1] == "direct":
                bw[label]["direct"] = v
                break
        for k, v in merged_lat.items():
            if k[0] == cfg and k[1] == "direct":
                lat[label]["direct"] = v
                break

        for tb in transport_buffers:
            # PxDx: use only markdown allow_nvlink_reassembly (no alltoall), NOT subset nvlink_ra_* (alltoall)
            col_md = f"allow_nvlink_reassembly_{tb}" if tb != "1G" else "allow_nvlink_reassembly_full"
            for k, v in merged_bw.items():
                if k[0] == cfg and k[1] == col_md:
                    bw[label][tb] = v
                    break
            for k, v in merged_lat.items():
                if k[0] == cfg and k[1] == col_md:
                    lat[label][tb] = v
                    break

    return bw, lat


def extract_pxdy_data(merged_bw, merged_lat, transport_buffers):
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

        for tb in transport_buffers:
            col_tb = tb if tb != "1G" else "full"

            # allow_nvlink: subset has nvlink_XM_dbl, nvlink_XM_nodbl; markdown has allow_nvlink_XM (single)
            def match_nvlink_dbl(c):
                return ("nvlink_" in c and tb in c and "ra_" not in c and "_dbl" in c) or False
            def match_nvlink_nodbl(c):
                return ("nvlink_" in c and tb in c and "ra_" not in c and "_nodbl" in c) or False
            def match_nvlink_md(c):
                return c == f"allow_nvlink_{col_tb}"

            vb_dbl = get_vals(merged_bw, cfgs, match_nvlink_dbl)
            vb_nodbl = get_vals(merged_bw, cfgs, match_nvlink_nodbl)
            vb_md = get_vals(merged_bw, cfgs, match_nvlink_md)
            if vb_dbl:
                bw[label][f"allow_nvlink_{tb}_dbl"] = sum(vb_dbl) / len(vb_dbl)
            if vb_nodbl:
                bw[label][f"allow_nvlink_{tb}_nodbl"] = sum(vb_nodbl) / len(vb_nodbl)
            if vb_md and f"allow_nvlink_{tb}_dbl" not in bw[label] and f"allow_nvlink_{tb}_nodbl" not in bw[label]:
                bw[label][f"allow_nvlink_{tb}"] = sum(vb_md) / len(vb_md)  # fallback

            vl_dbl = get_vals(merged_lat, cfgs, match_nvlink_dbl)
            vl_nodbl = get_vals(merged_lat, cfgs, match_nvlink_nodbl)
            vl_md = get_vals(merged_lat, cfgs, match_nvlink_md)
            if vl_dbl:
                lat[label][f"allow_nvlink_{tb}_dbl"] = sum(vl_dbl) / len(vl_dbl)
            if vl_nodbl:
                lat[label][f"allow_nvlink_{tb}_nodbl"] = sum(vl_nodbl) / len(vl_nodbl)
            if vl_md and f"allow_nvlink_{tb}_dbl" not in lat[label] and f"allow_nvlink_{tb}_nodbl" not in lat[label]:
                lat[label][f"allow_nvlink_{tb}"] = sum(vl_md) / len(vl_md)

            # allow_nvlink_reassembly_alltoall: dbl and nodbl
            def match_ra_dbl(c):
                return f"nvlink_ra_" in c and tb in c and "_dbl" in c
            def match_ra_nodbl(c):
                return f"nvlink_ra_" in c and tb in c and "_nodbl" in c
            def match_ra_md(c):
                return c == f"allow_nvlink_reassembly_alltoall_{col_tb}"

            vb_dbl = get_vals(merged_bw, cfgs, match_ra_dbl)
            vb_nodbl = get_vals(merged_bw, cfgs, match_ra_nodbl)
            vb_md = get_vals(merged_bw, cfgs, match_ra_md)
            if vb_dbl:
                bw[label][f"allow_nvlink_ra_alltoall_{tb}_dbl"] = sum(vb_dbl) / len(vb_dbl)
            if vb_nodbl:
                bw[label][f"allow_nvlink_ra_alltoall_{tb}_nodbl"] = sum(vb_nodbl) / len(vb_nodbl)
            if vb_md and f"allow_nvlink_ra_alltoall_{tb}_dbl" not in bw[label] and f"allow_nvlink_ra_alltoall_{tb}_nodbl" not in bw[label]:
                bw[label][f"allow_nvlink_ra_alltoall_{tb}"] = sum(vb_md) / len(vb_md)

            vl_dbl = get_vals(merged_lat, cfgs, match_ra_dbl)
            vl_nodbl = get_vals(merged_lat, cfgs, match_ra_nodbl)
            vl_md = get_vals(merged_lat, cfgs, match_ra_md)
            if vl_dbl:
                lat[label][f"allow_nvlink_ra_alltoall_{tb}_dbl"] = sum(vl_dbl) / len(vl_dbl)
            if vl_nodbl:
                lat[label][f"allow_nvlink_ra_alltoall_{tb}_nodbl"] = sum(vl_nodbl) / len(vl_nodbl)
            if vl_md and f"allow_nvlink_ra_alltoall_{tb}_dbl" not in lat[label] and f"allow_nvlink_ra_alltoall_{tb}_nodbl" not in lat[label]:
                lat[label][f"allow_nvlink_ra_alltoall_{tb}"] = sum(vl_md) / len(vl_md)

    return bw, lat


def build_pxdx_series_for_label(bw, lat, label, transport_buffers):
    """Build series for one PxDx config: direct (horizontal) + allow_nvlink_reassembly (per tb)."""
    x_labels = list(transport_buffers)
    direct_bw = bw.get(label, {}).get("direct")
    direct_lat = lat.get(label, {}).get("direct")
    ra_bw = [bw.get(label, {}).get(tb) for tb in x_labels]
    ra_lat = [lat.get(label, {}).get(tb) for tb in x_labels]
    return x_labels, direct_bw, direct_lat, ra_bw, ra_lat




def main():
    parser = argparse.ArgumentParser(description="Plot RDMA BW/Latency vs transport buffer")
    parser.add_argument("--result-files", nargs="+", help="Result file paths (relative to repo root or absolute)")
    parser.add_argument("--message-size", choices=["1GB", "128MB"], help="Message size for plot titles (default: infer from path)")
    parser.add_argument("--output-dir", help="Output directory for plots (default: BW_latency_tests)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else base / "BW_latency_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.result_files:
        result_files = args.result_files
    elif args.message_size == "128MB":
        result_files = RESULT_FILES_128MB
    else:
        result_files = RESULT_FILES_1GB

    msg_size = args.message_size
    if not msg_size:
        for p in result_files:
            if "128MB" in str(p):
                msg_size = "128MB"
                break
            if "1GB" in str(p):
                msg_size = "1GB"
                break
        msg_size = msg_size or "1GB"

    merged_bw, merged_lat, transport_buffers = load_all(base, result_files)
    print(f"Transport buffers (from data): {transport_buffers}")
    print(f"Message size: {msg_size}")

    mode_colors = {"direct": "tab:blue", "PXN dbl": "tab:green", "PXN nodbl": "tab:olive",
                   "PXN + reassembly dbl": "tab:orange", "PXN + reassembly nodbl": "tab:red"}
    mode_colors_pxdx = {"direct": "tab:blue", "PXN + reassembly": "tab:orange"}

    x_labels_pxdy = list(transport_buffers)

    # --- PxDx: one figure per P1D1, P2D2, P8D8. Lines: direct (horizontal), PXN + reassembly ---
    bw_pxdx, lat_pxdx = extract_pxdx_data(merged_bw, merged_lat, transport_buffers)
    for label in ["P1D1", "P2D2", "P8D8"]:
        x_labels, direct_bw, direct_lat, ra_bw, ra_lat = build_pxdx_series_for_label(bw_pxdx, lat_pxdx, label, transport_buffers)
        x_pos = list(range(len(x_labels)))

        fig, (ax_bw, ax_lat) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{label} (source = target) — direct, PXN + reassembly (full = {msg_size})")

        # BW: speed-of-light reference, direct, PXN + reassembly
        ax_bw.axhline(y=SPEED_OF_LIGHT_BW, color="gray", linestyle="--", alpha=0.7, label=f"speed of light (~{SPEED_OF_LIGHT_BW:.0f} GB/s)")
        if direct_bw is not None:
            ax_bw.axhline(y=direct_bw, color=mode_colors_pxdx["direct"], linestyle="-", label="direct")
        ra_x = [i for i, v in enumerate(ra_bw) if v is not None]
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
            ax_lat.axhline(y=direct_lat, color=mode_colors_pxdx["direct"], linestyle="-", label="direct")
        ra_x = [i for i, v in enumerate(ra_lat) if v is not None]
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
        fname = f"plot_{label.lower()}_pxdx_{msg_size.lower()}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")

    # --- PxDy: one figure per P1D4, P1D8, P2D4, P2D8, P8D8. Lines: direct (horizontal), PXN dbl/nodbl, PXN+reassembly dbl/nodbl ---
    bw_pxdy, lat_pxdy = extract_pxdy_data(merged_bw, merged_lat, transport_buffers)
    for label in ["P1D4", "P1D8", "P2D4", "P2D8", "P8D8"]:
        x_pos = list(range(len(x_labels_pxdy)))

        fig, (ax_bw, ax_lat) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{label} (source ≤ target) — direct --all-to-all, PXN, PXN + reassembly (full = {msg_size})")

        d = bw_pxdy.get(label, {})
        # BW: speed-of-light reference, direct_alltoall, PXN, PXN+reassembly
        ax_bw.axhline(y=SPEED_OF_LIGHT_BW, color="gray", linestyle="--", alpha=0.7, label=f"speed of light (~{SPEED_OF_LIGHT_BW:.0f} GB/s)")
        v_direct = d.get("direct_alltoall")
        if v_direct is not None:
            ax_bw.axhline(y=v_direct, color=mode_colors["direct"], linestyle="-", label="direct --all-to-all")

        # PXN and PXN+reassembly: dbl and nodbl (fallback to single value from markdown)
        pxn_dbl_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_{tb}_dbl") is not None]
        pxn_dbl_y = [d[f"allow_nvlink_{tb}_dbl"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_{tb}_dbl") is not None]
        pxn_nodbl_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_{tb}_nodbl") is not None]
        pxn_nodbl_y = [d[f"allow_nvlink_{tb}_nodbl"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_{tb}_nodbl") is not None]
        pxn_fb_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_{tb}") is not None and d.get(f"allow_nvlink_{tb}_dbl") is None and d.get(f"allow_nvlink_{tb}_nodbl") is None]
        pxn_fb_y = [d[f"allow_nvlink_{tb}"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_{tb}") is not None and d.get(f"allow_nvlink_{tb}_dbl") is None and d.get(f"allow_nvlink_{tb}_nodbl") is None]
        if pxn_dbl_x and pxn_dbl_y:
            ax_bw.plot(pxn_dbl_x, pxn_dbl_y, "o-", color=mode_colors["PXN dbl"], label="PXN dbl")
        if pxn_nodbl_x and pxn_nodbl_y:
            ax_bw.plot(pxn_nodbl_x, pxn_nodbl_y, "s-", color=mode_colors["PXN nodbl"], label="PXN nodbl")
        if pxn_fb_x and pxn_fb_y:
            ax_bw.plot(pxn_fb_x, pxn_fb_y, "o-", color=mode_colors["PXN dbl"], label="PXN")

        pra_dbl_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_ra_alltoall_{tb}_dbl") is not None]
        pra_dbl_y = [d[f"allow_nvlink_ra_alltoall_{tb}_dbl"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_ra_alltoall_{tb}_dbl") is not None]
        pra_nodbl_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_ra_alltoall_{tb}_nodbl") is not None]
        pra_nodbl_y = [d[f"allow_nvlink_ra_alltoall_{tb}_nodbl"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_ra_alltoall_{tb}_nodbl") is not None]
        pra_fb_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_ra_alltoall_{tb}") is not None and d.get(f"allow_nvlink_ra_alltoall_{tb}_dbl") is None and d.get(f"allow_nvlink_ra_alltoall_{tb}_nodbl") is None]
        pra_fb_y = [d[f"allow_nvlink_ra_alltoall_{tb}"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_ra_alltoall_{tb}") is not None and d.get(f"allow_nvlink_ra_alltoall_{tb}_dbl") is None and d.get(f"allow_nvlink_ra_alltoall_{tb}_nodbl") is None]
        if pra_dbl_x and pra_dbl_y:
            ax_bw.plot(pra_dbl_x, pra_dbl_y, "o-", color=mode_colors["PXN + reassembly dbl"], label="PXN + reassembly dbl")
        if pra_nodbl_x and pra_nodbl_y:
            ax_bw.plot(pra_nodbl_x, pra_nodbl_y, "s-", color=mode_colors["PXN + reassembly nodbl"], label="PXN + reassembly nodbl")
        if pra_fb_x and pra_fb_y:
            ax_bw.plot(pra_fb_x, pra_fb_y, "o-", color=mode_colors["PXN + reassembly dbl"], label="PXN + reassembly")

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
            ax_lat.axhline(y=v_direct, color=mode_colors["direct"], linestyle="-", label="direct --all-to-all")

        pxn_dbl_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_{tb}_dbl") is not None]
        pxn_dbl_y = [d[f"allow_nvlink_{tb}_dbl"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_{tb}_dbl") is not None]
        pxn_nodbl_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_{tb}_nodbl") is not None]
        pxn_nodbl_y = [d[f"allow_nvlink_{tb}_nodbl"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_{tb}_nodbl") is not None]
        pxn_fb_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_{tb}") is not None and d.get(f"allow_nvlink_{tb}_dbl") is None and d.get(f"allow_nvlink_{tb}_nodbl") is None]
        pxn_fb_y = [d[f"allow_nvlink_{tb}"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_{tb}") is not None and d.get(f"allow_nvlink_{tb}_dbl") is None and d.get(f"allow_nvlink_{tb}_nodbl") is None]
        if pxn_dbl_x and pxn_dbl_y:
            ax_lat.plot(pxn_dbl_x, pxn_dbl_y, "o-", color=mode_colors["PXN dbl"], label="PXN dbl")
        if pxn_nodbl_x and pxn_nodbl_y:
            ax_lat.plot(pxn_nodbl_x, pxn_nodbl_y, "s-", color=mode_colors["PXN nodbl"], label="PXN nodbl")
        if pxn_fb_x and pxn_fb_y:
            ax_lat.plot(pxn_fb_x, pxn_fb_y, "o-", color=mode_colors["PXN dbl"], label="PXN")

        pra_dbl_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_ra_alltoall_{tb}_dbl") is not None]
        pra_dbl_y = [d[f"allow_nvlink_ra_alltoall_{tb}_dbl"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_ra_alltoall_{tb}_dbl") is not None]
        pra_nodbl_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_ra_alltoall_{tb}_nodbl") is not None]
        pra_nodbl_y = [d[f"allow_nvlink_ra_alltoall_{tb}_nodbl"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_ra_alltoall_{tb}_nodbl") is not None]
        pra_fb_x = [i for i, tb in enumerate(x_labels_pxdy) if d.get(f"allow_nvlink_ra_alltoall_{tb}") is not None and d.get(f"allow_nvlink_ra_alltoall_{tb}_dbl") is None and d.get(f"allow_nvlink_ra_alltoall_{tb}_nodbl") is None]
        pra_fb_y = [d[f"allow_nvlink_ra_alltoall_{tb}"] for tb in x_labels_pxdy if d.get(f"allow_nvlink_ra_alltoall_{tb}") is not None and d.get(f"allow_nvlink_ra_alltoall_{tb}_dbl") is None and d.get(f"allow_nvlink_ra_alltoall_{tb}_nodbl") is None]
        if pra_dbl_x and pra_dbl_y:
            ax_lat.plot(pra_dbl_x, pra_dbl_y, "o-", color=mode_colors["PXN + reassembly dbl"], label="PXN + reassembly dbl")
        if pra_nodbl_x and pra_nodbl_y:
            ax_lat.plot(pra_nodbl_x, pra_nodbl_y, "s-", color=mode_colors["PXN + reassembly nodbl"], label="PXN + reassembly nodbl")
        if pra_fb_x and pra_fb_y:
            ax_lat.plot(pra_fb_x, pra_fb_y, "o-", color=mode_colors["PXN + reassembly dbl"], label="PXN + reassembly")

        ax_lat.set_xticks(x_pos)
        ax_lat.set_xticklabels(x_labels_pxdy, rotation=45, ha="right")
        ax_lat.set_ylabel("Latency (ms/iteration)")
        ax_lat.set_xlabel("Transport buffer")
        ax_lat.legend()
        ax_lat.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"plot_{label.lower()}_pxdy_{msg_size.lower()}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()
