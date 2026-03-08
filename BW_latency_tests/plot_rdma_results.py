#!/usr/bin/env python3
"""
Plot RDMA test results from rdma_tests_results_*.txt files.
Usage: python plot_rdma_results.py <result_file1> [result_file2 ...] [-o output_dir] [-m 128MB|128GB|1GB]

Creates:
- One image per source_gpus count
- Graph 1: All-to-all + allow_nvlink (direct_alltoall, allow_nvlink_reassembly_alltoall, allow_nvlink)
- Graph 2: Direct + allow_nvlink_reassembly (direct, allow_nvlink_reassembly)
- X axis: transport buffer size (8M..128M or 8M..1G). Use -m 128MB/128GB for 128 MB buffer tests.
- Y axis: Bandwidth (GB/s) or Latency (ms)
"""

import argparse
import re
import os
import matplotlib.pyplot as plt

# Transport buffer sizes: 128 MB config (no full) vs 1 GB config (full = 1G)
TB_CONFIG = {
    "128MB": {"sizes": [8, 16, 32, 64, 128], "labels": ["8M", "16M", "32M", "64M", "128M"], "xlim": (6, 150)},
    "1GB": {"sizes": [8, 16, 32, 64, 128, 1024], "labels": ["8M", "16M", "32M", "64M", "128M", "1G"], "xlim": (6, 1200)},
}

# Column name -> (mode, tb_index or -1 for direct)
COL_MAP = {
    "direct": ("direct", -1),
    "direct_alltoall": ("direct_alltoall", -1),
    "allow_nvlink_8M": ("allow_nvlink", 0),
    "allow_nvlink_16M": ("allow_nvlink", 1),
    "allow_nvlink_32M": ("allow_nvlink", 2),
    "allow_nvlink_64M": ("allow_nvlink", 3),
    "allow_nvlink_128M": ("allow_nvlink", 4),
    "allow_nvlink_full": ("allow_nvlink", 5),
    "allow_nvlink_reassembly_8M": ("allow_nvlink_reassembly", 0),
    "allow_nvlink_reassembly_16M": ("allow_nvlink_reassembly", 1),
    "allow_nvlink_reassembly_32M": ("allow_nvlink_reassembly", 2),
    "allow_nvlink_reassembly_64M": ("allow_nvlink_reassembly", 3),
    "allow_nvlink_reassembly_128M": ("allow_nvlink_reassembly", 4),
    "allow_nvlink_reassembly_full": ("allow_nvlink_reassembly", 5),
    "allow_nvlink_reassembly_alltoall_8M": ("allow_nvlink_reassembly_alltoall", 0),
    "allow_nvlink_reassembly_alltoall_16M": ("allow_nvlink_reassembly_alltoall", 1),
    "allow_nvlink_reassembly_alltoall_32M": ("allow_nvlink_reassembly_alltoall", 2),
    "allow_nvlink_reassembly_alltoall_64M": ("allow_nvlink_reassembly_alltoall", 3),
    "allow_nvlink_reassembly_alltoall_128M": ("allow_nvlink_reassembly_alltoall", 4),
    "allow_nvlink_reassembly_alltoall_full": ("allow_nvlink_reassembly_alltoall", 5),
}

# Graph 1: All-to-all options + allow_nvlink
MODES_ALLTOALL = ["direct_alltoall", "allow_nvlink_reassembly_alltoall", "allow_nvlink"]

# Graph 2: Direct + allow_nvlink_reassembly
MODES_DIRECT_REASSEMBLY = ["direct", "allow_nvlink_reassembly"]

COLORS = {
    "direct": "tab:blue",
    "direct_alltoall": "tab:orange",
    "allow_nvlink": "tab:green",
    "allow_nvlink_reassembly": "tab:red",
    "allow_nvlink_reassembly_alltoall": "tab:purple",
}


def parse_result_file(path):
    """Parse rdma_tests_results_*.txt and return {source_gpus: {metric: {mode: [vals] or scalar}}}"""
    with open(path) as f:
        content = f.read()

    data = {}
    # Split by section (support both original === format and Markdown ## format)
    if "=== Bandwidth (GB/s) ===" in content:
        bw_section = content.split("=== Bandwidth (GB/s) ===")[1].split("=== Latency")[0]
        lat_section = content.split("=== Latency (ms per iteration) ===")[1].split("All values")[0]
    elif "## Bandwidth (GB/s)" in content:
        bw_section = content.split("## Bandwidth (GB/s)")[1].split("## Latency")[0]
        lat_section = content.split("## Latency (ms per iteration)")[1].split("*All values")[0]
    else:
        raise ValueError(f"Cannot parse {path}: expected '=== Bandwidth (GB/s) ===' or '## Bandwidth (GB/s)'")

    def parse_section(section):
        lines = [l.strip() for l in section.strip().split("\n") if l.strip()]
        if not lines:
            return {}
        header = lines[0]
        # Parse header: source_gpus | col1 | col2 | ... (or | source_gpus | col1 | ... for Markdown)
        cols = [c.strip() for c in header.split("|")]
        col_names = [re.sub(r"\s+", "_", c) for c in cols]
        # Markdown tables have leading | so first element is empty; skip it for data column start
        data_start = 2 if cols and not cols[0] else 1
        result = {}
        for line in lines[1:]:
            if line.startswith("-"):
                continue
            parts = [p.strip() for p in line.split("|")]
            # Skip Markdown table separator rows (e.g. |:---|:---:|)
            first_cell = next((p for p in parts if p), "")
            if first_cell.startswith(":") or re.match(r"^-+$", first_cell):
                continue
            if len(parts) < data_start + 1:
                continue
            src_gpus = parts[data_start - 1].strip().rstrip(",")
            result[src_gpus] = {}
            for i, name in enumerate(col_names[data_start:], data_start):
                if i >= len(parts):
                    break
                val_str = parts[i].strip()
                if val_str == "--" or val_str == "":
                    continue
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                if name in COL_MAP:
                    mode, tb_idx = COL_MAP[name]
                    if mode not in result[src_gpus]:
                        result[src_gpus][mode] = {}
                    if tb_idx >= 0:
                        result[src_gpus][mode][tb_idx] = val
                    else:
                        result[src_gpus][mode]["direct"] = val
        return result

    bw_data = parse_section(bw_section)
    lat_data = parse_section(lat_section)

    for src in bw_data:
        if src not in data:
            data[src] = {"bw": {}, "lat": {}}
        data[src]["bw"] = bw_data[src]
    for src in lat_data:
        if src not in data:
            data[src] = {"bw": {}, "lat": {}}
        data[src]["lat"] = lat_data[src]

    return data


def merge_data(all_data):
    """Merge multiple parsed files. Later files override earlier for same source_gpus."""
    merged = {}
    for d in all_data:
        for src, metrics in d.items():
            if src not in merged:
                merged[src] = {"bw": {}, "lat": {}}
            for mode, vals in metrics.get("bw", {}).items():
                merged[src]["bw"][mode] = vals
            for mode, vals in metrics.get("lat", {}).items():
                merged[src]["lat"][mode] = vals
    return merged


def get_series(merged, src_gpus, metric, modes, is_direct_graph, tb_sizes):
    """Get (x_vals, y_vals) per mode for plotting."""
    series = {}
    src_data = merged.get(src_gpus, {})
    m_data = src_data.get("bw" if metric == "bw" else "lat", {})

    for mode in modes:
        if mode not in m_data:
            continue
        mode_data = m_data[mode]
        if "direct" in mode_data:
            val = mode_data["direct"]
            series[mode] = (tb_sizes, [val] * len(tb_sizes))
        else:
            xs, ys = [], []
            for i, tb in enumerate(tb_sizes):
                if i in mode_data:
                    xs.append(tb)
                    ys.append(mode_data[i])
            if xs:
                series[mode] = (xs, ys)
    return series


def plot_one_source(merged, src_gpus, output_dir, ngpus, message_size, tb_config):
    """Create one figure per source_gpus: 2 rows (BW, Lat) x 2 cols (direct+reassembly, alltoall+nvlink)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col")
    fig.suptitle(f"RDMA: {ngpus} source GPU(s) — {src_gpus} ({message_size})")

    tb_sizes = tb_config["sizes"]
    tb_labels = tb_config["labels"]
    xlim = tb_config["xlim"]

    def setup_axis(ax, xlabel=None):
        ax.set_xscale("log", base=2)
        ax.set_xticks(tb_sizes)
        ax.set_xticklabels(tb_labels)
        ax.set_xlim(*xlim)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    # Row 0: Bandwidth
    # Col 0: All-to-all + allow_nvlink
    ax = axes[0, 0]
    series = get_series(merged, src_gpus, "bw", MODES_ALLTOALL, False, tb_sizes)
    for mode, (xs, ys) in series.items():
        ax.plot(xs, ys, color=COLORS.get(mode, "gray"), marker="o", alpha=0.8, label=mode)
    setup_axis(ax)
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("All-to-all & allow_nvlink")
    ax.legend(fontsize=8)

    # Col 1: Direct + allow_nvlink_reassembly
    ax = axes[0, 1]
    series = get_series(merged, src_gpus, "bw", MODES_DIRECT_REASSEMBLY, True, tb_sizes)
    for mode, (xs, ys) in series.items():
        ax.plot(xs, ys, color=COLORS.get(mode, "gray"), marker="o", alpha=0.8, label=mode)
    setup_axis(ax)
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Direct & allow_nvlink_reassembly")
    ax.legend(fontsize=8)

    # Row 1: Latency
    ax = axes[1, 0]
    series = get_series(merged, src_gpus, "lat", MODES_ALLTOALL, False, tb_sizes)
    for mode, (xs, ys) in series.items():
        ax.plot(xs, ys, color=COLORS.get(mode, "gray"), marker="o", alpha=0.8, label=mode)
    setup_axis(ax, "Transport buffer size (MB)")
    ax.set_ylabel("Latency (ms / iteration)")
    ax.set_title("All-to-all & allow_nvlink")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    series = get_series(merged, src_gpus, "lat", MODES_DIRECT_REASSEMBLY, True, tb_sizes)
    for mode, (xs, ys) in series.items():
        ax.plot(xs, ys, color=COLORS.get(mode, "gray"), marker="o", alpha=0.8, label=mode)
    setup_axis(ax, "Transport buffer size (MB)")
    ax.set_ylabel("Latency (ms / iteration)")
    ax.set_title("Direct & allow_nvlink_reassembly")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    msg_suffix = message_size.replace(" ", "").replace("/", "_")
    outpath = os.path.join(output_dir, f"rdma_bw_lat_{ngpus}gpu_{src_gpus.replace(',', '_')}_{msg_suffix}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main():
    ap = argparse.ArgumentParser(description="Plot RDMA test results")
    ap.add_argument("files", nargs="+", help="rdma_tests_results_*.txt files")
    ap.add_argument("-o", "--output", default=".", help="Output directory for PNGs")
    ap.add_argument("-m", "--message-size", default=None,
                    help="Message/buffer size for title (e.g. 128MB, 128GB, 1GB). Auto-detected from filename if not set.")
    args = ap.parse_args()

    all_data = []
    for f in args.files:
        if not os.path.isfile(f):
            print(f"Warning: {f} not found, skipping")
            continue
        all_data.append(parse_result_file(f))

    if not all_data:
        print("No valid result files.")
        return 1

    # Infer message size from filenames if not set
    message_size = args.message_size
    if not message_size:
        paths = " ".join(args.files)
        if "128MB" in paths or "128mb" in paths.lower():
            message_size = "128 MB"
            tb_config = TB_CONFIG["128MB"]
        else:
            message_size = "1 GB"
            tb_config = TB_CONFIG["1GB"]
    else:
        # Normalize: 128MB -> 128 MB, 128GB -> 128 GB
        m = message_size.strip().upper()
        if "GB" in m:
            message_size = m.replace("GB", " GB").strip()
        elif "MB" in m:
            message_size = m.replace("MB", " MB").strip()
        else:
            message_size = m
        # 128 MB/GB tests use 128 MB x-axis (no full column)
        if "128" in message_size:
            tb_config = TB_CONFIG["128MB"]
        else:
            tb_config = TB_CONFIG["1GB"]

    merged = merge_data(all_data)
    os.makedirs(args.output, exist_ok=True)

    # Sort source_gpus by number of GPUs
    def sort_key(s):
        return len([x for x in s.split(",") if x.strip()])

    for src_gpus in sorted(merged.keys(), key=sort_key):
        ngpus = sort_key(src_gpus)
        outpath = plot_one_source(merged, src_gpus, args.output, ngpus, message_size, tb_config)
        print(f"Saved {outpath}")

    print(f"Done. Output in {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
