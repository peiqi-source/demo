import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime


# ============================================================
# 0. Global academic plotting style
# ============================================================
def set_academic_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    plt.rcParams["axes.linewidth"] = 1.35
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 1.15
    plt.rcParams["ytick.major.width"] = 1.15
    plt.rcParams["xtick.major.size"] = 4.5
    plt.rcParams["ytick.major.size"] = 4.5

    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 8.5
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["savefig.pad_inches"] = 0.02


# ============================================================
# 1. Basic configuration
# ============================================================
MAIN_METRIC = "ACC"
OBJ_COL = "Obj_History"
ALPHA_COL = "alphaA_History"

set_academic_style()

# Dataset order used in the paper
DATASET_NAME_MAP = {
    1: "UMIST",
    2: "VS",
    3: "COIL20",
    4: "SPF",
    5: "IS",
    6: "FCT",
    7: "MNIST",
    8: "OpticDigits",
    9: "LS",
    10: "ISOLET",
    11: "USPS",
    12: "PenDigits",
}


# ============================================================
# 2. Utility functions
# ============================================================
def extract_dataset_index(file_basename):
    """
    Extract dataset index from filenames such as:
    data1_xxx.csv, para_small_data7_xxx.csv, etc.
    """
    match = re.search(r"data(\d+)", file_basename, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def get_dataset_name(file_basename):
    idx = extract_dataset_index(file_basename)
    if idx in DATASET_NAME_MAP:
        return DATASET_NAME_MAP[idx]
    return file_basename


def parse_vector_from_string(s):
    if pd.isna(s):
        return np.array([], dtype=float)

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return np.array([float(x) for x in nums], dtype=float)


def parse_matrix_from_string(s):
    if pd.isna(s):
        return np.empty((0, 0), dtype=float)

    raw = str(s).strip()
    raw = raw.replace("[", "").replace("]", "")

    rows = [r.strip() for r in raw.split(";") if r.strip()]
    matrix = []

    for row in rows:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", row)
        if nums:
            matrix.append([float(x) for x in nums])

    if len(matrix) == 0:
        return np.empty((0, 0), dtype=float)

    max_len = max(len(r) for r in matrix)
    matrix = [r + [np.nan] * (max_len - len(r)) for r in matrix]
    return np.array(matrix, dtype=float)


def select_representative_run(df):
    required_cols = ["AnchorsRate", "K", "Order", "NumSampling", MAIN_METRIC]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the CSV file.")

    df_mean = (
        df.groupby(["AnchorsRate", "K", "Order", "NumSampling"])[MAIN_METRIC]
        .mean()
        .reset_index()
    )

    best_mean_idx = df_mean[MAIN_METRIC].idxmax()
    best_group = df_mean.loc[best_mean_idx]

    best_ar = best_group["AnchorsRate"]
    best_k = best_group["K"]
    best_ord = best_group["Order"]
    best_ns = best_group["NumSampling"]

    df_group = df[
        (df["AnchorsRate"] == best_ar)
        & (df["K"] == best_k)
        & (df["Order"] == best_ord)
        & (df["NumSampling"] == best_ns)
        ].copy()

    best_seed_idx = df_group[MAIN_METRIC].idxmax()
    best_row = df_group.loc[best_seed_idx]

    return best_row, {
        "AnchorsRate": best_ar,
        "K": best_k,
        "Order": best_ord,
        "NumSampling": best_ns,
        "ACC": best_group[MAIN_METRIC],
    }


def load_convergence_records(input_dir, target_datasets=None):
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    records = []

    for csv_file in csv_files:
        file_basename = os.path.basename(csv_file).replace(".csv", "")
        dataset_idx = extract_dataset_index(file_basename)
        dataset_name = get_dataset_name(file_basename)

        # 核心修改：同时支持序号(int)或名称(str)过滤
        if target_datasets is not None:
            if (dataset_idx not in target_datasets) and (dataset_name not in target_datasets):
                continue

        print(f"Processing {file_basename} ...")

        df = pd.read_csv(csv_file)
        best_row, best_params = select_representative_run(df)

        obj_vals = parse_vector_from_string(best_row[OBJ_COL])
        alpha_matrix = parse_matrix_from_string(best_row[ALPHA_COL])

        if obj_vals.size == 0:
            raise ValueError(f"Empty objective history in {file_basename}.")

        if alpha_matrix.size == 0:
            raise ValueError(f"Empty alpha history in {file_basename}.")

        if alpha_matrix.shape[1] != len(obj_vals) and alpha_matrix.shape[0] == len(obj_vals):
            alpha_matrix = alpha_matrix.T

        records.append(
            {
                "dataset_idx": dataset_idx if dataset_idx is not None else 999,
                "dataset_name": dataset_name,
                "file_basename": file_basename,
                "obj_vals": obj_vals,
                "alpha_matrix": alpha_matrix,
                "best_params": best_params,
            }
        )

    records = sorted(records, key=lambda x: x["dataset_idx"])
    return records


# ============================================================
# 3. Plotting functions
# ============================================================
def add_subfigure_label(ax, label_text, y_offset=-0.12):
    ax.text(
        0.5,
        y_offset,
        label_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )


def tune_axis_compactness(ax):
    ax.tick_params(axis="both", which="major", pad=2)
    ax.yaxis.labelpad = 2
    ax.xaxis.labelpad = 2
    ax.margins(x=0.025)


def enhance_axis_readability(ax):
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.15, length=4.5)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.xaxis.get_offset_text().set_fontsize(10.5)
    ax.yaxis.get_offset_text().set_fontsize(10.5)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)


VIVID_COLORS = [
    "#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E",
    "#17BECF", "#E377C2", "#8C564B", "#BCBD22", "#7F7F7F",
]


def plot_objective_grid(records, output_dir):
    num_records = len(records)
    if num_records == 0:
        return

    cols = min(6, num_records)
    rows = math.ceil(num_records / max(1, cols))

    fig_width = (17.2 / 6) * cols
    fig_height = (4.85 / 2) * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).flatten()

    color_obj = "#C00000"

    for idx, record in enumerate(records):
        ax = axes[idx]

        obj_vals = record["obj_vals"]
        iterations = np.arange(1, len(obj_vals) + 1)

        ax.plot(
            iterations, obj_vals,
            color=color_obj, marker="o", linewidth=2.6, markersize=4.8,
            markerfacecolor=color_obj, markeredgecolor="white", markeredgewidth=0.65, zorder=3
        )

        ax.grid(True, linestyle="--", alpha=0.32, zorder=0)
        ax.set_xticks(iterations)
        tune_axis_compactness(ax)
        enhance_axis_readability(ax)

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)

        if idx % cols == 0:
            ax.set_ylabel("Objective value", fontsize=13, fontweight="bold", labelpad=2)

        label = f"({chr(97 + idx)}) {record['dataset_name']}"
        add_subfigure_label(ax, label, y_offset=-0.12)

    for j in range(num_records, len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(left=0.052, right=0.995, top=0.965, bottom=0.075, wspace=0.19, hspace=0.30)

    pdf_path = os.path.join(output_dir, f"Objective_Convergence_{num_records}Datasets.pdf")
    png_path = os.path.join(output_dir, f"Objective_Convergence_{num_records}Datasets.png")

    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_weight_grid(records, output_dir):
    num_records = len(records)
    if num_records == 0:
        return

    cols = min(6, num_records)
    rows = math.ceil(num_records / max(1, cols))

    fig_width = (17.2 / 6) * cols
    fig_height = (5.05 / 2) * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).flatten()

    legend_handles = []
    legend_labels = []

    for idx, record in enumerate(records):
        ax = axes[idx]

        alpha_matrix = record["alpha_matrix"]
        num_views, num_iter = alpha_matrix.shape
        iterations = np.arange(1, num_iter + 1)

        for v in range(num_views):
            color_v = VIVID_COLORS[v % len(VIVID_COLORS)]
            line, = ax.plot(
                iterations, alpha_matrix[v, :],
                color=color_v, marker="o", linewidth=1.85, markersize=3.45,
                markerfacecolor=color_v, markeredgecolor="white", markeredgewidth=0.45,
                alpha=0.95, label=f"Partition {v + 1}", zorder=3
            )

            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(f"Partition {v + 1}")

        uniform_line = ax.axhline(
            1.0 / num_views, color="#111111", linestyle="-.", linewidth=1.7,
            alpha=0.95, label="Uniform", zorder=1
        )

        if idx == 0:
            legend_handles.append(uniform_line)
            legend_labels.append("Uniform")

        ax.grid(True, linestyle="--", alpha=0.32, zorder=0)
        ax.set_xticks(iterations)
        tune_axis_compactness(ax)
        enhance_axis_readability(ax)

        if idx % cols == 0:
            ax.set_ylabel(r"Weight $\alpha$", fontsize=13, fontweight="bold", labelpad=2)

        label = f"({chr(97 + idx)}) {record['dataset_name']}"
        add_subfigure_label(ax, label, y_offset=-0.12)

    for j in range(num_records, len(axes)):
        axes[j].axis("off")

    fig.legend(
        handles=legend_handles, labels=legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.975),
        ncol=len(legend_labels), frameon=True, edgecolor="black",
        columnspacing=0.72, handlelength=1.35, handletextpad=0.30,
        borderpad=0.25, labelspacing=0.15, fontsize=8.2
    )

    fig.subplots_adjust(left=0.052, right=0.995, top=0.865, bottom=0.075, wspace=0.19, hspace=0.31)

    pdf_path = os.path.join(output_dir, f"Alpha_Weights_{num_records}Datasets.pdf")
    png_path = os.path.join(output_dir, f"Alpha_Weights_{num_records}Datasets.png")

    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================
# 4. Main function
# ============================================================
def main(target_datasets=None):
    INPUT_DIR = r"F:/学习/MATLAB/demo/plot/D. Hyperparameter_Sensitivity_Analysis/results_ord3ns3_ARvsK"
    OUTPUT_DIR = r"plot_results_convergence_grid"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = load_convergence_records(INPUT_DIR, target_datasets)

    print(f"\nFound {len(records)} matching datasets.")
    if not records:
        print("Exiting...")
        return

    print("\nSelected representative runs:")
    for r in records:
        p = r["best_params"]
        print(
            f"{r['dataset_name']:>10s}: "
            f"AR={p['AnchorsRate']}, K={p['K']}, "
            f"ord={p['Order']}, ns={p['NumSampling']}, "
            f"ACC={p['ACC']:.4f}"
        )

    plot_objective_grid(records, OUTPUT_DIR)
    plot_weight_grid(records, OUTPUT_DIR)

    print("\nAll figures have been generated successfully.")


if __name__ == "__main__":
    # 既可以输入序号列表，也可以输入名称列表
    # 示例1 (序号): target_datasets = [1, 3, 7, 12]
    # 示例2 (名称): target_datasets = ["UMIST", "COIL20", "MNIST"]
    # 示例3 (全部): target_datasets = None

    target_datasets = [1, 3, 5, 7, 9, 11]  # 仅绘制 Dataset 1, 2, 7 作为示例
    main(target_datasets)