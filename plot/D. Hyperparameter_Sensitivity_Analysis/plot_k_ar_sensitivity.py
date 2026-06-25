from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_best_ar_curves_from_folder(
    folder_path: str,
    save_dir: Optional[str] = None,
    metric_name: str = "ACC",
    fixed_ord: Optional[int] = None,
    fixed_ns: Optional[int] = None,
    k_range: Optional[Union[Tuple[float, float], Iterable[float]]] = None,
    ar_range: Optional[Union[Tuple[float, float], Iterable[float]]] = None,
    dataset_name_map: Optional[Dict[int, str]] = None,
    file_pattern: str = "*.csv"
) -> None:
    """
    遍历指定文件夹中的每个 CSV，并分别绘制一张 K-best_ar 折线图。

    Parameters
    ----------
    folder_path : str
        存放 csv 文件的文件夹路径。

    save_dir : str, optional
        图片保存目录。若为 None，则默认保存在 folder_path 下的
        'best_ar_figures' 文件夹。

    metric_name : str, default='ACC'
        用于选择 best_ar 的评价指标，例如 'ACC' / 'NMI' / 'ARI'。

    fixed_ord : int, optional
        若指定，则只使用 Order == fixed_ord 的记录。

    fixed_ns : int, optional
        若指定，则只使用 NumSampling == fixed_ns 的记录。

    k_range : tuple or iterable, optional
        K 的筛选范围。
        - 若传入 tuple(min_k, max_k)，则筛选 min_k <= K <= max_k
        - 若传入 list/set/ndarray，则只保留其中列出的 K

    ar_range : tuple or iterable, optional
        AnchorsRate 的筛选范围。
        - 若传入 tuple(min_ar, max_ar)，则筛选 min_ar <= ar <= max_ar
        - 若传入 list/set/ndarray，则只保留其中列出的 ar

    dataset_name_map : dict, optional
        数据集编号到名称的映射。若为 None，则使用默认映射。

    file_pattern : str, default='*.csv'
        文件匹配模式。

    Returns
    -------
    None
        函数直接将图片保存到磁盘，不返回对象。

    Notes
    -----
    1. 每张图对应一个 CSV 文件。
    2. 对于每个 K，先在固定条件下对 (ar, seed) 的 metric 取均值，
       再选出 mean(metric) 最大时对应的 best_ar。
    3. 图风格为论文风格，适合直接用于参数敏感性分析。
    """

    # =========================
    # 1. 默认数据集名称映射
    # =========================
    if dataset_name_map is None:
        dataset_name_map = {
            1: "UMIST",
            2: "VS",
            3: "COIL20",
            4: "SPF",
            5: "IS",
            6: "FCT",
            7: "MNIST",
            8: "OptDigits",
            9: "LS",
            10: "ISOLET",
            11: "USPS",
            12: "PenDigits"
        }

    # =========================
    # 2. 顶刊风格设置
    # =========================
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "savefig.dpi": 600,
        "figure.dpi": 150
    })

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    if save_dir is None:
        save_dir = folder / "best_ar_figures"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(str(folder / file_pattern)))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files matched pattern '{file_pattern}' in {folder_path}")

    # =========================
    # 3. 内部工具函数
    # =========================
    def _apply_range_filter(series: pd.Series, value_range):
        """
        对 Series 应用范围过滤。
        支持 tuple(min,max) 或显式取值集合。
        """
        if value_range is None:
            return pd.Series([True] * len(series), index=series.index)

        if isinstance(value_range, tuple) and len(value_range) == 2:
            lo, hi = value_range
            return (series >= lo) & (series <= hi)

        allowed = set(value_range)
        return series.isin(allowed)

    # =========================
    # 4. 遍历每个 CSV，分别绘图
    # =========================
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        # -------- 字段检查 --------
        required_cols = ["DatasetID", "AnchorsRate", "K", metric_name]
        for col in required_cols:
            if col not in df.columns:
                print(f"[Skip] {file_name}: missing column '{col}'")
                continue

        # -------- 固定 ord / ns --------
        if fixed_ord is not None:
            if "Order" not in df.columns:
                print(f"[Skip] {file_name}: 'Order' not found but fixed_ord was given")
                continue
            df = df[df["Order"] == fixed_ord]

        if fixed_ns is not None:
            if "NumSampling" not in df.columns:
                print(f"[Skip] {file_name}: 'NumSampling' not found but fixed_ns was given")
                continue
            df = df[df["NumSampling"] == fixed_ns]

        if df.empty:
            print(f"[Skip] {file_name}: no rows after filtering ord/ns")
            continue

        # -------- K / ar 范围筛选 --------
        df = df[_apply_range_filter(df["K"], k_range)]
        df = df[_apply_range_filter(df["AnchorsRate"], ar_range)]

        if df.empty:
            print(f"[Skip] {file_name}: no rows after filtering K/ar range")
            continue

        # -------- 数据集名称 --------
        dataset_id = int(df["DatasetID"].iloc[0])
        dataset_name = dataset_name_map.get(dataset_id, f"Data {dataset_id}")

        # -------- 先按 (K, ar) 聚合 metric --------
        grouped = (
            df.groupby(["K", "AnchorsRate"])[metric_name]
            .mean()
            .reset_index()
        )

        k_vals = sorted(grouped["K"].unique())
        best_ar = []
        best_metric = []

        for k in k_vals:
            sub = grouped[grouped["K"] == k].copy()
            if sub.empty:
                continue

            idx = sub[metric_name].idxmax()
            best_ar.append(sub.loc[idx, "AnchorsRate"])
            best_metric.append(sub.loc[idx, metric_name])

        if len(best_ar) == 0:
            print(f"[Skip] {file_name}: no valid best_ar extracted")
            continue

        best_ar = np.array(best_ar, dtype=float)
        best_metric = np.array(best_metric, dtype=float)
        k_vals = np.array(k_vals, dtype=float)

        # =========================
        # 5. 绘图
        # =========================
        fig, ax = plt.subplots(figsize=(7.2, 5.2))

        # 主折线
        ax.plot(
            k_vals,
            best_ar,
            color="#1f4e79",
            linewidth=2.6,
            marker="o",
            markersize=6.5,
            markerfacecolor="#1f4e79",
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=rf"Best $ar$ selected by {metric_name}"
        )

        # 可选：给每个点标注值
        for x, y in zip(k_vals, best_ar):
            ax.text(
                x, y + 0.4, f"{int(y)}" if float(y).is_integer() else f"{y:.1f}",
                ha="center", va="bottom", fontsize=9, color="#1f4e79"
            )

        # 坐标轴与标题
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"Best $ar$")
        title_parts = [dataset_name]
        if fixed_ord is not None:
            title_parts.append(rf"$ord={fixed_ord}$")
        if fixed_ns is not None:
            title_parts.append(rf"$ns={fixed_ns}$")

        ax.set_title(
            " | ".join([title_parts[0], ", ".join(title_parts[1:])]) if len(title_parts) > 1 else title_parts[0],
            fontweight="bold",
            pad=10
        )

        # 网格
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        # 坐标范围优化，让变化更明显
        y_margin = max(1.0, 0.05 * (best_ar.max() - best_ar.min() + 1e-8))
        ax.set_ylim(best_ar.min() - y_margin, best_ar.max() + y_margin)

        # 若 ar 取值本身是离散整数，可直接用离散 ticks
        unique_ar_ticks = sorted(np.unique(best_ar))
        if len(unique_ar_ticks) <= 10:
            ax.set_yticks(unique_ar_ticks)

        ax.set_xticks(k_vals)

        # 图例
        ax.legend(frameon=False, loc="best")

        # 边距调整
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.88)

        # =========================
        # 6. 保存
        # =========================
        stem = Path(file_name).stem
        suffix_parts = []
        if fixed_ord is not None:
            suffix_parts.append(f"ord{fixed_ord}")
        if fixed_ns is not None:
            suffix_parts.append(f"ns{fixed_ns}")
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

        pdf_path = save_dir / f"{stem}_best_ar_vs_K_{metric_name}{suffix}.pdf"
        png_path = save_dir / f"{stem}_best_ar_vs_K_{metric_name}{suffix}.png"

        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)

        print(f"[Saved] {dataset_name}:")
        print(f"        {pdf_path}")
        print(f"        {png_path}")


# =========================================================
# 用法示例
# =========================================================
if __name__ == "__main__":
    plot_best_ar_curves_from_folder(
        folder_path=r"F:\学习\MATLAB\demo\plot\results\results_固定ord3ns3",
        metric_name="ACC",
        fixed_ord=3,
        fixed_ns=3,
        k_range=(4, 20),         # 例如只看 4 <= K <= 20
        ar_range=(10, 100)       # 例如只看 10 <= ar <= 100
    )