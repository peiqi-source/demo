from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_ablation_csv(csv_path):
    """读取消融实验 CSV，并兼容常见的 UTF-8/GBK 编码。"""
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(csv_path)


def _normalize_variant_names(df):
    """规范化变体名称：将空值或 NULL 统一显示为 wo_all。"""
    df = df.copy()
    df["Variant"] = df["Variant"].fillna("wo_all")
    df["Variant"] = df["Variant"].replace({"NULL": "wo_all", "null": "wo_all", "None": "wo_all"})
    return df


def _dataset_display_name(name):
    """将较长的数据集名称压缩成适合横坐标显示的短名称。"""
    name_map = {
        "VS(vehicle)": "VS",
        "SPF(steel plate)": "SPF",
        "IS(image segmentation)": "IS",
        "FCT(forest)": "FCT",
        "LS(Landsat)": "LS",
    }
    return name_map.get(name, name)


def _select_datasets(df, datasets):
    """
    根据用户输入筛选数据集。
    datasets=None 时默认显示 CSV 中按 DatasetID 排列的全部数据集；
    datasets 可传入 DatasetID，如 [1, 3, 7]，也可传入显示名，如 ["UMIST", "COIL20"]。
    """
    dataset_table = (
        df[["DatasetID", "DatasetName"]]
        .drop_duplicates()
        .sort_values("DatasetID")
        .reset_index(drop=True)
    )
    dataset_table["DisplayName"] = dataset_table["DatasetName"].map(_dataset_display_name)

    if datasets is None:
        return dataset_table

    selected_rows = []
    for item in datasets:
        if isinstance(item, int):
            hit = dataset_table[dataset_table["DatasetID"] == item]
        else:
            key = str(item).strip().lower()
            hit = dataset_table[
                dataset_table["DatasetName"].str.lower().eq(key)
                | dataset_table["DisplayName"].str.lower().eq(key)
            ]
        if hit.empty:
            raise ValueError(f"Unknown dataset selector: {item}")
        selected_rows.append(hit.iloc[0])

    return pd.DataFrame(selected_rows).drop_duplicates("DatasetID").reset_index(drop=True)


def _metric_column(metric):
    """把 ACC/NMI/ARI 等简写转换为 CSV 中的均值列名。"""
    metric = metric.replace("_mean", "").upper()
    column = f"{metric}_mean"
    title_map = {"ACC": "Acc", "NMI": "NMI", "ARI": "ARI"}
    return column, title_map.get(metric, metric)


def _set_information_fusion_style():
    """设置接近 Information Fusion 示例图的全局字体、线条和导出风格。"""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
            "axes.linewidth": 0.9,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def plot_information_fusion_ablation(
    csv_path=None,
    output_path=None,
    datasets=None,
    metrics=("ACC", "NMI"),
    variants=("wo_HO", "wo_MS", "wo_AW", "wo_all"),
    variant_labels=None,
    colors=None,
    bar_width=0.16,
    bar_gap=0.035,
    group_gap=1.0,
    value_scale=100.0,
    value_format="{:.2f}",
    show_values=True,
    value_rotation=60,
    x_rotation=38,
    subplot_wspace=0.12,
    legend_y=0.045,
    bottom_margin=0.235,
    dpi=600,
):
    """
    绘制 Information Fusion 风格的前端消融实验柱状图。

    Parameters
    ----------
    csv_path : str or Path, optional
        消融实验 CSV 路径；默认读取当前脚本同目录下的
        Master_AHD_Frontend_NULL_Ablation_20260511_154014.csv。
    output_path : str or Path, optional
        输出图片路径；默认保存为 Fig_Ablation_Frontend_IF_Style.pdf。
    datasets : list[int | str], optional
        控制图中显示哪些数据集。None 表示显示全部 12 个数据集。
        示例：[1, 3, 7] 或 ["UMIST", "COIL20", "MNIST"]。
    metrics : tuple[str, ...]
        控制子图显示哪些指标，默认复刻参考图的 Acc 和 NMI 双子图。
    variants : tuple[str, ...]
        控制柱子的显示顺序。CSV 中的 NULL 会自动显示为 wo_all。
    variant_labels : dict, optional
        图例名称映射；默认直接显示 wo_HO、wo_MS、wo_AW、wo_all。
    colors : list[str], optional
        柱子颜色。默认使用接近参考图的红、橙、粉、浅灰紫。
    bar_width : float
        单个柱子的宽度。
    bar_gap : float
        同一数据集组内相邻柱子之间的留白。
    group_gap : float
        相邻数据集组之间的横向间距。
    value_scale : float
        数值缩放系数。CSV 为 0--1，小数乘以 100 后显示为百分制。
    value_format : str
        柱顶数值格式，默认保留两位小数。
    show_values : bool
        是否在柱顶斜着显示浅色数值。
    value_rotation : float
        柱顶数值倾斜角度。
    x_rotation : float
        横坐标数据集名称倾斜角度。
    subplot_wspace : float
        两个子图之间的水平空白；值越小，两幅图越靠近。
    legend_y : float
        图例在整张图中的纵向位置；值越大，图例越靠近图体。
    bottom_margin : float
        子图底部边距；用于控制横坐标标签和图例占用的空间。
    dpi : int
        图片导出分辨率。

    Returns
    -------
    matplotlib.figure.Figure
        返回 figure 对象，方便外部继续微调或嵌入其他流程。
    """
    script_dir = Path(__file__).resolve().parent
    csv_path = Path(csv_path) if csv_path is not None else script_dir / "Master_AHD_Frontend_NULL_Ablation_20260511_154014.csv"
    output_path = Path(output_path) if output_path is not None else script_dir / "Fig_Ablation_Frontend_IF_Style.pdf"

    # 1. 读取并标准化数据，确保 NULL 最终以 wo_all 进入绘图。
    df = _normalize_variant_names(_read_ablation_csv(csv_path))

    # 2. 按用户要求选择数据集，默认保留全部 12 个数据集。
    dataset_table = _select_datasets(df, datasets)
    dataset_names = dataset_table["DatasetName"].tolist()
    display_names = dataset_table["DisplayName"].tolist()
    n_datasets = len(dataset_names)

    # 3. 按给定顺序选择变体，缺失变体直接报错，避免图例和数据不一致。
    available_variants = set(df["Variant"].unique())
    missing_variants = [variant for variant in variants if variant not in available_variants]
    if missing_variants:
        raise ValueError(f"Variants not found in CSV: {missing_variants}")

    # 4. 配置图例文字和配色；颜色贴近参考图的低饱和红/橙/粉体系。
    variant_labels = variant_labels or {
        "wo_HO": "Base + HO",
        "wo_MS": "Base + MS",
        "wo_AW": "Base + AW",
        "wo_all": "Base",
    }
    colors = colors or ["#D65A6A", "#F3B36B", "#F7A9B3", "#B9B5CF"]

    # 5. 配置全局图形风格，并根据数据集数量自适应画布宽度。
    _set_information_fusion_style()
    n_metrics = len(metrics)
    panel_width = max(5.9, 0.42 * n_datasets + 2.0)
    fig_width = panel_width * n_metrics
    fig_height = 4.55
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.ravel()

    # 6. 手动计算柱子坐标，用 bar_gap 明确制造柱子之间的留白。
    x = np.arange(n_datasets) * group_gap
    n_variants = len(variants)
    total_group_width = n_variants * bar_width + (n_variants - 1) * bar_gap
    start_offset = -total_group_width / 2 + bar_width / 2

    for ax, metric in zip(axes, metrics):
        metric_column, metric_title = _metric_column(metric)
        if metric_column not in df.columns:
            raise ValueError(f"Metric column not found in CSV: {metric_column}")

        # 7. 逐变体绘制柱子：每个数据集组内有固定留白，每个子图共用同一布局。
        all_values = []
        for variant_idx, variant in enumerate(variants):
            values = []
            for dataset_name in dataset_names:
                hit = df[(df["DatasetName"] == dataset_name) & (df["Variant"] == variant)]
                if hit.empty:
                    raise ValueError(f"Missing value for dataset={dataset_name}, variant={variant}")
                values.append(float(hit.iloc[0][metric_column]) * value_scale)

            values = np.asarray(values)
            all_values.extend(values.tolist())
            offset = start_offset + variant_idx * (bar_width + bar_gap)
            bars = ax.bar(
                x + offset,
                values,
                width=bar_width,
                color=colors[variant_idx % len(colors)],
                edgecolor="white",
                linewidth=0.65,
                label=variant_labels.get(variant, variant),
                zorder=3,
            )

            # 8. 柱顶数值：解决“遮挡”和“不明显”问题
            if show_values:
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 1.5,  # 【修改】Y轴间距从 1.2 提升至 1.5，避免文字贴近柱体边缘
                        value_format.format(value),
                        ha="center",  # 【修改】配合 90 度垂直显示，改为 center 居中对齐
                        va="bottom",
                        rotation=value_rotation,
                        color="#333333",
                        fontsize=8.0,
                        alpha=0.88,
                    )

        # 9. 子图坐标轴：灰色标题、浅网格、弱化边框，贴近参考图视觉。
        y_min = max(0, np.floor((min(all_values) - 5) / 10) * 10)
        y_max = min(105, max(100, np.ceil((max(all_values) + 8) / 10) * 10))
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 1, 10))
        # 标题手动放置在坐标轴左边界上方，确保 Acc/NMI 与纵轴左对齐。
        ax.text(
            0.0,
            1.045,
            metric_title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#777777",
            fontsize=15,
            fontweight="bold",
            clip_on=False,
        )
        ax.grid(axis="y", color="#DEDEDE", linewidth=1.0, linestyle="-", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=x_rotation, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="both", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DDDDDD")
        ax.spines["bottom"].set_color("#DDDDDD")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

    # 10. 共享图例：放在两个子图下方，使用虚线边框复刻参考图安排。
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=n_variants,
        frameon=True,
        fancybox=False,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=1.35,
        borderpad=0.25,
    )
    legend.get_frame().set_edgecolor("#BEBEBE")
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_linestyle("--")
    legend.get_frame().set_facecolor("white")

    # 11. 紧凑排版并导出，bbox_inches 保证图例不会被裁剪。
    fig.subplots_adjust(
        left=0.052,
        right=0.995,
        top=0.895,
        bottom=bottom_margin,
        wspace=subplot_wspace,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # 默认运行：显示全部 12 个数据集，绘制 Acc 与 NMI 两个子图。
    plot_information_fusion_ablation(output_path='ablation_frontend.pdf', datasets=[1, 3, 6, 7, 9, 10, 11, 12])
    print("Saved: Fig_Ablation_Frontend_IF_Style.pdf")
