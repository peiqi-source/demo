import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime


def set_plot_style():
    """配置顶刊风格的全局绘图参数"""
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
    })


def load_data(filepath: str) -> pd.DataFrame:
    """读取数据，提供统一的数据入口"""
    return pd.read_csv(filepath)


def plot_acc_vs_anchorsrate_combined(df: pd.DataFrame, output_dir: str, output_filename: str):
    """
    模块 A (正方形+内嵌图例版): 将不同 K 值下 ACC 随 AnchorsRate 变化的曲线合并在一张图中
    """
    os.makedirs(output_dir, exist_ok=True)

    k_values = sorted(df['K'].unique())
    num_k = len(k_values)

    base_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
                   '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, k in enumerate(k_values):
        color = base_colors[i % len(base_colors)]
        df_k = df[df['K'] == k]

        grouped = df_k.groupby('AnchorsRate')['ACC'].agg(['mean', 'std']).reset_index()

        ax.plot(grouped['AnchorsRate'], grouped['mean'], marker='o', markersize=6,
                color=color, linewidth=2.0, label=f'K = {k}')

        ax.fill_between(grouped['AnchorsRate'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        color=color, alpha=0.15, edgecolor='none')

    ax.set_xlabel('Initial Anchor Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average ACC', fontsize=14, fontweight='bold')
    # ax.set_title('Performance vs. AnchorsRate across varying K', fontsize=15, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 【核心修改 2】：为 Y 轴底部增加额外的留白空间，防止图例遮挡数据
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    # 底部向下延伸 15% 的空间，顶部延伸 5%
    ax.set_ylim(ymin - y_range * 0.15, ymax + y_range * 0.05)

    # 【核心修改 3】：图例回归框内，设为双列排布，并开启 Matplotlib 的智能防遮挡 (loc='best')
    ax.legend(
        loc='best',  # 智能寻找图中最空旷的地方放置图例（通常是右下角）
        ncol=2,  # 折叠成 2 列，让图例框变成一个小方块，更省空间
        frameon=True,
        framealpha=0.85,  # 设定 85% 的不透明度，允许底层网格和曲线微弱透出
        edgecolor='black',
        fontsize=11
    )

    plt.tight_layout()
    filepath = os.path.join(output_dir, output_filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_acc_vs_k_single(df: pd.DataFrame, target_ar: int, output_dir: str):
    """
    模块 B: 绘制特定 AnchorsRate 下，平均 ACC 随 K 变化的折线图

    :param df: 原始数据集
    :param target_ar: 指定的 AnchorsRate 阈值
    :param output_filename: 输出图片名
    """
    os.makedirs(output_dir, exist_ok=True)

    df_filtered = df[df['AnchorsRate'] == target_ar]
    if df_filtered.empty:
        print(f"警告：未找到 AnchorsRate = {target_ar} 的数据。跳过绘制此图。")
        return

    grouped = df_filtered.groupby('K')['ACC'].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values(by='K')

    plt.figure(figsize=(7, 5))
    main_color = '#E64B35'

    plt.plot(grouped['K'], grouped['mean'], marker='s', markersize=8,
             color=main_color, linewidth=2.5, label=f'AR = {target_ar}')

    plt.fill_between(grouped['K'],
                     grouped['mean'] - grouped['std'],
                     grouped['mean'] + grouped['std'],
                     color=main_color, alpha=0.2, edgecolor='none')

    plt.xlabel('K (Number of Clusters)')
    plt.ylabel('Average ACC')
    plt.title(f'Performance vs. K at AnchorsRate = {target_ar}')
    plt.xticks(grouped['K'])  # 强制X轴只显示实际包含的K
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    filename = f'ACC_vs_K_AR{target_ar}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_single_3d_sensitivity(csv_file,
                               output_dir,
                               metrics=['ACC', 'Runtime'],
                               find_best_by='ACC',
                               target_ar=None,
                               view_elev=40,
                               view_azim=-150):
    """
    根据单个 CSV 文件绘制 1-3 层的 3D 柱状图，展示参数敏感性。

    参数:
        csv_file (str): 输入的 CSV 文件路径。
        output_dir (str): 图片输出的文件夹路径。
        metrics (list of str): 需要展示的指标列表，最少1个，最多3个 (默认 ['ACC', 'Runtime'])。
        find_best_by (str): 用来寻找最优 AnchorsRate (ar) 的基准指标 (默认 'ACC')。
        target_ar (float/int/None): 指定的 AnchorsRate。如果为 None，则根据 find_best_by 自动寻优。
        view_elev (int): 3D 视角的仰角 (默认 40)。
        view_azim (int): 3D 视角的方位角 (默认 -150)。

    返回:
        str: 成功生成的图片完整路径，若失败则返回 None。
    """
    if len(metrics) > 3 or len(metrics) < 1:
        raise ValueError("metrics 列表中的指标数量必须在 1 到 3 之间！")

    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(csv_file)
    file_name_no_ext = os.path.splitext(file_name)[0]

    # 预设每一层的配色方案 (蓝系、红橙系、紫系)
    color_maps = ['GnBu', 'YlOrRd', 'Purples']

    try:
        df = pd.read_csv(csv_file)

        # 为了防错，确保 df 中包含我们需要的分组依据和提取指标
        required_cols = ['AnchorsRate', 'K', 'Order', 'NumSampling'] + metrics
        if find_best_by not in required_cols:
            required_cols.append(find_best_by)

        df_mean = df.groupby(['AnchorsRate', 'K', 'Order', 'NumSampling'])[required_cols[4:]].mean().reset_index()

        # ==========================================
        # 1. 确定底座 AnchorsRate (指定 vs 自动寻优)
        # ==========================================
        print(f"\n正在处理文件: {file_name}")
        if target_ar is not None:
            best_ar = target_ar
            print(f"--> [指定底座]: AnchorsRate (ar) = {best_ar}")
        else:
            best_idx = df_mean[find_best_by].idxmax()
            best_ar = df_mean.loc[best_idx, 'AnchorsRate']
            print(f"--> [自动寻优]: 根据 {find_best_by} 锁定最优 AnchorsRate (ar) = {best_ar}")

        df_plot = df_mean[df_mean['AnchorsRate'] == best_ar].copy()

        if df_plot.empty:
            print(f"警告：文件中未找到 AnchorsRate = {best_ar} 的数据，跳过该文件。")
            return None

        # ==========================================
        # 2. 动态构建 3D 画布与坐标系
        # ==========================================
        k_values = sorted(df_plot['K'].unique())
        num_k = len(k_values)
        num_metrics = len(metrics)

        plt.style.use('default')
        # 根据层数动态调整画布高度
        fig = plt.figure(figsize=(5 * num_k, 5 * num_metrics))

        x_vals = sorted(df_plot['Order'].unique())
        y_vals = sorted(df_plot['NumSampling'].unique())
        X, Y = np.meshgrid(x_vals, y_vals)
        xpos = X.flatten() - 0.3
        ypos = Y.flatten() - 0.3
        dx, dy = 0.6, 0.6

        # ==========================================
        # 3. 遍历 K 值与 Metrics，绘制 3D 柱状图矩阵
        # ==========================================
        for i, k_val in enumerate(k_values):
            df_k = df_plot[df_plot['K'] == k_val]

            for row_idx, metric in enumerate(metrics):
                Z = np.full(X.shape, np.nan)

                # 填充当前 metric 的 Z 轴数据
                for r in range(len(y_vals)):
                    for c in range(len(x_vals)):
                        row_data = df_k[(df_k['Order'] == x_vals[c]) & (df_k['NumSampling'] == y_vals[r])]
                        if not row_data.empty:
                            Z[r, c] = row_data[metric].values[0]

                # 获取指标极值与基准线 (如果包含 Time/Runtime 则底座为 0，否则为最小值的 98%)
                m_min, m_max = df_plot[metric].min(), df_plot[metric].max()
                baseline = 0 if ('Time' in metric or 'Runtime' in metric) else (m_min * 0.98)
                Z = np.nan_to_num(Z, nan=baseline)

                # 添加子图
                ax = fig.add_subplot(num_metrics, num_k, row_idx * num_k + i + 1, projection='3d')
                dz = Z.flatten() - baseline
                zpos = np.full_like(dz, baseline)

                # 设置颜色映射
                cmap = plt.get_cmap(color_maps[row_idx])
                norm = plt.Normalize(baseline, m_max)
                colors = cmap(norm(Z.flatten()))

                # 绘制 3D 柱子
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
                         color=colors, edgecolor='black', linewidth=0.3, alpha=0.95)

                # 设置视角和标签
                ax.view_init(elev=view_elev, azim=view_azim)
                ax.set_title(f'K = {k_val} ({metric})', pad=10, fontweight='bold', fontsize=14)
                ax.set_xlabel('Order ($ord$)', labelpad=8)
                ax.set_ylabel('Samplings ($ns$)', labelpad=8)

                # 只在最左侧的图显示 Z 轴标签
                if i == 0:
                    label_str = f'{metric} (s)' if 'Time' in metric or 'Runtime' in metric else metric
                    ax.set_zlabel(label_str, labelpad=8, fontweight='bold')
                else:
                    ax.set_zticklabels([])

                ax.set_xticks(x_vals)
                ax.set_yticks(y_vals)
                ax.set_zlim(baseline, m_max * 1.02)

        # ==========================================
        # 4. 总标题与 Colorbars 高清导出
        # ==========================================
        title_mode = f"Target $ar={best_ar}$" if target_ar else f"Optimal $ar={best_ar}$ (by {find_best_by})"
        fig.suptitle(f'Parameter Sensitivity: Metrics vs. Order & Samplings [{title_mode}]',
                     y=0.98, fontsize=18, fontweight='bold')

        # 动态添加 Colorbar
        for row_idx, metric in enumerate(metrics):
            m_min, m_max = df_plot[metric].min(), df_plot[metric].max()
            baseline = 0 if ('Time' in metric or 'Runtime' in metric) else (m_min * 0.98)
            norm = plt.Normalize(baseline, m_max)
            sm = cm.ScalarMappable(cmap=color_maps[row_idx], norm=norm)
            sm.set_array([])

            # 计算 colorbar 的高度和底部位置，使其与子图的层数对应对齐
            cb_height = (0.8 / num_metrics) - 0.05
            cb_bottom = 0.1 + (num_metrics - 1 - row_idx) * (0.8 / num_metrics)

            cbar_ax = fig.add_axes([0.92, cb_bottom, 0.015, cb_height])
            label_str = f'{metric} (s)' if 'Time' in metric or 'Runtime' in metric else metric
            fig.colorbar(sm, cax=cbar_ax, label=label_str)

        # 调整排版防止重叠
        plt.subplots_adjust(left=0.05, right=0.88, top=0.9, bottom=0.05, wspace=0.15, hspace=0.25)

        # 文件命名逻辑
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_str = "FixedAR" if target_ar else "AutoAR"
        param = f'{mode_str}{best_ar}_{timestamp}.pdf'

        parts = file_name_no_ext.split('_')
        data_key = next((part for part in parts if part.startswith('data')), file_name_no_ext)

        filename = f"{data_key}_{param}"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=600)
        plt.show()
        plt.close(fig)  # 清理内存
        print(f"--> [成功]: 3D 柱状图已保存至: {filepath}\n")

        return filepath

    except Exception as e:
        print(f"--> [错误]: 处理文件 {file_name} 时发生错误: {e}")
        return None

# 脚本入口
if __name__ == "__main__":
    """
    主程序控制流
    """
    # 1. 核心参数配置
    filepath = 'results_ord3ns3_ARvsK/para_small_data11_20260424_130501.csv'
    # filepath = 'para_small_data8_20260603_120752.csv'

    # 2. 初始化环境与加载数据
    set_plot_style()
    df = load_data(filepath)

    # 3. 运行绘图模块 A：所有K值的自适应网格图
    # 定义输出图片名称
    out_grid_filename = 'data11_ar.pdf'
    # out_grid_dir = 'plot_ar_results_ord3ns3_ARvsK/data12'
    out_grid_dir = 'data1'
    print(f"正在绘制网格图并保存为文件夹 {out_grid_dir} ...")
    plot_acc_vs_anchorsrate_combined(df, out_grid_dir, out_grid_filename)

    # 4. 运行绘图模块 B：特定 AnchorsRate 下的 ACC vs K
    # 定义输出图片名称
    # target_ar_for_plot = 40  # 针对模块B，你想观察的具体 AnchorsRate
    # out_single_filename = f'ACC_vs_K_AR{target_ar_for_plot}.png'

    # output_dir = 'plot_ar_results_ord3ns3_ARvsK/data12/ACC_vs_K'
    # for i in range(10, 101, 2):
    #     target_ar_for_plot = i
    #     print(f"正在绘制单线图并保存到 {output_dir} ...")
    #     plot_acc_vs_k_single(df, target_ar_for_plot, output_dir)

    # plot_single_3d_sensitivity(
    #     csv_file='small_data4/para_small_data4_20260331_123434.csv',
    #     output_dir='small_data4',
    #     metrics=['ACC', 'Runtime'],
    #     find_best_by='ACC'  # 告诉函数基于 NMI 最高点找 ar
    #  )

    print("所有图表绘制完毕！")


