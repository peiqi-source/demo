import matplotlib.pyplot as plt
import numpy as np
import os


def generate_ranking_chart(algorithms, rank_1_data, top_3_data, save_path="ranking_chart.pdf",
                           figsize=(12, 2.8), bar_color='#343477', font_family='sans-serif', font_size=10):
    """
    生成算法排名并排水平柱状图，并导出为高清无损矢量图。

    参数:
        algorithms (list): 算法名称列表 (按从上到下的顺序排列)
        rank_1_data (list): 对应算法获得第 1 名的次数
        top_3_data (list): 对应算法进入前 3 名的次数
        save_path (str): 图片保存路径，建议使用 .pdf 或 .eps 以保证放大无损
        figsize (tuple): 图片尺寸 (宽, 高)，默认 (12, 2.8) 适合双栏论文通栏排版
        bar_color (str): 柱状图的填充颜色
        font_family (str): 字体系列，如 'sans-serif' 或 'Times New Roman'
        font_size (int): 全局字体大小
    """
    # 替换名称中的下划线为连字符，符合学术规范格式
    display_labels = [alg.replace('_', '-') for alg in algorithms]

    # 全局样式配置：刻度向内，无损输出设定
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['pdf.fonttype'] = 42  # 确保PDF中字体可编辑且不缺失
    plt.rcParams['ps.fonttype'] = 42

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    edge_color = 'black'
    bar_height = 0.7

    # ------------------ 左图：Rank #1 ------------------
    bars1 = ax1.barh(display_labels, rank_1_data, color=bar_color, edgecolor=edge_color, height=bar_height)
    ax1.set_xlabel('Number of times being ranked #1')

    # 动态计算 x 轴的最大值，留出 15% 的空间用于显示数字标签
    max_rank1 = max(rank_1_data) if max(rank_1_data) > 0 else 1
    ax1.set_xlim(0, max_rank1 * 1.15)
    ax1.invert_yaxis()
    ax1.tick_params(top=True, bottom=True, left=True, right=True)

    # 绘制数值标签
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + (max_rank1 * 0.02), bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}', va='center', ha='left', fontsize=font_size)

    # ------------------ 右图：Top 3 ------------------
    bars2 = ax2.barh(display_labels, top_3_data, color=bar_color, edgecolor=edge_color, height=bar_height)
    ax2.set_xlabel('Number of times being ranked in top 3')

    max_top3 = max(top_3_data) if max(top_3_data) > 0 else 1
    ax2.set_xlim(0, max_top3 * 1.15)
    ax2.invert_yaxis()
    ax2.tick_params(top=True, bottom=True, left=True, right=True)

    # 绘制数值标签
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + (max_top3 * 0.02), bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}', va='center', ha='left', fontsize=font_size)

    # ------------------ 调整与导出 ------------------
    plt.subplots_adjust(wspace=0.3)

    # 使用 bbox_inches='tight' 自动裁剪白边
    plt.savefig(save_path, format=save_path.split('.')[-1], bbox_inches='tight', dpi=600)
    plt.close()  # 释放内存
    print(f">>> 图表已成功保存为无损格式: {os.path.abspath(save_path)}")


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 准备你的数据 (按照表格从右到左的顺序)
    alg_names = ['AHD-EC', 'PTGP', 'PTA', 'WSCE', 'DREC', 'ECPCS', 'LWGP', 'LWEA', 'HBGF', 'CSPA', 'MCLA']
    r1_counts = [35, 1, 0, 9, 0, 0, 0, 0, 0, 3, 0]
    t3_counts = [45, 10, 5, 21, 3, 2, 8, 14, 11, 17, 8]

    # 2. 调用函数生成 PDF 矢量图 (极力推荐放入 LaTeX)
    generate_ranking_chart(
        algorithms=alg_names,
        rank_1_data=r1_counts,
        top_3_data=t3_counts,
        save_path="Experiment_Results_Ranking.pdf",  # 后缀改为 .pdf 即可导出矢量图
        figsize=(12, 2.8)  # 论文跨栏长图尺寸
    )

    # 3. 如果特定期刊要求 EPS 格式，只需更改后缀
    # generate_ranking_chart(..., save_path="Experiment_Results_Ranking.eps")