import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime

# ==========================================
# 1. 读取真实数据与设置文件夹
# ==========================================
csv_file = 'results/merged_output/para_small_PenDigits.csv'  # 替换为你的文件
output_dir = 'figures0321_double3D'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file)

# ==========================================
# 2. 自动寻优：锁定全局最优的 AnchorsRate
# ==========================================
# 对 Seed 求平均
df_mean = df.groupby(['AnchorsRate', 'K', 'Order', 'NumSampling'])[['ACC', 'Runtime']].mean().reset_index()

# 找出平均 ACC 最高的 ar
best_idx = df_mean['ACC'].idxmax()
best_ar = df_mean.loc[best_idx, 'AnchorsRate']

print(f"🔧 自动锁定最优底座: AnchorsRate (ar) = {best_ar}")
df_plot = df_mean[df_mean['AnchorsRate'] == best_ar].copy()

# ==========================================
# 3. 动态构建 2x4 真实数据 3D 画布
# ==========================================
k_values = sorted(df_plot['K'].unique())
num_k = len(k_values)

plt.style.use('default')
# 设置巨幅画布：宽20，高10
fig = plt.figure(figsize=(5 * num_k, 10))

# 提取独特的 X (Order) 和 Y (NumSampling) 构造网格
x_vals = sorted(df_plot['Order'].unique())
y_vals = sorted(df_plot['NumSampling'].unique())
X, Y = np.meshgrid(x_vals, y_vals)

# 为了让同一层的 Z 轴比例统一，提取全局的最大最小值
acc_min, acc_max = df_plot['ACC'].min(), df_plot['ACC'].max()
time_min, time_max = df_plot['Runtime'].min(), df_plot['Runtime'].max()

# ==========================================
# 4. 遍历 K 值，绘制上下两层图
# ==========================================
for i, k_val in enumerate(k_values):
    df_k = df_plot[df_plot['K'] == k_val]

    # 构建 Z 轴真实数据矩阵
    Z_acc = np.full(X.shape, np.nan)
    Z_time = np.full(X.shape, np.nan)

    for r in range(len(y_vals)):
        for c in range(len(x_vals)):
            row = df_k[(df_k['Order'] == x_vals[c]) & (df_k['NumSampling'] == y_vals[r])]
            if not row.empty:
                Z_acc[r, c] = row['ACC'].values[0]
                Z_time[r, c] = row['Runtime'].values[0]

    # 填充缺失值，防止绘图断裂 (使用底部的极小值兜底)
    Z_acc = np.nan_to_num(Z_acc, nan=acc_min)
    Z_time = np.nan_to_num(Z_time, nan=time_min)

    # -----------------------------------------------------
    # 【上层】：ACC 曲面图 (i + 1 位置)
    # -----------------------------------------------------
    ax1 = fig.add_subplot(2, num_k, i + 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_acc, cmap='GnBu', edgecolor='k', linewidth=0.3, alpha=0.9)
    ax1.view_init(elev=25, azim=-135)

    ax1.set_title(f'K = {k_val} (Accuracy)', pad=10, fontweight='bold', fontsize=14)
    ax1.set_xlabel('Order ($ord$)', labelpad=8)
    ax1.set_ylabel('Samplings ($ns$)', labelpad=8)

    # 只在最左边的图显示 Z 轴标签
    if i == 0:
        ax1.set_zlabel('ACC', labelpad=8, fontweight='bold')
    else:
        ax1.set_zticklabels([])

    ax1.set_xticks(x_vals)
    ax1.set_yticks(y_vals)
    ax1.set_zlim(acc_min * 0.98, acc_max * 1.02)  # 留出一点边距

    # -----------------------------------------------------
    # 【下层】：Runtime 曲面图 (i + 1 + num_k 位置)
    # -----------------------------------------------------
    ax2 = fig.add_subplot(2, num_k, i + 1 + num_k, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_time, cmap='YlOrRd', edgecolor='k', linewidth=0.3, alpha=0.9)
    # 时间图的视角稍微偏一点，以看清指数上升的陡峭面
    ax2.view_init(elev=25, azim=-125)

    ax2.set_title(f'K = {k_val} (Runtime)', pad=10, fontweight='bold', fontsize=14)
    ax2.set_xlabel('Order ($ord$)', labelpad=8)
    ax2.set_ylabel('Samplings ($ns$)', labelpad=8)

    if i == 0:
        ax2.set_zlabel('Time (s)', labelpad=8, fontweight='bold')
    else:
        ax2.set_zticklabels([])

    ax2.set_xticks(x_vals)
    ax2.set_yticks(y_vals)
    ax2.set_zlim(0, time_max * 1.05)  # 时间从 0 开始显得更严谨

# ==========================================
# 5. 总标题与高清导出
# ==========================================
fig.suptitle(f'Parameter Sensitivity: Accuracy vs. Runtime (Optimal $ar={best_ar}$)',
             y=0.98, fontsize=18, fontweight='bold')

# 添加色阶条 (Colorbar) - 给上下两层分别加一个细长的刻度尺
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.3])  # [left, bottom, width, height]
fig.colorbar(surf1, cax=cbar_ax1, label='ACC')

cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.3])
fig.colorbar(surf2, cax=cbar_ax2, label='Runtime (s)')

plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05, wspace=0.1, hspace=0.2)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'PenDigits_3D_AR{best_ar}_{timestamp}.pdf'
filepath = os.path.join(output_dir, filename)

# plt.savefig(filepath, dpi=600)
# print(f" 2x4 真实数据 3D 矩阵图已保存至: {filepath}")

plt.show()