import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from matplotlib import cm
from datetime import datetime


# ==========================================
# 0. 核心配置接口 (直接在这里控制视角和指标)
# ==========================================
MAIN_METRIC = 'ACC'
SECOND_METRIC = 'Runtime'
FIND_BEST_BY = MAIN_METRIC

# --- 3D 视角控制 ---
VIEW_ELEV = 35  # 仰角 (上下倾斜，一般 30-40 度最佳)
VIEW_AZIM = -150  # 方位角 (左右旋转)。你可以尝试改成 -45, 45, 135, -135 来寻找把 (6,6) 放在最后面的完美角度

# ==========================================
# 1. 读取真实数据与设置文件夹
# ==========================================
folder_path = 'results_c'  # 替换为你的文件
output_dir = 'double3D_results_AR_c'
os.makedirs(output_dir, exist_ok=True)

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
if not csv_files:
    print(f"No csv files found in {folder_path}!")

print(f"开始处理，共发现 {len(csv_files)} 个 CSV 文件...\n")

for csv_file in csv_files:
    file_name = os.path.basename(csv_file)
    file_name_no_ext = os.path.splitext(file_name)[0]

    try:
        df = pd.read_csv(csv_file)

        # ==========================================
        # 2. 自动寻优：锁定全局最优的 AnchorsRate
        # ==========================================
        df_mean = df.groupby(['AnchorsRate', 'K', 'Order', 'NumSampling'])[[MAIN_METRIC, SECOND_METRIC]].mean().reset_index()
        best_idx = df_mean[FIND_BEST_BY].idxmax()
        best_ar = df_mean.loc[best_idx, 'AnchorsRate']

        print(f"自动锁定最优底座: AnchorsRate (ar) = {best_ar}")
        df_plot = df_mean[df_mean['AnchorsRate'] == best_ar].copy()

        # ==========================================
        # 3. 动态构建 3D 画布与坐标系
        # ==========================================
        k_values = sorted(df_plot['K'].unique())
        num_k = len(k_values)

        plt.style.use('default')
        fig = plt.figure(figsize=(5 * num_k, 10))

        # --- 设置画布的最外层边框 ---
        fig.patch.set_linewidth(3)
        fig.patch.set_edgecolor('black')

        x_vals = sorted(df_plot['Order'].unique())
        y_vals = sorted(df_plot['NumSampling'].unique())
        X, Y = np.meshgrid(x_vals, y_vals)

        # 动态获取两个指标的极值
        m1_min, m1_max = df_plot[MAIN_METRIC].min(), df_plot[MAIN_METRIC].max()
        m2_min, m2_max = df_plot[SECOND_METRIC].min(), df_plot[SECOND_METRIC].max()

        baseline_m1 = m1_min * 0.98
        norm_m1 = plt.Normalize(m1_min, m1_max)
        norm_m2 = plt.Normalize(0, m2_max)

        # ==========================================
        # 4. 遍历 K 值，绘制上下两层 3D 柱状图
        # ==========================================
        for i, k_val in enumerate(k_values):
            df_k = df_plot[df_plot['K'] == k_val]

            Z_m1 = np.full(X.shape, np.nan)
            Z_m2 = np.full(X.shape, np.nan)

            for r in range(len(y_vals)):
                for c in range(len(x_vals)):
                    row = df_k[(df_k['Order'] == x_vals[c]) & (df_k['NumSampling'] == y_vals[r])]
                    if not row.empty:
                        Z_m1[r, c] = row[MAIN_METRIC].values[0]
                        Z_m2[r, c] = row[SECOND_METRIC].values[0]

            Z_m1 = np.nan_to_num(Z_m1, nan=m1_min)
            Z_m2 = np.nan_to_num(Z_m2, nan=0)

            xpos = X.flatten() - 0.3
            ypos = Y.flatten() - 0.3
            dx, dy = 0.6, 0.6

            # -----------------------------------------------------
            # 【上层】：主指标 3D 柱状图
            # -----------------------------------------------------
            ax1 = fig.add_subplot(2, num_k, i + 1, projection='3d')
            dz_m1 = Z_m1.flatten() - baseline_m1
            zpos_m1 = np.full_like(dz_m1, baseline_m1)
            colors_m1 = cm.GnBu(norm_m1(Z_m1.flatten()))

            ax1.bar3d(xpos, ypos, zpos_m1, dx, dy, dz_m1,
                      color=colors_m1, edgecolor='black', linewidth=0.3, alpha=0.95)

            # 直接使用你指定的固定视角
            ax1.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

            ax1.set_title(f'K = {k_val} ', pad=10, fontweight='bold', fontsize=14)

            ax1.set_xlabel('Relation order', labelpad=8)
            ax1.set_ylabel('Anchor configurations', labelpad=8)

            # 仅在最左侧标注 Z 轴单位说明，但保留所有子图的 Z 轴数字刻度
            if i == 0:
                # 1. 关闭默认的自动旋转
                ax1.zaxis.set_rotate_label(False)
                # 2. 强制将旋转角度设为 90 度（竖直）
                ax1.set_zlabel(MAIN_METRIC, labelpad=8, fontweight='bold', rotation=90)

            ax1.set_xticks(x_vals)
            ax1.set_yticks(y_vals)
            ax1.set_zlim(baseline_m1, m1_max * 1.02)

            # -----------------------------------------------------
            # 【下层】：副指标 3D 柱状图
            # -----------------------------------------------------
            ax2 = fig.add_subplot(2, num_k, i + 1 + num_k, projection='3d')
            dz_m2 = Z_m2.flatten()
            zpos_m2 = np.zeros_like(dz_m2)
            colors_m2 = cm.YlOrRd(norm_m2(Z_m2.flatten()))

            ax2.bar3d(xpos, ypos, zpos_m2, dx, dy, dz_m2,
                      color=colors_m2, edgecolor='black', linewidth=0.3, alpha=0.95)

            ax2.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

            # 修改坐标轴文字
            ax2.set_xlabel('Relation order', labelpad=8)
            ax2.set_ylabel('Anchor configurations', labelpad=8)

            # 仅在最左侧标注 Z 轴单位说明，同样保留所有子图的 Z 轴数字刻度
            if i == 0:
                ax2.set_zlabel(
                    f'{SECOND_METRIC} (s)' if 'Time' in SECOND_METRIC or 'Runtime' in SECOND_METRIC else SECOND_METRIC,
                    labelpad=8, fontweight='bold')

            ax2.set_xticks(x_vals)
            ax2.set_yticks(y_vals)
            ax2.set_zlim(0, m2_max * 1.05)

        # ==========================================
        # 5. 总标题与高清导出
        # ==========================================
        sm_m1 = cm.ScalarMappable(cmap='GnBu', norm=norm_m1)
        sm_m1.set_array([])
        cbar_ax1 = fig.add_axes([0.95, 0.55, 0.015, 0.3])
        fig.colorbar(sm_m1, cax=cbar_ax1)

        sm_m2 = cm.ScalarMappable(cmap='YlOrRd', norm=norm_m2)
        sm_m2.set_array([])
        cbar_ax2 = fig.add_axes([0.95, 0.10, 0.015, 0.3])
        label_m2 = f'{SECOND_METRIC} (s)' if 'Time' in SECOND_METRIC or 'Runtime' in SECOND_METRIC else SECOND_METRIC
        fig.colorbar(sm_m2, cax=cbar_ax2)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.05, hspace=0.1)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        param = f'AR{best_ar}_{timestamp}.pdf'
        for part in file_name_no_ext.split('_'):
            if part.startswith('data'):
                data_key = part
            else:
                data_key = file_name_no_ext
        filename = '_'.join([data_key, param])
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=600)

        print(f"3D 柱状图已保存至: {filepath}\n")

        # plt.show()

    except Exception as e:
        print(f"处理文件 {file_name} 时发生错误 {e}")