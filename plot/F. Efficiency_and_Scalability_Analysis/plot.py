import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ==============================================================================
# 1. 顶刊样式全局设置 (Times New Roman, 大字号, 矢量图标准)
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 11,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

# ==============================================================================
# 2. 数据读取与清洗 (处理 'Mean±Std', 'OOM', 解决GBK编码问题)
# ==============================================================================
file_path = 'Master_Scalability_MNIST.csv'
# 必须使用 GBK 编码读取，防止 ± 符号报错
df = pd.read_csv(file_path, encoding='GBK')

def extract_mean(val):
    if pd.isna(val) or 'OOM' in str(val):
        return np.nan
    if isinstance(val, str) and '±' in val:
        return float(val.split('±')[0])
    try:
        return float(val)
    except:
        return np.nan

df['ACC_Mean'] = df['ACC(Mean±Std)'].apply(extract_mean)
df['Runtime'] = df['Runtime(s)'].apply(lambda x: np.nan if 'OOM' in str(x) else float(x))

methods = df['Method'].unique()
x_sizes = sorted(df['Dataset_Size'].dropna().unique())

# ==============================================================================
# 3. 颜色、线宽与标记策略设计 (AHD线宽加倍)
# ==============================================================================
colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'h', '<', '>']

style_dict = {}
for i, m in enumerate(methods):
    if m.upper() == 'AHD':
        # [修改一]: AHD 的线宽设为 3.0，并作为视觉焦点
        style_dict[m] = {'color': '#d62728', 'marker': '*', 'markersize': 12, 'linewidth': 3.0, 'zorder': 10}
    else:
        # [修改一]: 其他所有对比算法线宽设为 1.5 (AHD的一半)
        style_dict[m] = {'color': colors[i], 'marker': markers[i % len(markers)], 'markersize': 8, 'linewidth': 1.5, 'zorder': 5}

# ==============================================================================
# 4. 绘制 Figure 1: 准确率 (ACC) 随数据量变化图
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(8, 6))

for method in methods:
    df_m = df[df['Method'] == method].sort_values('Dataset_Size')
    ax1.plot(df_m['Dataset_Size'], df_m['ACC_Mean'],
             label=method,
             color=style_dict[method]['color'],
             marker=style_dict[method]['marker'],
             markersize=style_dict[method]['markersize'],
             linewidth=style_dict[method]['linewidth'],
             zorder=style_dict[method]['zorder'])

ax1.set_xlabel('Number of Samples ($N$)')
ax1.set_ylabel('Clustering Accuracy (ACC)')
ax1.set_xlim([min(x_sizes), max(x_sizes)])
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.)
plt.tight_layout()

fig1.savefig('Scalability_ACC_vs_N.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.close(fig1)

# ==============================================================================
# 5. 绘制 Figure 2: 运行时间 (Runtime) 随数据量变化图 (2为底对数坐标)
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

for method in methods:
    df_m = df[df['Method'] == method].sort_values('Dataset_Size')
    ax2.plot(df_m['Dataset_Size'], df_m['Runtime'],
             label=method,
             color=style_dict[method]['color'],
             marker=style_dict[method]['marker'],
             markersize=style_dict[method]['markersize'],
             linewidth=style_dict[method]['linewidth'],
             zorder=style_dict[method]['zorder'])

ax2.set_xlabel('Number of Samples ($N$)')
ax2.set_ylabel('Execution Time (Seconds)')
ax2.set_xlim([min(x_sizes), max(x_sizes)])

# [修改二]: 将 Y 轴设置为以 2 为底的对数坐标系
ax2.set_yscale('log', base=2)
# 使用 LogLocator 专门优化 2 为底时的刻度显示，确保 Y 轴会出现 2^0, 2^1, 2^2... 的漂亮刻度
ax2.yaxis.set_major_locator(LogLocator(base=2.0))

ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.)
plt.tight_layout()

fig2.savefig('Scalability_Runtime_vs_N_Log2.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.close(fig2)

print("画图完成！已生成 Scalability_ACC_vs_N.pdf 和 Scalability_Runtime_vs_N_Log2.pdf")