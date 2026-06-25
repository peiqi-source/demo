# =========================================================================
# 大规模可拓展性研究 (Scalability Study) 自动化测试脚本 [Python 版]
# 数据集: MNIST (5000 到 70000，步长 5000)
# 算法: MCLA, CSPA, HGPA, HBGF (基于顶刊 100 选 20 同步重采样机制)
# =========================================================================

import os
import time
import warnings
import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans
import ensembleclustering as CE

# 屏蔽烦人的 CSPA UserWarning 警告
warnings.filterwarnings('ignore', category=UserWarning)


# ==========================================
# 模块 1：对齐 MATLAB 的评价指标计算
# ==========================================
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = y_true.size
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / n
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='max')
    purity = np.sum(np.amax(cm, axis=0)) / n

    # 计算 Fscore 和 ARI
    sum_rows = np.sum(cm, axis=1)
    sum_cols = np.sum(cm, axis=0)
    comb_cm = cm * (cm - 1) / 2
    comb_rows = sum_rows * (sum_rows - 1) / 2
    comb_cols = sum_cols * (sum_cols - 1) / 2
    sum_comb_cm = np.sum(comb_cm)
    sum_comb_rows = np.sum(comb_rows)
    sum_comb_cols = np.sum(comb_cols)

    precision = sum_comb_cm / sum_comb_cols if sum_comb_cols > 0 else 0
    recall = sum_comb_cm / sum_comb_rows if sum_comb_rows > 0 else 0
    fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    ari = adjusted_rand_score(y_true, y_pred)

    return acc, nmi, purity, fscore, ari


# ==========================================
# 模块 2：实验全局设置
# ==========================================
results_dir = 'results_scalability_python'
details_dir = os.path.join(results_dir, 'Scalability_Details')
os.makedirs(details_dir, exist_ok=True)

data_dir = 'MNIST_Scalability_Data'  # 数据集存放目录
data_sizes = list(range(5000, 75000, 5000))
algorithms = ['hgpa', 'hbgf']
algorithms_display = [algo.upper() for algo in algorithms]

total_M = 100  # [新增]: 生成的基聚类大池子总数
M = 20  # 每次集成真正抽取并喂给算法的数量
num_seeds = 20  # 每个算法运行的种子数
CSPA_MAX_N = 15000  # [保护机制]: CSPA 样本数超过此值直接熔断

timestamp_global = time.strftime('%Y%m%d_%H%M%S')
master_csv_name = os.path.join(results_dir, f'Master_Scalability_Python_4Methods_{timestamp_global}.csv')

# 初始化 Master CSV 表头
header_master = ['Dataset_Size', 'Method', 'N_Samples',
                 'ACC(Mean±Std)', 'ACC(Max)', 'NMI(Mean±Std)', 'NMI(Max)',
                 'PUR(Mean±Std)', 'PUR(Max)', 'Fscore(Mean±Std)', 'Fscore(Max)',
                 'ARI(Mean±Std)', 'ARI(Max)', 'Runtime(s)']

with open(master_csv_name, 'w', encoding='utf-8') as f:
    f.write(','.join(header_master) + '\n')

print('====================================================================')
print('>>> 启动 Python 可拓展性研究 (含同步重采样机制): MNIST 5000 -> 70000')
print(f'>>> 对比算法包含: {", ".join(algorithms_display)}')
print(f'>>> 终极总表将保存至: {master_csv_name}')
print('====================================================================')

# ==========================================
# 模块 3：主循环 (遍历数据集规模)
# ==========================================
for d_size in data_sizes:
    size_str = str(d_size)
    file_name = os.path.join(data_dir, f'MNIST_{d_size}.mat')

    print('\n--------------------------------------------------------------------')
    print(f'>>> 正在处理: MNIST 规模 N = {size_str} ...')

    if not os.path.exists(file_name):
        print(f'    [警告] 未找到文件: {file_name}，跳过...')
        continue

    try:
        mat_data = sio.loadmat(file_name)
        X = mat_data.get('X') if 'X' in mat_data else mat_data.get('fea')
        Y = mat_data.get('Y') if 'Y' in mat_data else (mat_data.get('gt') if 'gt' in mat_data else mat_data.get('gnd'))

        if X is None or Y is None:
            print(f'    [警告] {file_name} 中未找到标准特征或标签，跳过...')
            continue

        X = X.astype(float)
        Y = Y.astype(int).flatten()
        N = X.shape[0]
        trueK = len(np.unique(Y))

        max_vals = np.max(X, axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1
        X_norm = X / max_vals
        X_norm = np.nan_to_num(X_norm)

    except Exception as e:
        print(f'    [错误] 读取文件 {file_name} 失败: {e}')
        continue

    # =====================================================================
    # [预处理]: 顶刊重采样范式 - 生成 100 个基聚类的大池子
    # =====================================================================
    print(f'    [预处理] 正在对 {N} 个样本执行 MiniBatchKMeans 生成 {total_M} 个基聚类大池...')
    base_clusters_pool = np.zeros((total_M, N), dtype=int)
    for m in range(total_M):
        kmeans = MiniBatchKMeans(n_clusters=trueK, n_init=1, max_iter=50, random_state=m)
        base_clusters_pool[m, :] = kmeans.fit_predict(X_norm)
    print('    [预处理] 基聚类大池生成完毕！进入同步重采样评估阶段...')

    res_pool = {algo: [] for algo in algorithms}
    raw_detail_list = []

    # --- 执行 20 次 Seed 测试 ---
    for s_idx in range(1, num_seeds + 1):
        seed = d_size + s_idx
        np.random.seed(seed)

        # ---------------------------------------------------------
        # 【核心】：同步重采样！从 100 个里面无放回抽出 20 个。
        # 此 Seed 下，MCLA, CSPA 等 4 个算法将共享这挑出来的 20 个！
        # ---------------------------------------------------------
        sampled_indices = np.random.choice(total_M, M, replace=False)
        sampled_base_clusters = base_clusters_pool[sampled_indices, :]

        for algo, algo_display in zip(algorithms, algorithms_display):
            is_oom = False
            try:
                if algo == 'cspa' and N > CSPA_MAX_N:
                    raise MemoryError(f"数据量 ({N}) 超过 CSPA 承受极限 ({CSPA_MAX_N})，主动熔断。")

                t0 = time.time()
                # 传入的是抽样后的 20 个基聚类
                pred_y = CE.cluster_ensembles(sampled_base_clusters, solver=algo)
                t_run = time.time() - t0

                acc, nmi, pur, fsc, ari = calculate_metrics(Y, pred_y)

                res_pool[algo].append([acc, nmi, pur, fsc, ari, t_run])
                raw_detail_list.append([size_str, algo_display, seed, acc, nmi, pur, fsc, ari, t_run])

            except Exception as e:
                if s_idx == 1:
                    print(f'    [熔断] {algo_display} 算法在 {size_str} 样本上异常: {e}')
                res_pool[algo].append([np.nan] * 6)
                raw_detail_list.append([size_str, algo_display, seed, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    # --- 结算汇总表 ---
    with open(master_csv_name, 'a', encoding='utf-8') as f:
        for algo, algo_display in zip(algorithms, algorithms_display):
            cur_res = np.array(res_pool[algo], dtype=float)

            if np.all(np.isnan(cur_res[:, 0])):
                f.write(f'{size_str},{algo_display},{N},' + 'OOM,' * 10 + '\n')
            else:
                avg_m = np.nanmean(cur_res, axis=0)
                std_m = np.nanstd(cur_res, axis=0)  # numpy 默认 ddof=0, 与 MATLAB 行为一致
                max_m = np.nanmax(cur_res, axis=0)

                str_stats = [f"{avg_m[i]:.4f}±{std_m[i]:.4f}" for i in range(5)]

                row_str = f"{size_str},{algo_display},{N}," + \
                          f"{str_stats[0]},{max_m[0]:.4f},{str_stats[1]},{max_m[1]:.4f}," + \
                          f"{str_stats[2]},{max_m[2]:.4f},{str_stats[3]},{max_m[3]:.4f}," + \
                          f"{str_stats[4]},{max_m[4]:.4f},{avg_m[5]:.4f}\n"
                f.write(row_str)

                print(f'  |-- [{algo_display:<6}] ACC: {str_stats[0]} | T: {avg_m[5]:.4f}s')

    # --- 保存详细记录 ---
    detail_csv = os.path.join(details_dir, f'Scalability_Details_{size_str}_{timestamp_global}.csv')
    df_detail = pd.DataFrame(raw_detail_list,
                             columns=['Dataset_Size', 'Method', 'Seed', 'ACC', 'NMI', 'PUR', 'Fscore', 'ARI',
                                      'Runtime'])
    df_detail.to_csv(detail_csv, index=False)

print('====================================================================')
print(f'[SUCCESS] Python 算法拓展测试 (同步重采样版) 结束！\n汇总表已保存: {master_csv_name}')
print('====================================================================')