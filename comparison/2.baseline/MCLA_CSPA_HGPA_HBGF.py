import os
import numpy as np
import time
import scipy.io as sio
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
import ensembleclustering as CE


# ==========================================
# 模块 1：对齐 MATLAB 的评价指标计算
# ==========================================
def calculate_metrics(y_true, y_pred):
    """
    计算完全等价于 MATLAB ClusteringMeasure4 的 ACC, NMI, Purity, Fscore, ARI
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = y_true.size

    # --- 0. 生成混淆矩阵 ---
    cm = confusion_matrix(y_true, y_pred)

    # --- 1. 计算 ACC (使用匈牙利算法) ---
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / n

    # --- 2. 计算 NMI ---
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='max')

    # --- 3. 计算 Purity ---
    purity = np.sum(np.amax(cm, axis=0)) / n

    # --- 4. 计算 Fscore (基于配对) ---
    TP = np.sum(cm * (cm - 1)) / 2
    sum_col = np.sum(cm, axis=0)
    TP_FP = np.sum(sum_col * (sum_col - 1)) / 2
    sum_row = np.sum(cm, axis=1)
    TP_FN = np.sum(sum_row * (sum_row - 1)) / 2

    P = TP / TP_FP if TP_FP > 0 else 0.0
    R = TP / TP_FN if TP_FN > 0 else 0.0
    fscore = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0

    # --- 5. 计算 ARI ---
    ari = adjusted_rand_score(y_true, y_pred)

    return acc, nmi, purity, fscore, ari


# ==========================================
# 模块 2：封装聚类集成共识生成 (带随机种子)
# ==========================================
def run_consensus_clustering(base_labels, true_labels, solver_name, seed):
    """
    运行指定的集成算法，并返回 5 项评价分数
    """
    # 强制设置全局 numpy 随机种子，确保可重复性
    np.random.seed(seed)

    n_class = len(np.unique(true_labels))

    # 调用 ClusterEnsembles 库进行融合
    consensus_labels = CE.cluster_ensembles(base_labels, solver=solver_name, nclass=n_class)

    # 计算性能指标
    acc, nmi, purity, fscore, ari = calculate_metrics(true_labels, consensus_labels)

    return acc, nmi, purity, fscore, ari


# ==========================================
# 模块 3：主控程序 (遍历 12 个数据集)
# ==========================================
def main():
    mat_folder = 'Base_LabelsPool_MAT_Allsample'
    dataset_names = ['Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT',
                     'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits']

    NUM_SEEDS = 20  # 设定的随机种子数量
    MAX_N = 20000
    # algorithms = ['mcla', 'cspa', 'hgpa', 'hbgf']  # 需要跑的集成算法
    algorithms = ['hgpa']  # 需要跑的集成算法


    results_list = []

    print("======================================================")
    print(f">>> 开始执行共识聚类 Pipeline (带 Mean±Std 统计)")
    print(f">>> 每个算法独立运行 {NUM_SEEDS} 个随机种子")
    print("======================================================")

    for data_idx in range(1, 13):
        file_name = f'LabelsPool_{dataset_names[data_idx - 1]}.mat'
        file_path = os.path.join(mat_folder, file_name)

        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，跳过该数据集。")
            continue

        print(f"\n>>> 正在处理: Dataset {data_idx:02d} ({dataset_names[data_idx - 1]}) ...")

        mat_data = sio.loadmat(file_path)
        base_labels = np.asarray(mat_data['base_labels'], dtype=float)
        ground_truth = np.asarray(mat_data['Y'], dtype=int).flatten()

        N_samples = len(ground_truth)
        current_res = {'Dataset_ID': data_idx, 'Dataset_Name': dataset_names[data_idx - 1], 'N_samples': N_samples}

        # 遍历每一种算法
        for algo in algorithms:
            algo_upper = algo.upper()
            print(f"    - 正在运行 {algo_upper:<4s} ...", end=" ", flush=True)

            # CSPA 内存溢出保护：超过一定样本直接熔断
            if algo == 'cspa' and N_samples > MAX_N:
                print(f" [触发熔断: 样本数 N={N_samples} 过大自动跳过]")
                current_res.update({
                    f'{algo_upper}_ACC(Mean±Std)': 'OOM',
                    f'{algo_upper}_NMI(Mean±Std)': 'OOM',
                    f'{algo_upper}_Purity(Mean±Std)': 'OOM',
                    f'{algo_upper}_Fscore(Mean±Std)': 'OOM',
                    f'{algo_upper}_ARI(Mean±Std)': 'OOM',
                    f'{algo_upper}_RUNTIME(s)': 'OOM'
                })
                continue

            # 用于存储 20 次运行结果的字典
            metrics_pool = {'ACC': [], 'NMI': [], 'Purity': [], 'Fscore': [], 'ARI': [], 'RUNTIME': []}
            is_oom = False

            M = 20  # 假设你每次只想抽取 15 个基聚类

            # 跑 20 个随机种子
            for seed in range(1, NUM_SEEDS + 1):
                try:
                    t0 = time.time()

                    # 强行设定随机种子，确保每次抽样的可追溯性
                    np.random.seed(seed)

                    # [新增]: 模拟从 100 个池子中随机无放回抽取 M 个基聚类
                    # 注意: Python 的 shape 可能是 (N_samples, M_total) 或相反，请根据你的 mat 维度自行调整 axis
                    total_M = base_labels.shape[1]  # 假设基聚类在列上
                    selected_indices = np.random.choice(total_M, M, replace=False)
                    selected_base_labels = base_labels[:, selected_indices]

                    # 使用抽样后的基聚类进行集成
                    acc, nmi, pur, fsc, ari = run_consensus_clustering(selected_base_labels, ground_truth,
                                                                       solver_name=algo, seed=seed)
                    runtime = time.time() - t0

                    # ... 后续追加到 metrics_pool ...

                    metrics_pool['ACC'].append(acc)
                    metrics_pool['NMI'].append(nmi)
                    metrics_pool['Purity'].append(pur)
                    metrics_pool['Fscore'].append(fsc)
                    metrics_pool['ARI'].append(ari)
                    metrics_pool['RUNTIME'].append(runtime)

                except Exception as e:
                    is_oom = True
                    print(f" [Seed {seed} 异常崩溃/OOM: {e}，该算法直接熔断!]")
                    break  # 只要出现一次失败，直接跳出 20 次循环

            # 结果结算 (Mean ± Std)
            if is_oom:
                current_res.update({
                    f'{algo_upper}_ACC(Mean±Std)': 'OOM',
                    f'{algo_upper}_NMI(Mean±Std)': 'OOM',
                    f'{algo_upper}_Purity(Mean±Std)': 'OOM',
                    f'{algo_upper}_Fscore(Mean±Std)': 'OOM',
                    f'{algo_upper}_ARI(Mean±Std)': 'OOM',
                    f'{algo_upper}_RUNTIME(s)': 'OOM'
                })
            else:
                # 正常跑完 20 次，计算无偏标准差 (ddof=1 严格对齐 MATLAB)
                avg_metrics = {k: np.mean(v) for k, v in metrics_pool.items()}
                std_metrics = {k: np.std(v, ddof=1) for k, v in metrics_pool.items() if k != 'RUNTIME'}

                # 格式化拼接为 'Mean±Std'
                current_res.update({
                    f'{algo_upper}_ACC(Mean±Std)': f"{avg_metrics['ACC']:.4f}±{std_metrics['ACC']:.4f}",
                    f'{algo_upper}_NMI(Mean±Std)': f"{avg_metrics['NMI']:.4f}±{std_metrics['NMI']:.4f}",
                    f'{algo_upper}_Purity(Mean±Std)': f"{avg_metrics['Purity']:.4f}±{std_metrics['Purity']:.4f}",
                    f'{algo_upper}_Fscore(Mean±Std)': f"{avg_metrics['Fscore']:.4f}±{std_metrics['Fscore']:.4f}",
                    f'{algo_upper}_ARI(Mean±Std)': f"{avg_metrics['ARI']:.4f}±{std_metrics['ARI']:.4f}",
                    f'{algo_upper}_RUNTIME(s)': round(avg_metrics['RUNTIME'], 4)
                })

                print(
                    f"完成! ACC: {current_res[f'{algo_upper}_ACC(Mean±Std)']} | NMI: {current_res[f'{algo_upper}_NMI(Mean±Std)']} "
                    f"| PUR: {current_res[f'{algo_upper}_Purity(Mean±Std)']} | Fscore: {current_res[f'{algo_upper}_Fscore(Mean±Std)']} "
                    f"| ARI: {current_res[f'{algo_upper}_ARI(Mean±Std)']} | T: {avg_metrics['RUNTIME']:.4f}s")

        # 保存当前数据集成绩
        results_list.append(current_res)

    # 4. 将所有结果输出为 CSV 文件
    if results_list:
        # 动态生成带有 Mean±Std 的表头顺序
        columns_order = ['Dataset_ID', 'Dataset_Name', 'N_samples']
        for algo in algorithms:
            algo_upper = algo.upper()
            columns_order.extend([
                f'{algo_upper}_ACC(Mean±Std)', f'{algo_upper}_NMI(Mean±Std)',
                f'{algo_upper}_Purity(Mean±Std)', f'{algo_upper}_Fscore(Mean±Std)',
                f'{algo_upper}_ARI(Mean±Std)', f'{algo_upper}_RUNTIME(s)'
            ])

        df_results = pd.DataFrame(results_list, columns=columns_order)
        csv_filename = "Comparison_MCLA_CSPA_HGPA_HBGF_5Metrics_20Seeds_MeanStd.csv"

        # 核心：使用 utf_8_sig 编码保存，强力防止 Excel 出现 "±" 乱码
        df_results.to_csv(csv_filename, index=False, encoding='utf_8_sig')

        print("\n======================================================")
        print(f">>> 实验运行完毕！综合平均结果已保存至: {csv_filename}")
        print("======================================================")

if __name__ == '__main__':
    main()