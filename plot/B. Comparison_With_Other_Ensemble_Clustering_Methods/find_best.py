import pandas as pd
import os
import glob

def process_single_csv(csv_file, target_metric):
    # 1. 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # ==========================================
    # 搜索一：寻找“平均指标”最高的超参数组合 (全局最优/最稳)
    # ==========================================
    # 动态传入你选择的 metric
    df_mean = df.groupby(['AnchorsRate', 'K', 'Order', 'NumSampling'])[[target_metric, 'Runtime']].mean().reset_index()
    best_mean_idx = df_mean[target_metric].idxmax()

    # 提取平均最优的超参数和对应的最高平均指标
    mean_ar = df_mean.loc[best_mean_idx, 'AnchorsRate']
    mean_k = df_mean.loc[best_mean_idx, 'K']
    mean_ord = df_mean.loc[best_mean_idx, 'Order']
    mean_ns = df_mean.loc[best_mean_idx, 'NumSampling']
    best_mean_val = df_mean.loc[best_mean_idx, target_metric]
    best_mean_runtime = df_mean.loc[best_mean_idx, 'Runtime']

    # 过滤出这组配置下的所有 Seed 数据，用于核对
    df_mean_check = df[(df['AnchorsRate'] == mean_ar) &
                       (df['K'] == mean_k) &
                       (df['Order'] == mean_ord) &
                       (df['NumSampling'] == mean_ns)]

    # ==========================================
    # 搜索二：寻找“单次指标”最高的那一行 (巅峰/历史最高)
    # ==========================================
    best_single_idx = df[target_metric].idxmax()

    # 提取单次最高的超参数、具体的 Seed 以及巅峰指标
    single_ar = df.loc[best_single_idx, 'AnchorsRate']
    single_k = df.loc[best_single_idx, 'K']
    single_ord = df.loc[best_single_idx, 'Order']
    single_ns = df.loc[best_single_idx, 'NumSampling']
    single_seed = df.loc[best_single_idx, 'Seed']
    best_single_val = df.loc[best_single_idx, target_metric]
    best_single_runtime = df.loc[best_single_idx, 'Runtime']

    # ==========================================
    # 打印真相
    # ==========================================
    print("==================================================")
    print(f">>> 全局平均最优 (基于所有 Seed 的平均 {target_metric})")
    print(f"最稳参数组合: ar={mean_ar}, k={mean_k}, ord={mean_ord}, ns={mean_ns}")
    print(f"全局最高平均 {target_metric} = {best_mean_val:.6f}")
    print(f"对应的 Runtime = {best_mean_runtime:.6f}")
    print("--------------------------------------------------")
    print(f"Pandas 实际找到了该配置下的 {len(df_mean_check)} 行 Seed 数据：\n")

    # 动态打印出你选择的那个指标列
    print(df_mean_check[['Seed', target_metric, 'Runtime']].to_string(index=False))

    print(f"\n[验证] {target_metric} 实时求和: {df_mean_check[target_metric].sum():.6f} | 实时平均: {df_mean_check[target_metric].mean():.6f}")
    print(f"[验证] Runtime 实时求和: {df_mean_check['Runtime'].sum():.6f} | 实时平均: {df_mean_check['Runtime'].mean():.6f}")
    print("==================================================\n")

    print("==================================================")
    print(f">>> 历史单次最高 (表格中绝对值最高的那一行数据)")
    print(f"巅峰参数组合: ar={single_ar}, k={single_k}, ord={single_ord}, ns={single_ns}")
    print(f"幸运 Seed : {single_seed}")
    print(f"巅峰单次 {target_metric} = {best_single_val:.6f}")
    print(f"对应的 Runtime = {best_single_runtime:.6f}")
    print("==================================================")

    # 检查全局是否有跑崩的 NaN
    nan_count = df[target_metric].isna().sum()
    if nan_count > 0:
        print(f"\n警告: 整个 CSV 中发现了 {nan_count} 个 {target_metric} 的 NaN (空值)，请注意排查。")


def process_all_csv(folder_path, out_file, metric):
    # 1. 获取文件夹下所有的 csv 文件
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if not csv_files:
        print(f"警告：在文件夹 '{folder_path}' 中没有找到任何 .csv 文件！")
        return

    print(f"开始处理，共发现 {len(csv_files)} 个 CSV 文件...\n")

    # 用于存储所有文件的解析结果
    all_results = []

    # 2. 遍历每个文件
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)

        try:
            df = pd.read_csv(csv_file)

            # 检查目标指标是否在表格中
            if metric not in df.columns:
                print(f"跳过文件 {file_name}: 找不到列 '{metric}'")
                continue

            # ==========================================
            # 搜索一：寻找“平均指标”最高的超参数组合 (全局最优/最稳)
            # ==========================================
            df_mean = df.groupby(['AnchorsRate', 'K', 'Order', 'NumSampling'])[[metric, 'Runtime']].mean().reset_index()
            best_mean_idx = df_mean[metric].idxmax()

            mean_ar = df_mean.loc[best_mean_idx, 'AnchorsRate']
            mean_k = df_mean.loc[best_mean_idx, 'K']
            mean_ord = df_mean.loc[best_mean_idx, 'Order']
            mean_ns = df_mean.loc[best_mean_idx, 'NumSampling']
            best_mean_val = df_mean.loc[best_mean_idx, metric]
            best_mean_runtime = df_mean.loc[best_mean_idx, 'Runtime']

            # ==========================================
            # 搜索二：寻找“单次指标”最高的那一行 (巅峰/历史最高)
            # ==========================================
            best_single_idx = df[metric].idxmax()

            single_ar = df.loc[best_single_idx, 'AnchorsRate']
            single_k = df.loc[best_single_idx, 'K']
            single_ord = df.loc[best_single_idx, 'Order']
            single_ns = df.loc[best_single_idx, 'NumSampling']
            single_seed = df.loc[best_single_idx, 'Seed']
            best_single_val = df.loc[best_single_idx, metric]
            best_single_runtime = df.loc[best_single_idx, 'Runtime']

            # 检查 NaN
            nan_count = df[metric].isna().sum()

            # 3. 将该文件的所有关键信息打包为字典
            file_result = {
                'File Name': file_name,
                'Target Metric': metric,

                # 全局平均最优信息
                'Best Mean AR': mean_ar,
                'Best Mean K': mean_k,
                'Best Mean Order': mean_ord,
                'Best Mean NS': mean_ns,
                f'Global Max Mean {metric}': best_mean_val,
                'Mean Runtime': best_mean_runtime,

                # 历史单次最高信息
                'Best Single AR': single_ar,
                'Best Single K': single_k,
                'Best Single Order': single_ord,
                'Best Single NS': single_ns,
                'Lucky Seed': single_seed,
                f'Peak Single {metric}': best_single_val,
                'Single Runtime': best_single_runtime,

                # 数据健康状况
                'NaN Count': nan_count
            }

            all_results.append(file_result)
            print(f"成功处理: {file_name}")

        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")

    # 4. 将所有结果转换为 DataFrame 并输出为表格
    if all_results:
        summary_df = pd.DataFrame(all_results)

        # 根据后缀名自动选择保存为 CSV 还是 Excel
        if out_file.endswith('.xlsx'):
            # 注意：保存为 xlsx 需要安装 openpyxl 库 (pip install openpyxl)
            summary_df.to_excel(out_file, index=False)
        else:
            # 默认保存为 CSV，加上 utf-8-sig 防止在 Excel 中打开时中文乱码
            summary_df.to_csv(out_file, index=False, encoding='utf-8-sig')

        print("\n==================================================")
        print(f"所有文件处理完毕！共汇总了 {len(all_results)} 个文件的结果。")
        print(f"汇总表格已成功保存至: {out_file}")
        print("==================================================")

def process_Kmeans(folder_path, out_file, metric):
    # 1. 获取文件夹下所有的 csv 文件
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if not csv_files:
        print(f"警告：在文件夹 '{folder_path}' 中没有找到任何 .csv 文件！")
        return

    print(f"开始处理，共发现 {len(csv_files)} 个 CSV 文件...\n")

    # 用于存储所有文件的解析结果
    all_results = []

    # 2. 遍历每个文件
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)

        try:
            df = pd.read_csv(csv_file)

            # 检查目标指标是否在表格中
            if metric not in df.columns:
                print(f"跳过文件 {file_name}: 找不到列 '{metric}'")
                continue

            # ==========================================
            # 搜索一：寻找“平均指标”最高的超参数组合 (全局最优/最稳)
            # ==========================================
            df_mean = df.groupby("DatasetID")[[metric]].mean().reset_index()
            mean = df_mean.loc[0, metric]

            # ==========================================
            # 搜索二：寻找“单次指标”最高的那一行 (巅峰/历史最高)
            # ==========================================
            best_idx = df[metric].idxmax()
            seed = df.loc[best_idx, 'Seed']
            best = df.loc[best_idx, metric]

            # 检查 NaN
            nan_count = df[metric].isna().sum()

            # 3. 将该文件的所有关键信息打包为字典
            file_result = {
                'File Name': file_name,
                'Target Metric': metric,

                # 全局平均最优信息
                'Mean': mean,

                # 历史单次最高信息
                'Best Seed': seed,
                'Best': best,

                # 数据健康状况
                'NaN Count': nan_count
            }

            all_results.append(file_result)
            print(f"成功处理: {file_name}")

        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")

    # 4. 将所有结果转换为 DataFrame 并输出为表格
    if all_results:
        summary_df = pd.DataFrame(all_results)

        # 根据后缀名自动选择保存为 CSV 还是 Excel
        if out_file.endswith('.xlsx'):
            # 注意：保存为 xlsx 需要安装 openpyxl 库 (pip install openpyxl)
            summary_df.to_excel(out_file, index=False)
        else:
            # 默认保存为 CSV，加上 utf-8-sig 防止在 Excel 中打开时中文乱码
            summary_df.to_csv(out_file, index=False, encoding='utf-8-sig')

        print("\n==================================================")
        print(f"所有文件处理完毕！共汇总了 {len(all_results)} 个文件的结果。")
        print(f"汇总表格已成功保存至: {out_file}")
        print("==================================================")


# 执行主函数
if __name__ == '__main__':
    # ==========================================
    # input_folder = 'results/results_kmeans'  # 输入：包含多个 csv 的文件夹路径
    # target_metric = 'NMI'  # 寻优的指标列名 (例如: 'ACC', 'NMI', 'Purity', 'Fscore')
    # output_file = "".join(['SummaryResults_KMeans_', target_metric, '.csv'])  # 输出：汇总的表格文件名 (如果是保存为Excel，请改为 .xlsx)
    # process_Kmeans(input_folder, output_file, target_metric)
    # ==========================================

    # ==========================================
    csv_file = 'F:\学习\MATLAB\demo\plot\D. Hyperparameter_Sensitivity_Analysis\para_small_data1_20260603_102921.csv'
    target_metric = 'ACC'     # 在这里填入你想寻优的指标列名 (例如: 'ACC', 'NMI', 'Purity', 'Fscore')
    process_single_csv(csv_file, target_metric)
    # ==========================================

    # ==========================================
    # input_folder = 'results/results_fix_newdatasets'  # 输入：包含多个 csv 的文件夹路径
    # output_file = 'results/summary_results_ACC_newdatasets.csv'  # 输出：汇总的表格文件名 (如果是保存为Excel，请改为 .xlsx)
    # target_metric = 'ACC'  # 寻优的指标列名 (例如: 'ACC', 'NMI', 'Purity', 'Fscore')
    # process_all_csv(input_folder, output_file, target_metric)
    # ==========================================
