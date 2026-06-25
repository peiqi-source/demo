import scipy.io
import numpy as np
import pandas as pd  # <--- 修复报错1：必须包含这一行
from sklearn.datasets import fetch_openml, fetch_covtype
import os

# 网络代理设置（若下载速度慢或仍断连，请取消注释并确认端口准确）
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'


def process_and_save(X, y, name):
    """通用的预处理和保存函数"""
    try:
        # 使用 pandas 自动对齐所有文本/数字标签，映射为 1, 2, 3... 的整数
        y_encoded = pd.Categorical(y).codes + 1
        y_encoded = y_encoded.reshape(-1, 1)

        # 确保特征矩阵是纯数值型，防止影响后续距离计算
        X_num = np.array(X, dtype=np.float64)

        # 保存为 .mat 文件
        file_name = f'{name}_raw.mat'
        scipy.io.savemat(file_name, {'X': X_num, 'Y': y_encoded})
        print(f"  [√] {name} 保存成功！特征: {X_num.shape}, 标签簇数: {len(np.unique(y_encoded))} -> {file_name}")
    except Exception as e:
        print(f"  [x] 保存 {name} 时出错: {e}")


def process_and_save_scalability(X, y, file_path):
    """通用的预处理和保存函数"""
    try:
        # 使用 pandas 自动对齐所有文本/数字标签，映射为 1, 2, 3... 的整数
        y_encoded = pd.Categorical(y).codes + 1
        y_encoded = y_encoded.reshape(-1, 1)

        # 确保特征矩阵是纯数值型，防止影响后续距离计算
        X_num = np.array(X, dtype=np.float64)

        # 保存为 .mat 文件
        scipy.io.savemat(file_path, {'X': X_num, 'Y': y_encoded})
        print(f"  [√] 保存成功！特征: {X_num.shape}, 标签簇数: {len(np.unique(y_encoded))} -> {file_path}")
    except Exception as e:
        print(f"  [x] 保存失败: {e}")



def download_all_datasets():
    # 修复报错2：全部替换为绝对数字 ID，彻底绕过易断连的搜索 API！
    openml_datasets = {
        # 'LS': 182,  # Landsat Satellite
        # 'ODR': 28,  # optdigits
        # 'PD': 32,  # pendigits
        # 'USPS': 41082,  # usps
        # 'ISOLET': 300,  # isolet
        # 'LR': 6,  # letter
        # 'SPF': 1504,  # steel-plates-fault
        # 'IS': 36,  # segment
        # 'VS': 54,  # vehicle
        # 'Semeion': 1501,  # semeion
        'MNIST': 554  # mnist_784
    }

    print("=== 开始批量下载标准基准数据集 ===")
    for name, data_id in openml_datasets.items():
        print(f"\n正在下载 {name} (Data ID: {data_id})...")
        try:
            # 直接使用绝对 ID 拉取，稳定且快速
            data = fetch_openml(data_id=data_id, as_frame=False, parser='auto')
            process_and_save(data.data, data.target, name)
        except Exception as e:
            print(f"  [x] 下载 {name} 失败: {e}")

    # print("\n正在下载 FCT (Forest Covertype, 约 58 万条数据)...")
    # try:
    #     fct_data = fetch_covtype()
    #     process_and_save(fct_data.data, fct_data.target, 'FCT')
    # except Exception as e:
    #     print(f"  [x] 下载 FCT 失败: {e}")

def create_fct_subset():
    print("正在加载全量 FCT 数据集 (FCT_raw.mat)，这可能需要几秒钟...")
    try:
        data = scipy.io.loadmat('FCT_raw.mat')
        X_full = data['X']
        Y_full = data['Y']
    except FileNotFoundError:
        print("错误：找不到 FCT_raw.mat，请先运行之前的批量下载脚本！")
        return

    print(f"原始数据加载成功！特征形状: {X_full.shape}, 标签形状: {Y_full.shape}")

    # 论文设定的每个类别的抽样数量
    samples_per_class = 540
    num_classes = 7

    X_sampled_list = []
    Y_sampled_list = []

    # 设置随机种子以保证结果可复现
    # 这样每次运行抽出来的 3780 条数据都是固定的，方便调试聚类算法
    np.random.seed(42)

    print(f"\n开始执行分层随机抽样 (每类严格抽取 {samples_per_class} 条)...")

    for c in range(1, num_classes + 1):
        # 找到属于当前类别 c 的所有样本的行号 (索引)
        # 注意：Y_full 是 (N, 1) 的列向量，用 flatten() 展平以便寻找
        idx_c = np.where(Y_full.flatten() == c)[0]

        # 在该类别下，随机无放回地抽取 540 个索引
        sampled_idx = np.random.choice(idx_c, size=samples_per_class, replace=False)

        # 提取对应的数据并存入列表
        X_sampled_list.append(X_full[sampled_idx])
        Y_sampled_list.append(Y_full[sampled_idx])

        print(f"  - 第 {c} 类：原样本量 {len(idx_c):6d} -> 抽取 {len(sampled_idx)} 条")

    # 将 7 个列表垂直拼接成完整的矩阵
    X_subset = np.vstack(X_sampled_list)
    Y_subset = np.vstack(Y_sampled_list)

    # 【极其重要的一步】：打乱拼接后的数据顺序
    # 因为刚才是一类一类拼的，不打乱的话前540条全是第1类，这会严重影响后续聚类
    shuffle_idx = np.random.permutation(len(X_subset))
    X_subset = X_subset[shuffle_idx]
    Y_subset = Y_subset[shuffle_idx]

    # 保存为新的 .mat 文件
    output_file = 'FCT_3780.mat'
    scipy.io.savemat(output_file, {'X': X_subset, 'Y': Y_subset})

    print(f"\n=== 抽样完成！ ===")
    print(f"子集特征形状: {X_subset.shape}")
    print(f"子集标签形状: {Y_subset.shape}")
    print(f"已完美复刻论文实验设定，并成功保存为: {output_file}")

def create_mnist_subset():
    print("正在从 OpenML 下载完整的 MNIST 数据集 (70,000 条数据)，这可能需要几十秒...")
    try:
        # 554 是 mnist_784 的官方绝对数据 ID
        mnist = fetch_openml(data_id=554, as_frame=False, parser='auto')
    except Exception as e:
        print(f"下载失败，请检查网络或挂载代理: {e}")
        return

    # 提取特征并确保是纯数值型 float64
    X_full = np.array(mnist.data, dtype=np.float64)

    # 原始标签是 '0'-'9' 的字符串，这里用 pandas 自动映射为 0-9，然后 +1 变成 MATLAB 习惯的 1-10
    Y_full = pd.Categorical(mnist.target).codes + 1
    Y_full = Y_full.reshape(-1, 1)

    print(f"原始数据加载成功！特征形状: {X_full.shape}, 标签形状: {Y_full.shape}")

    # 论文设定：总共 5000 样本，10 个类，每类 500 个
    samples_per_class = 500
    num_classes = 10

    X_sampled_list = []
    Y_sampled_list = []

    # 设定随机种子 42，保证无论跑多少次，抽出来的 5000 张图片都是固定的同一批！
    np.random.seed(42)

    print(f"\n开始执行分层随机抽样 (每类严格抽取 {samples_per_class} 条)...")

    for c in range(1, num_classes + 1):
        # 找到属于当前类别 c（代表数字 c-1）的所有样本的行号
        idx_c = np.where(Y_full.flatten() == c)[0]

        # 在该类别下，随机无放回地抽取 500 个
        sampled_idx = np.random.choice(idx_c, size=samples_per_class, replace=False)

        # 将抽出的数据存入列表
        X_sampled_list.append(X_full[sampled_idx])
        Y_sampled_list.append(Y_full[sampled_idx])

        # 打印抽样日志（数字 1 对应手写体 '0'，数字 10 对应手写体 '9'）
        print(f"  - 第 {c} 类 (数字 {c - 1})：原样本量 {len(idx_c):4d} -> 随机抽取 {len(sampled_idx)} 条")

    # 将 10 个类的列表垂直拼接成完整的矩阵
    X_subset = np.vstack(X_sampled_list)
    Y_subset = np.vstack(Y_sampled_list)

    # 【极其重要】强制打乱所有数据的顺序，模拟真实的无序分布
    shuffle_idx = np.random.permutation(len(X_subset))
    X_subset = X_subset[shuffle_idx]
    Y_subset = Y_subset[shuffle_idx]

    # 保存为最终的 .mat 文件
    output_file = 'MNIST_5000.mat'
    scipy.io.savemat(output_file, {'X': X_subset, 'Y': Y_subset})

    print(f"\n=== 抽样完成！ ===")
    print(f"子集特征形状: {X_subset.shape}")
    print(f"子集标签形状: {Y_subset.shape}")
    print(f"已完美复刻论文实验设定，并成功保存为: {os.path.abspath(output_file)}")


def generate_mnist_scalability_subsets():
    # 1. 创建目标文件夹
    output_dir = "MNIST_Scalability_Data"
    os.makedirs(output_dir, exist_ok=True)
    print(f"已创建/确认目标文件夹: {output_dir}\n")

    # 2. 获取完整 MNIST 数据集
    print("正在从 OpenML 下载或加载 MNIST 完整数据集 (70000条)，请稍候...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X_full = mnist.data.values
    # 原始标签是 '0'-'9'，转为整数后加 1，变成 1-10 (符合 MATLAB 聚类习惯)
    Y_full = mnist.target.values.astype(int) + 1

    total_samples = len(X_full)
    print(f"数据集加载完毕！总样本数: {total_samples}")

    # 3. 设定采样规模: 5000 到 70000, 步长 5000
    sizes = range(5000, 70001, 5000)

    # 设定随机种子，保证每次运行生成的子集绝对一致（极其重要）
    np.random.seed(42)

    # 【核心逻辑】：生成一个包含所有索引的全局打乱数组 (0 到 69999)
    # 通过截取这个打乱数组的前 N 个元素，完美实现 "嵌套子集抽样"
    shuffled_indices = np.random.permutation(total_samples)

    for size in sizes:
        print(f"\n开始生成并提取 {size} 样本规模的数据集...")

        # 提取前 size 个打乱后的索引（保证了子集的嵌套性：1w包含5k，2w包含1w）
        sampled_idx = shuffled_indices[:size]

        X_subset = X_full[sampled_idx]
        Y_subset = Y_full[sampled_idx]

        # 拼装保存的文件名和路径
        file_name = f'MNIST_{size}.mat'
        file_path = os.path.join(output_dir, file_name)

        # 调用预处理与保存
        process_and_save_scalability(X_subset, Y_subset, file_path)

    print("\n 所有 14 个可扩展性数据集已成功生成完毕！")


if __name__ == "__main__":
    # download_all_datasets()
    # print("\n=== 所有数据集处理完毕！ ===")
    #
    # create_fct_subset()
    #
    # create_mnist_subset()

    generate_mnist_scalability_subsets()

