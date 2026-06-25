import os
import re
import ensembleclustering as CE

# 1. 找到库的根目录（不再是单个文件）
lib_dir = os.path.dirname(CE.__file__)
print(">>>正在运行的是 v2 ！！！")
print(f">>> 正在扫描并修复库目录: {lib_dir}")

# 2. 这是极其强大的安全转换器
# 不管原来是稀疏矩阵、旧版 np.matrix 还是 ndarray，统统转化为现代系统支持的 ndarray
safe_replace = r'(__import__("numpy").asarray(\1.todense()) if hasattr(\1, "todense") else __import__("numpy").asarray(\1))'

# 3. 遍历目录下所有的 Python 文件
for filename in os.listdir(lib_dir):
    if filename.endswith('.py'):
        filepath = os.path.join(lib_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        # 将所有的 .toarray() 进行安全替换
        new_code = re.sub(r'([a-zA-Z0-9_]+)\.toarray\(\)', safe_replace, code)

        # 以防万一，把 .todense() 也加入安全保护
        new_code = re.sub(r'([a-zA-Z0-9_]+)\.todense\(\)', safe_replace, new_code)

        if code != new_code:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_code)
            print(f"成功修复底层文件: {filename}")

print(">>> 地毯式修复完成！这下真的无敌了。")