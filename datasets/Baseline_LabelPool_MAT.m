clear;
clc;
close all;

%% 1. 实验参数设置
datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                'MNIST', 'OptDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
data_list = 1:12;          % 遍历 12 个数据集
M = 100;                   % 生成 100 轮基础聚类
subsample_ratio = 1.0;     % 随机抽取 80% 的数据
TrueC = true

rng(2026);

% 创建一个专门存放结果的文件夹，保持目录整洁
saveDir = fullfile(pwd, 'Base_LabelsPool_MAT_TrueC');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

%% 2. 主循环开始
for data_idx = data_list
    dataname = datasetNames{data_idx};
    fprintf('\n======================================================\n');
    fprintf('>>> 正在处理数据集: %d / 12 , 数据集为: %s <<<\n', data_idx, dataname);
    
    % 加载数据
    [X, Y] = loaddata_small(data_idx);
    [num, dim] = size(X);
    c = length(unique(Y)); % 真实簇数
    
    % --- 预处理防坑 ---
    X = X ./ max(X, [], 2); % 归一化
    X(isnan(X)) = 0;        % 将是Nan的地方赋值为0，防止除以0产生NaN导致 kmeans 报错
    
    % 初始化我们要输出的标签矩阵 (M x N)，全部填满 NaN
    base_labels = NaN(M, num);
    
    % 计算每次需要抽取的样本数量
    num_samples = round(num * subsample_ratio);
    
    fprintf('    包含样本: %d, 特征: %d, 真实类数: %d\n', num, dim, c);
    fprintf('    开始生成 %d 轮基础标签 (每次抽取 %d 个样本)...\n', M, num_samples);
    
    % 进度条准备
    reverseStr = '';
    
    for m = 1:M
        % 为了保证实验可复现，可以设置随轮次变化的种子 (可选)
        rng(data_idx * 1000 + m); 
        
        % 1. 核心机制：随机抽取 80% 数据的行索引
        idx = randperm(num, num_samples);
        X_sub = X(idx, :);
        
        % 2. 多样性增强：让 K 值在真实类数附近波动 (Over-clustering)
        % 如果你只想用严格的真实类数，把这行改成 k_current = c; 即可
        if TrueC
            k_current = c;
        else
            k_current = randi([c, 2*c]); 
        end
        
        % 3. 运行基础聚类 (注意：传入的是 X_sub)
        [sub_labels, ~, ~, ~, ~] = litekmeans(X_sub, k_current);
        
        % 4. 结果回填：把算出来的标签填回 base_labels 对应的位置
        base_labels(m, idx) = sub_labels;
        
        % 打印进度
        msg = sprintf('    已完成: %d / %d 轮', m, M);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
    fprintf('\n');
    
    % 3. 保存当前数据集的结果到 .mat 文件
    matFilename = fullfile(saveDir, sprintf('LabelsPool_%s.mat', dataname));
    
    % 保存标签矩阵 base_labels 和 真实标签 Y 
    % (Y 留着给 Python 算 ACC 等评价指标用)
    save(matFilename, 'dataname', 'base_labels', 'Y');
    
    fprintf('>>> 数据集 %d 结果已保存至: %s\n', data_idx, matFilename);
end

fprintf('\n======================================================\n');
fprintf('🎉 所有 12 个数据集的基础标签生成完毕！\n');
fprintf('======================================================\n');