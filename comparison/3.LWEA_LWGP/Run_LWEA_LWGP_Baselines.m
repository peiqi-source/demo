% =========================================================================
% LWEA & LWGP 算法基线复现脚本 (顶刊 Mean±Std 格式 + 5指标全面评估 + OOM防御)
% 自动遍历 12 个数据集，执行 LWEA 和 LWGP
% 输出指标: ACC, NMI, Purity, Fscore, ARI 及 Runtime
% =========================================================================

clear; clc; close all;

%% 1. 实验参数与路径设置
folderPath = 'Base_LabelsPool_MAT_Allsample';  % 你的数据集所在文件夹路径
numDatasets = 12;                     % 12 个数据集
datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};

M = 10;                               % 集成规模 
cntTimes = 20;                        % 重复次数 (随机种子数) 
para_theta = 0.4;                     % 参数 theta 
clsNums = 2:30;                       % 聚类数目搜索范围 

% 标明了 MeanStd 后缀
csvFileName = 'Baseline_LWEA_LWGP_5Metrics_20Seeds_MeanStd.csv';

%% 2. 初始化 CSV 文件 (强制 GBK 编码防止 ± 乱码)
fid = fopen(csvFileName, 'w', 'n', 'GBK');
if fid == -1
    error('无法创建或打开 CSV 文件，请检查路径或文件是否被占用。');
end

% 动态生成包含 Mean±Std 的顶级学术表头，共 23 列
header = {'Dataset_Name'};
algos = {'LWEA', 'LWGP'};
types = {'BestK', 'TrueK'};
metrics = {'ACC', 'NMI', 'PUR', 'Fscore', 'ARI'};

for a = 1:2
    for t = 1:2
        for m = 1:5
            header{end+1} = sprintf('%s_%s_%s(Mean±Std)', algos{a}, types{t}, metrics{m});
        end
    end
    header{end+1} = sprintf('%s_Runtime(s)', algos{a});
end

fprintf(fid, '%s\n', strjoin(header, ','));

disp('==============================================================');
disp('>>> 开始批量复现 LWEA 和 LWGP 算法 (带 Mean±Std 统计)...');
disp(['>>> 共计 ', num2str(numDatasets), ' 个数据集，每个独立运行 ', num2str(cntTimes), ' 个种子']);
disp('==============================================================');

%% 3. 遍历数据集进行实验
for d = 1:numDatasets
    currentDatasetName = datasetNames{d};
    fileName = sprintf('LabelsPool_%s.mat', currentDatasetName);
    fullPath = fullfile(folderPath, fileName);
    
    if ~exist(fullPath, 'file')
        warning(['未找到文件: ', fullPath, '，跳过。']);
        continue;
    end
    
    fprintf('\n>>> 正在处理: Dataset %02d (%s) ...\n', d, currentDatasetName);
    
    % 加载数据
    data = load(fullPath);
    base_labels = data.base_labels; % 尺寸: 100 x N
    Y = data.Y;                     % 尺寸: N x 1 (真实标签)
    
    [poolSize, ~] = size(base_labels); 
    trueK = numel(unique(Y));       % 获取当前数据集的真实类别数
    
    % 预分配矩阵存储每次运行的 11 个数值: [Best(5), True(5), Runtime]
    res_lwea = zeros(cntTimes, 11);
    res_lwgp = zeros(cntTimes, 11);
    
    is_oom = false; % OOM 及异常标志位
    
    %% 4. 多次独立运行集成实验 (带 try-catch 保护)
    try
        for runIdx = 1:cntTimes
            % 严格控制每次循环的随机种子，保证可复现且彼此独立
            rng(runIdx * 1000 + d, 'twister'); 
            
            % 随机抽取 M 个基聚类并转置为 N x M
            selected_idx = randperm(poolSize, M);
            baseCls = base_labels(selected_idx, :)'; 
            
            % --- 原论文核心算法步骤 ---
            [bcs, baseClsSegs] = getAllSegs(baseCls);
            
            tic;
            ECI = computeECI(bcs, baseClsSegs, para_theta);
            time_eci = toc;
            
            tic;
            LWCA = computeLWCA(baseClsSegs, ECI, M);
            time_lwca = toc;        

            tic;
            predY_LWGP_all = runLWGP(bcs, baseClsSegs, ECI, clsNums);
            time_lwgp_core = toc;
            
            tic;
            predY_LWEA_all = runLWEA(LWCA, clsNums);
            time_lwea_core = toc;     

            % --- 统计耗时 ---
            total_time_lwgp = time_eci + time_lwgp_core;
            total_time_lwea = time_eci + time_lwca + time_lwea_core;
            
            % --- 获取 5 项评价指标 ---
            scores_lwea = evaluateAllCriteria(Y, predY_LWEA_all, clsNums, trueK);
            scores_lwgp = evaluateAllCriteria(Y, predY_LWGP_all, clsNums, trueK);
            
            % 存入结果矩阵
            res_lwea(runIdx, :) = [scores_lwea, total_time_lwea];
            res_lwgp(runIdx, :) = [scores_lwgp, total_time_lwgp];   
        end
    catch ME
        % 捕捉到内存溢出或计算异常，直接熔断
        is_oom = true;
        fprintf('    [触发熔断] 该数据集发生崩溃或内存溢出: %s\n', ME.message);
    end
        
    %% 5. 结果结算与 CSV 写入 (Mean ± Std 核心逻辑)
    if is_oom
        % 如果发生异常，全行写入 OOM
        row_data_str = repmat({'OOM'}, 1, 22); % 11(LWEA) + 11(LWGP) = 22 个指标占位
        fprintf(fid, '%s,%s\n', currentDatasetName, strjoin(row_data_str, ','));
    else
        % 计算 LWEA 的均值和无偏标准差
        avg_lwea = mean(res_lwea, 1);
        std_lwea = std(res_lwea, 0, 1);
        
        % 计算 LWGP 的均值和无偏标准差
        avg_lwgp = mean(res_lwgp, 1);
        std_lwgp = std(res_lwgp, 0, 1);
        
        % 将前 10 个指标格式化为 'Mean±Std' (LWEA)
        str_lwea = cell(1, 10);
        for i = 1:10
            str_lwea{i} = sprintf('%.4f±%.4f', avg_lwea(i), std_lwea(i));
        end
        
        % 将前 10 个指标格式化为 'Mean±Std' (LWGP)
        str_lwgp = cell(1, 10);
        for i = 1:10
            str_lwgp{i} = sprintf('%.4f±%.4f', avg_lwgp(i), std_lwgp(i));
        end
        
        % 拼接写入: LWEA(10项±)+Runtime, LWGP(10项±)+Runtime
        row_str_lwea = strjoin(str_lwea, ',');
        row_str_lwgp = strjoin(str_lwgp, ',');
        
        fprintf(fid, '%s,%s,%.4f,%s,%.4f\n', ...
            currentDatasetName, ...
            row_str_lwea, avg_lwea(11), ...
            row_str_lwgp, avg_lwgp(11));
            
        % 控制台打印部分指标 (第6项是 TrueK_ACC)
        fprintf('    完成! LWEA TrueK_ACC: %s | LWGP TrueK_ACC: %s\n', str_lwea{6}, str_lwgp{6});
    end
end

fclose(fid);
disp('==============================================================');
disp(['>>> 所有复现实验运行结束！综合平均结果已保存至: ', csvFileName]);

% 如果确定不需要了再rmpath，防止影响后续脚本运行
% rmpath('Base_LabelsPool_MAT_Allsample');

%% =========================================================================
%% 辅助函数：同时提取 Best-k 和 True-k 的 5 项指标
function scores = evaluateAllCriteria(Y, predY_all, clsNums, trueK)
    % 提取 5 个指标: ACC(1), NMI(2), Purity(3), Fscore(4), ARI(8)
    numK = size(predY_all, 2);
    temp_metrics = zeros(numK, 5);
    
    for i = 1:numK
        [acc, nmi, pur, fsc, ~, ~, ~, ari] = ClusteringMeasure4(Y, predY_all(:, i));
        temp_metrics(i, :) = [acc, nmi, pur, fsc, ari];
    end
    
    % Best-k: 针对每个指标寻找最大值 (各个指标在不同 K 下的最优解)
    best_metrics = max(temp_metrics, [], 1);
    
    % True-k: 寻找 k 等于真实类别数所在的索引
    k_idx = find(clsNums == trueK);
    if isempty(k_idx)
        % 防止 trueK 不在 [2, 30] 范围内导致报错
        true_metrics = NaN(1, 5);
    else
        true_metrics = temp_metrics(k_idx, :);
    end
    
    % 输出尺寸为 1 x 10 的向量 (前5个Best，后5个True)
    scores = [best_metrics, true_metrics];
end