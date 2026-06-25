% =========================================================================
% DREC 对比算法自动化测试脚本 (顶刊 Mean±Std 格式 + 基聚类池评估 + OOM防御)
% 适配 12 个数据集、100基聚类池、ClusteringMeasure4 极限版
% =========================================================================

clear all; close all; clc; 

%% 1. 全局超参数与接口设置
M = 20;               % 每次集成选取的基聚类数量
lambda = 100;         % DREC 的固定正则化参数
numRuns = 20;         % 独立运行次数 (Seed数量，对齐其他基准测试)

% 添加工作路径 (请根据实际情况确保当前工作路径在项目根目录)
addpath(fullfile(pwd, 'DREC'));
addpath(fullfile(pwd, 'Functions'));

% 路径设置对齐，确保使用统一的数据集文件命名格式
folderPath = 'Base_LabelsPool_MAT_Allsample';
datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
numDatasets = length(datasetNames);

% 更新了输出文件后缀，标明使用了 MeanStd 统计
csvFileName = sprintf('Baseline_DREC_PoolStats_5Metrics_M%d_20Seeds_MeanStd.csv', M);

%% 2. 初始化 CSV 文件 (强制 GBK 编码防止 ± 乱码)
fid = fopen(csvFileName, 'w', 'n', 'GBK');
if fid == -1
    error('无法创建或打开 CSV 文件，请检查路径或文件是否被占用。');
end

% 动态生成超长表头，每个指标包含4项，并标注 (Mean±Std)
metricNames = {'ACC', 'NMI', 'PUR', 'Fscore', 'ARI'};
header = {'Dataset_Name'};
for m = 1:5
    mName = metricNames{m};
    header = [header, {sprintf('Pool_Mean_%s(Mean±Std)', mName), ...
                       sprintf('Pool_Max_%s(Mean±Std)', mName), ...
                       sprintf('Pool_Min_%s(Mean±Std)', mName), ...
                       sprintf('DREC_%s(Mean±Std)', mName)}];
end
header = [header, {'DREC_Runtime(s)'}]; % 加上 Runtime，总共 22 列

fprintf(fid, '%s\n', strjoin(header, ','));

disp('====================================================================');
disp(['>>> 启动 DREC (含基聚类池评估 + Mean±Std) 自动化基准测试 | M = ', num2str(M)]);
disp(['>>> 共计 ', num2str(numDatasets), ' 个数据集，每个独立运行 ', num2str(numRuns), ' 个种子']);
disp('====================================================================');

%% 3. 主实验 Pipeline 遍历数据集
for d_idx = 1:numDatasets
    % 动态加载数据集
    currentDatasetName = datasetNames{d_idx};
    fileName = sprintf('LabelsPool_%s.mat', currentDatasetName);
    filePath = fullfile(folderPath, fileName);
    
    if ~exist(filePath, 'file')
        warning('未找到文件: %s，已跳过。', filePath);
        continue;
    end
    
    % 加载数据
    load(filePath, 'base_labels', 'Y');
    
    % 矩阵方向校正：DREC 需要 "样本数 x 集成数"，必须进行转置
    if size(base_labels, 1) > size(base_labels, 2)
        E_pool = base_labels;   
    else
        E_pool = base_labels';  
    end
    
    truelabels = Y(:);
    K = max(truelabels);
    [N, total_M] = size(E_pool);
    
    fprintf('\n>>> 正在处理: Dataset %02d (%s) | 样本数 N=%d, 类别数 K=%d\n', d_idx, currentDatasetName, N, K);
    
    % 存放 20 次随机种子的所有结果 (5个指标 * 4个统计量 + 1个耗时 = 21列)
    res_all_seeds = zeros(numRuns, 21);
    is_oom = false; % OOM 及异常标志位
    
    %% 4. 独立重复实验与基聚类抽样 (带 try-catch 保护)
    try
        for runIdx = 1:numRuns
            % 严格控制每次循环的随机种子，保证可复现
            rng(runIdx * 1000 + d_idx, 'twister'); 
            
            % 从 100 个基聚类池中随机挑选 M 个
            sel_idx = randperm(total_M, M);
            E_current = E_pool(:, sel_idx);
            
            % -------------------------------------------------------------
            % 环节 A：评估当前抽取的 M 个基聚类的质量 (Pool Stats)
            % -------------------------------------------------------------
            % pool_metrics 尺寸: M x 5 (存放当前 M 个基聚类的 5 个指标)
            pool_metrics = zeros(M, 5);
            for i = 1:M
                [acc, nmi, pur, fsc, ~, ~, ~, ari] = ClusteringMeasure4(truelabels, E_current(:, i));
                pool_metrics(i, :) = [acc, nmi, pur, fsc, ari];
            end
            
            % 计算该 seed 下池子的统计量: 1x5 向量
            pool_mean = mean(pool_metrics, 1);
            pool_max = max(pool_metrics, [], 1);
            pool_min = min(pool_metrics, [], 1);
            
            % -------------------------------------------------------------
            % 环节 B：运行 DREC 核心算法并评估
            % -------------------------------------------------------------
            tic;
            OutD = DREC(E_current, K, lambda);
            run_time = toc; 
            
            % 提取 DREC 结果标签并评估
            ESDN_ids = OutD.Blable;
            [d_acc, d_nmi, d_pur, d_fsc, ~, ~, ~, d_ari] = ClusteringMeasure4(truelabels, ESDN_ids);
            drec_metrics = [d_acc, d_nmi, d_pur, d_fsc, d_ari];
            
            % -------------------------------------------------------------
            % 环节 C：整合当前 seed 的单行数据
            % 按照 [Pool_Mean, Pool_Max, Pool_Min, DREC] 顺序交叉排列
            % -------------------------------------------------------------
            current_seed_res = zeros(1, 21);
            for m = 1:5
                current_seed_res((m-1)*4 + 1 : m*4) = [pool_mean(m), pool_max(m), pool_min(m), drec_metrics(m)];
            end
            current_seed_res(21) = run_time;
            
            % 存入总矩阵
            res_all_seeds(runIdx, :) = current_seed_res;
        end
    catch ME
        % 捕捉到内存溢出或计算异常，直接熔断
        is_oom = true;
        fprintf('    [触发熔断] 该数据集发生崩溃或内存溢出: %s\n', ME.message);
    end
    
    %% 5. 结果结算与 CSV 写入 (Mean ± Std 核心逻辑)
    if is_oom
        % 如果发生异常，21 列全部填入 OOM
        row_data_str = repmat({'OOM'}, 1, 21); 
        fprintf(fid, '%s,%s\n', currentDatasetName, strjoin(row_data_str, ','));
    else
        % 正常跑完 20 个 seed，计算均值 (Mean) 和无偏标准差 (Std)
        avg_metrics = mean(res_all_seeds, 1);
        std_metrics = std(res_all_seeds, 0, 1);
        
        % 将前 20 个指标项(Pool统计+DREC得分)格式化为 Mean±Std 的字符串数组
        str_metrics = cell(1, 20);
        for i = 1:20
            str_metrics{i} = sprintf('%.4f±%.4f', avg_metrics(i), std_metrics(i));
        end
        
        % 拼接前 20 个字符串项，再加上最后一个纯数字时间项 (保留4位小数)
        row_str = strjoin(str_metrics, ',');
        fprintf(fid, '%s,%s,%.4f\n', currentDatasetName, row_str, avg_metrics(21));
        
        % 控制台打印部分只截取 ACC 进行展示 (Pool_Mean是第1列, Pool_Max是第2列, DREC是第4列)
        fprintf('    完成! ACC (Pool_Mean: %s | Pool_Max: %s | DREC: %s)\n', ...
            str_metrics{1}, str_metrics{2}, str_metrics{4});
    end
end

fclose(fid);

% 移除临时加入的路径，保持环境整洁
rmpath(fullfile(pwd, 'DREC'));
rmpath(fullfile(pwd, 'Functions'));

disp('====================================================================');
fprintf('[SUCCESS] 实验全部跑完！对比基准已输出至文件:\n%s\n', fullfile(pwd, csvFileName));
disp('====================================================================');