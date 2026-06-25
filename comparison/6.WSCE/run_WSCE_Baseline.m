% =========================================================================
% WSCE 算法自动化测试脚本 (顶刊 Mean±Std 格式 + 5指标全面评估 + OOM防御)
% 适配 12 个数据集、ClusteringMeasure4 极限版、20次随机种子
% =========================================================================

clc; clear; close all;

%% 1. 全局超参数与环境配置
numRuns = 20;           % 独立运行次数 (严格对齐20次)
block_size = 10;        % WSCE 内部参数
num_neighbors = 10;     % KNN的近邻数
ShowDendrogram = 0;     % 批量运行必须关闭树状图，防止卡死电脑

% 将包含 loaddata 和数据集的文件夹加入路径
datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
numDatasets = length(datasetNames);

csvFileName = fullfile(pwd, 'Baseline_WSCE_5Metrics_20Seeds_MeanStd.csv');

%% 2. 初始化 CSV 文件 (全局句柄统一管理，强行指定 GBK 编码防止 ± 乱码)
fid = fopen(csvFileName, 'w', 'n', 'GBK');
if fid == -1
    error('无法创建或打开 CSV 文件，请检查路径或文件是否被占用。');
end

% 动态生成包含 Mean±Std 的顶级学术表头
header = {'Dataset_Name', 'WSCE_ACC(Mean±Std)', 'WSCE_NMI(Mean±Std)', ...
          'WSCE_PUR(Mean±Std)', 'WSCE_Fscore(Mean±Std)', 'WSCE_ARI(Mean±Std)', 'WSCE_Runtime(s)'};
fprintf(fid, '%s\n', strjoin(header, ','));

disp('====================================================================');
disp('>>> 启动 WSCE 端到端集成自动化基准测试 (带 Mean±Std 统计)');
disp(['>>> 共计 ', num2str(numDatasets), ' 个数据集，每个独立运行 ', num2str(numRuns), ' 个种子']);
disp('====================================================================');

%% 3. 主实验 Pipeline 遍历数据集
for d_idx = 1:numDatasets
    currentDatasetName = datasetNames{d_idx};
    
    try
        % WSCE 需要原始数据 X，复用数据加载接口
        [X, Y] = loaddata(d_idx);
    catch ME
        warning('加载数据集 %s 失败，请检查 ec_data 文件夹。跳过...', currentDatasetName);
        continue;
    end
    
    truelabels = Y(:);
    K = numel(unique(truelabels));
    N = size(X, 1);
    
    fprintf('\n>>> 正在处理: Dataset %02d (%s) | 样本数 N=%d, 类别数 K=%d\n', d_idx, currentDatasetName, N, K);
    
    % 存放 20 次随机种子的结果：[ACC, NMI, PUR, Fscore, ARI, Runtime]
    res_wsce = zeros(numRuns, 6);
    is_oom = false; % OOM 及异常标志位
    
    %% 4. 独立重复实验 (带 try-catch 保护)
    try
        for runIdx = 1:numRuns
            % 严格控制每次循环的随机种子，保证可复现
            rng(runIdx * 1000 + d_idx, 'twister'); 
            
            % 运行 WSCE 算法并独立计时
            tic;
            predY = WSCE(X, K, block_size, num_neighbors, ShowDendrogram);
            run_time = toc; 
            
            % 接入极限版测评函数 (提取 5 个指标)
            [ACC, NMI, PUR, Fscore, ~, ~, ~, ARI] = ClusteringMeasure4(truelabels, predY);
            
            res_wsce(runIdx, :) = [ACC, NMI, PUR, Fscore, ARI, run_time];
        end
    catch ME
        % 捕捉到内存溢出或计算异常，直接熔断
        is_oom = true;
        fprintf('    [触发熔断] 该数据集发生崩溃或内存溢出: %s\n', ME.message);
    end
    
    %% 5. 统计与写入 CSV (Mean ± Std 核心逻辑)
    if is_oom
        % 如果发生异常，全行写入 OOM
        row_data_str = repmat({'OOM'}, 1, 6); 
        fprintf(fid, '%s,%s\n', currentDatasetName, strjoin(row_data_str, ','));
    else
        % 正常跑完 20 个 seed，计算均值 (Mean) 和无偏标准差 (Std)
        avg_metrics = mean(res_wsce, 1);
        std_metrics = std(res_wsce, 0, 1);
        
        % 将 5 个指标格式化为 'Mean±Std' 的字符串数组
        str_metrics = cell(1, 5);
        for i = 1:5
            str_metrics{i} = sprintf('%.4f±%.4f', avg_metrics(i), std_metrics(i));
        end
        
        % 写入 CSV (指标用 %s 占位接收拼接好的字符串，时间用 %.4f)
        fprintf(fid, '%s,%s,%s,%s,%s,%s,%.4f\n', ...
                currentDatasetName, ...
                str_metrics{1}, str_metrics{2}, str_metrics{3}, str_metrics{4}, str_metrics{5}, ...
                avg_metrics(6));
        
        % 控制台精简打印
        fprintf('    完成! ACC: %s | NMI: %s | Runtime: %.4fs\n', ...
                str_metrics{1}, str_metrics{2}, avg_metrics(6));
    end
end

fclose(fid);
disp('====================================================================');
fprintf('[SUCCESS] 实验全部跑完！对比基准已输出至文件:\n%s\n', csvFileName);
disp('====================================================================');