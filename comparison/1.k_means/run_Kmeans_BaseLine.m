% =========================================================================
% K-means 基线算法自动化测试脚本 (顶刊 Mean±Std 格式 + 5指标全面评估 + OOM防御)
% 适配 12 个数据集、ClusteringMeasure4 极限版、20次随机种子
% =========================================================================

clear; clc; close all;

%% 1. 全局超参数与环境配置

% 原始代码的 20 个种子参数设置
param_rng = 20:20:400;
numRuns = length(param_rng); % 20 次

datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                 'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
numDatasets = length(datasetNames);

% 原始代码保留项：为每个数据集的每次独立运行保存详细记录的文件夹
thisFile = mfilename("fullpath");
KMeansDir = fileparts(thisFile);
resultsDir = fullfile(KMeansDir, 'results_kmeans');
if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

% 更新了主汇总输出文件后缀，标明使用了 MeanStd 统计
csvFileName = fullfile(pwd, 'Baseline_Kmeans_5Metrics_20Seeds_MeanStd.csv');

%% 2. 初始化汇总 CSV 文件 (强制 GBK 编码防止 ± 乱码，全局句柄)
fid = fopen(csvFileName, 'w', 'n', 'GBK');
if fid == -1
    error('无法创建或打开 CSV 文件，请检查路径或文件是否被占用。');
end

% 动态生成包含 Mean±Std 的顶级学术表头
header = {'Dataset_Name', 'Kmeans_ACC(Mean±Std)', 'Kmeans_NMI(Mean±Std)', ...
          'Kmeans_PUR(Mean±Std)', 'Kmeans_Fscore(Mean±Std)', 'Kmeans_ARI(Mean±Std)', 'Kmeans_Runtime(s)'};
fprintf(fid, '%s\n', strjoin(header, ','));

disp('====================================================================');
disp('>>> 启动 K-means 自动化基准测试 (带 Mean±Std 统计)');
disp(['>>> 共计 ', num2str(numDatasets), ' 个数据集，每个独立运行 ', num2str(numRuns), ' 个种子']);
disp('====================================================================');

%% 3. 主实验 Pipeline 遍历数据集
for d_idx = 1:numDatasets
    dataname = datasetNames{d_idx};
    
    try
        % 接入数据加载器
        [X, Y] = loaddata_small(d_idx);
    catch ME
        warning('加载数据集 %s 失败，请检查 ec_data 文件夹。跳过...', dataname);
        continue;
    end
    
    [N, dim] = size(X);
    
    % 原代码特色：按行归一化
    X = X ./ max(X, [], 2);
    c = length(unique(Y));

    fprintf('\n>>> 正在处理: Dataset %02d (%s) | 样本数 N=%d, 类别数 K=%d\n', d_idx, dataname, N, c);

    % 存放 20 次随机种子的汇总结果：[ACC, NMI, PUR, Fscore, ARI, Runtime]
    res_kmeans = zeros(numRuns, 6);
    % 存放 20 次随机种子的极度详细记录 (保留原代码习惯)
    raw_results_cell = cell(numRuns, 10); 
    is_oom = false; % OOM 及异常标志位
    
    %% 4. 独立重复实验 (带 try-catch 保护)
    try
        loop_idx = 1;
        for seed = param_rng
            % 严格控制随机种子
            rng(seed, 'twister'); 

            % 调用蔡登教授的 Litekmeans1
            [labels, ~, ~, ~, ~, runtime] = litekmeans1(X, c);
            
            % 接入极限版测评函数 (提取全量指标)
            [ACC, NMI, PUR, Fscore, P, R, ~, ARI] = ClusteringMeasure4(Y, labels);
            
            % 提取需要统计 Mean±Std 的 5 项指标 + Runtime
            res_kmeans(loop_idx, :) = [ACC, NMI, PUR, Fscore, ARI, runtime];
            
            % 提取原代码需要的 10 项细粒度指标保存 raw data
            raw_results_cell(loop_idx, :) = {d_idx, seed, ACC, NMI, PUR, Fscore, P, R, ARI, runtime};
            
            loop_idx = loop_idx + 1;
        end
    catch ME
        is_oom = true;
        fprintf('    [触发熔断] 该数据集发生崩溃或内存溢出: %s\n', ME.message);
    end
    
    %% 5. 详细日志保存与汇总 CSV 写入
    if is_oom
        % 如果发生异常，全行写入 OOM
        row_data_str = repmat({'OOM'}, 1, 6); 
        fprintf(fid, '%s,%s\n', dataname, strjoin(row_data_str, ','));
    else
        % --- A. 保留原始特性：导出单数据集独立 raw data 详细表 ---
        timestamp = datestr(now, 'yyyymmdd_HHMMSS'); 
        detailCsv = fullfile(resultsDir, sprintf("KMeans_results_%s_%s.csv", dataname, timestamp));
        varNames = {'DatasetID', 'Seed', 'ACC', 'NMI', 'Purity', 'Fscore', 'P', 'R', 'ARI', 'Runtime'};
        result_table = cell2table(raw_results_cell, 'VariableNames', varNames);
        writetable(result_table, detailCsv); 
        
        % --- B. 核心汇总：Mean ± Std 统计写入全局 CSV ---
        avg_metrics = mean(res_kmeans, 1);
        std_metrics = std(res_kmeans, 0, 1);
        
        % 将 5 个指标格式化为 'Mean±Std' 的字符串数组
        str_metrics = cell(1, 5);
        for i = 1:5
            str_metrics{i} = sprintf('%.4f±%.4f', avg_metrics(i), std_metrics(i));
        end
        
        % 写入全局汇总 CSV
        fprintf(fid, '%s,%s,%s,%s,%s,%s,%.4f\n', ...
                dataname, ...
                str_metrics{1}, str_metrics{2}, str_metrics{3}, str_metrics{4}, str_metrics{5}, ...
                avg_metrics(6));
                
        % 控制台精简输出
        fprintf('    完成! ACC: %s | NMI: %s | T: %.4fs (明细已存入 results_kmeans/)\n', ...
            str_metrics{1}, str_metrics{2}, avg_metrics(6));
    end
end

% 统一释放全局文件句柄
fclose(fid);

disp('====================================================================');
fprintf('[SUCCESS] 所有实验运行完毕！综合汇总结果已保存至:\n%s\n', csvFileName);
disp('====================================================================');