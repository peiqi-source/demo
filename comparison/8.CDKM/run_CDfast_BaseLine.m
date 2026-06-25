% =========================================================================
% CDfast 算法自动化测试脚本 (顶刊 Mean±Std 格式 + 独立 Max 列 + 详细记录留档)
% 适配 12 个数据集、多种初始化策略、ClusteringMeasure4 极限版、20次随机种子
% =========================================================================

clear all; clc; close all;
warning('off', 'stats:kmeans:FailedToConverge'); % 关闭 K-means 提前中断的警告

%% 1. 全局超参数与环境配置
datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                 'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
numDatasets = length(datasetNames);

numRuns = 20; % 独立运行次数 (严格对齐 20 次)
init_strategies = {'Random', 'K-means++'}; % 初始化策略

% 1.1 创建专门用于存放 20 次详细记录的文件夹
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
detailDir = fullfile(pwd, sprintf('results_CDfast_details_%s', timestamp));
if ~exist(detailDir, 'dir')
    mkdir(detailDir);
end

% 主汇总表文件路径 (更新了命名)
csvFileName = fullfile(pwd, 'Baseline_CDfast_5Metrics_20Seeds_MeanStd_and_Max.csv');

%% 2. 初始化汇总 CSV 文件 (全局句柄统一管理，强行指定 GBK 编码防止 ± 乱码)
fid = fopen(csvFileName, 'w', 'n', 'GBK');
if fid == -1
    error('无法创建或打开 CSV 文件，请检查路径或文件是否被占用。');
end

% 动态生成顶级学术表头：将 Mean±Std 和 Max 彻底拆分为独立的列
header = {'Dataset_Name', 'Init_Strategy', ...
          'CDfast_ACC(Mean±Std)', 'CDfast_ACC(Max)', ...
          'CDfast_NMI(Mean±Std)', 'CDfast_NMI(Max)', ...
          'CDfast_PUR(Mean±Std)', 'CDfast_PUR(Max)', ...
          'CDfast_Fscore(Mean±Std)', 'CDfast_Fscore(Max)', ...
          'CDfast_ARI(Mean±Std)', 'CDfast_ARI(Max)', ...
          'CDfast_Runtime(s)'};
fprintf(fid, '%s\n', strjoin(header, ','));

disp('====================================================================');
disp('>>> 启动 CDfast 加速坐标下降聚类 自动化基准测试');
disp(['>>> 详细日志将保存至: ', detailDir]);
disp(['>>> 共计 ', num2str(numDatasets), ' 个数据集，每种策略独立运行 ', num2str(numRuns), ' 个种子']);
disp('====================================================================');

%% 3. 主实验 Pipeline 遍历数据集
for d_idx = 1:numDatasets
    dataname = datasetNames{d_idx};
    
    try
        % 接入数据加载器
        [X, Y] = loaddata(d_idx); 
    catch ME
        warning('加载数据集 %s 失败，请检查 ec_data 文件夹。跳过...', dataname);
        continue;
    end
    
    [N, dim] = size(X);
    nC = length(unique(Y));

    % 严格归一化：按列缩放至 [0, 1]
    X = double(X);
    X_min = min(X, [], 1);
    X_max = max(X, [], 1);
    X_range = X_max - X_min;
    X_range(X_range == 0) = 1; 
    X = (X - X_min) ./ X_range;

    % [核心对齐]：CDKM_fast 源码要求输入矩阵维度必须是 d*n (特征维度 x 样本数)
    X_trans = X';
    
    fprintf('\n>>> 正在处理: Dataset %02d (%s) | 样本数 N=%d, 类别数 K=%d\n', d_idx, dataname, N, nC);
    
    % 用于预分配当前数据集下所有策略、所有 seed 的详细记录 [共 40 行 x 9 列]
    raw_results_cell = cell(length(init_strategies) * numRuns, 9);
    row_idx = 1; % 详细记录的行指针
    
    % 4. 遍历不同的初始化策略
    for s_idx = 1:length(init_strategies)
        strategy = init_strategies{s_idx};
        is_oom = false;
        
        % 存放当前策略下 20 次随机种子的结果：[ACC, NMI, PUR, Fscore, ARI, Runtime]
        res_cdfast = zeros(numRuns, 6);
        
        try
            for runIdx = 1:numRuns
                % 严格控制每次循环的随机种子，保证可复现
                current_seed = runIdx * 100 + d_idx * 1;
                rng(current_seed, 'twister'); 
                
                Tstart = tic;
                
                % ==============================================
                % 初始标签生成逻辑
                % ==============================================
                if strcmp(strategy, 'Random')
                    rand_indices = randperm(N, nC);
                    centroids = X(rand_indices, :);
                    [~, init_label] = min(pdist2(X, centroids), [], 2);
                else
                    [~, centroids] = kmeans(X, nC, 'MaxIter', 0, 'Replicates', 1);
                    [~, init_label] = min(pdist2(X, centroids), [], 2);
                end

                % 调用加速版坐标下降 K-means 算法
                [y_pred, iter_num, obj_max] = CDKM_fast(X_trans, init_label, nC);
                
                time_record = toc(Tstart);
                
                % 接入极限版测评函数
                [ACC, NMI, PUR, Fscore, ~, ~, ~, ARI] = ClusteringMeasure4(Y, y_pred); 
                
                res_cdfast(runIdx, :) = [ACC, NMI, PUR, Fscore, ARI, time_record];
                
                % 存入详细日志 cell
                raw_results_cell(row_idx, :) = {dataname, strategy, current_seed, ACC, NMI, PUR, Fscore, ARI, time_record};
                row_idx = row_idx + 1;
            end
            
        catch ME
            is_oom = true;
            fprintf('    [触发熔断] 策略 [%s] 发生崩溃或内存溢出: %s\n', strategy, ME.message);
            
            % 如果崩溃，用 OOM 填充详细日志的剩余部分
            for fillIdx = runIdx:numRuns
                 raw_results_cell(row_idx, :) = {dataname, strategy, 'OOM', 'OOM', 'OOM', 'OOM', 'OOM', 'OOM', 'OOM'};
                 row_idx = row_idx + 1;
            end
        end
        
        % =========================================================
        % 5. 统计与写入主汇总 CSV (独立 Mean±Std 与 Max 列)
        % =========================================================
        if is_oom
            % 如果 OOM，填入 11 个 'OOM' (5个MeanStd + 5个Max + 1个Runtime)
            row_data_str = repmat({'OOM'}, 1, 11); 
            fprintf(fid, '%s,%s,%s\n', dataname, strategy, strjoin(row_data_str, ','));
        else
            % 计算均值、无偏标准差 和 最大值
            avg_metrics = mean(res_cdfast, 1); 
            std_metrics = std(res_cdfast, 0, 1); 
            max_metrics = max(res_cdfast, [], 1);
            
            % 仅将 Mean±Std 格式化为字符串
            str_metrics = cell(1, 5);
            for i = 1:5
                str_metrics{i} = sprintf('%.4f±%.4f', avg_metrics(i), std_metrics(i));
            end
            
            % 写入主 CSV (格式：Mean±Std, Max, Mean±Std, Max ...)
            fprintf(fid, '%s,%s,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%.4f\n', ...
                    dataname, strategy, ...
                    str_metrics{1}, max_metrics(1), ...
                    str_metrics{2}, max_metrics(2), ...
                    str_metrics{3}, max_metrics(3), ...
                    str_metrics{4}, max_metrics(4), ...
                    str_metrics{5}, max_metrics(5), ...
                    avg_metrics(6));
            
            % 控制台精简打印，展示一下 Mean 和 Max 的对比
            fprintf('  -> [%-9s] 完毕! ACC: %s (Max: %.4f) | NMI: %s (Max: %.4f) | T: %.4fs\n', ...
                    strategy, str_metrics{1}, max_metrics(1), str_metrics{2}, max_metrics(2), avg_metrics(6));
        end
    end % 结束 strategy 循环
    
    % =========================================================
    % 6. 当前数据集跑完，导出 20 次独立运行的详细记录 CSV
    % =========================================================
    detailCsvName = fullfile(detailDir, sprintf('CDfast_Details_%s_%s.csv', dataname, timestamp));
    varNames = {'Dataset', 'Init_Strategy', 'Seed', 'ACC', 'NMI', 'PUR', 'Fscore', 'ARI', 'Runtime'};
    detailTable = cell2table(raw_results_cell, 'VariableNames', varNames);
    writetable(detailTable, detailCsvName);
    
end % 结束 dataset 循环

% 统一释放文件句柄
fclose(fid);

fprintf('\n====================================================================\n');
fprintf('[SUCCESS] 实验全部运行完毕！\n');
fprintf('  - 综合汇总表: %s\n', csvFileName);
fprintf('  - 详细实验日志已保存至文件夹: %s\n', detailDir);
fprintf('====================================================================\n');