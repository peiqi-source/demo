%% exp_parameter_write.m 性能指标提升实验 (彻底解决动态扩容与高频 I/O 写入慢的问题)
clear;
clc;
close all;

%% setup paths
thisFile = mfilename("fullpath");
expDir = fileparts(thisFile);
rootDir = fileparts(expDir);
resultsDir = fullfile(rootDir, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end % 确保结果目录存在

%% 数据集设置 
dataset_list = 9; 
select_seeds = [2 5 8];
selection_metric = 'ACC';   % 可选: 'ACC', 'NMI', 'Purity', 'Fscore'

%% 参数定义 (网格搜索空间)
% param_anchors_rate = [10 12];
% param_order = 2:6;
% param_num_sampling = 2:5;
% param_k = [5 7 10 15];
% delta = 5;

% 大规模数据集
param_anchors_rate = [10 50];
param_order = 2:5;
param_num_sampling = 3:5;
param_k = [5 7 10 15];
delta = 15;

% 计算单个数据集的总循环次数，用于重置进度提示与预分配
total_loops_per_dataset = length(param_anchors_rate) * length(param_order) * length(param_num_sampling) * length(param_k);

%% 开始最外层数据集遍历
for data_idx = dataset_list
    
    % 强制回收上一轮的所有大型变量
    clear X Y all_results_mat result_cell F H B B1_cell C U;
    
    loop_idx = 1; 
    timestamp = datestr(now, 'yyyymmdd_HHMMSS'); 
    csvFileName = fullfile(resultsDir, sprintf('SelectModel_Data%d_%s.csv', data_idx, timestamp));
    matFileName = fullfile(resultsDir, sprintf('Data%d_%s_select.mat', data_idx, timestamp));
    
    % 预分配 .mat 的结构体空间
    empty_struct = struct('Anchors', [], 'Order', [], 'NumSampling', [], 'K', [],...
         'MeanACC', [], 'MeanNMI', [], 'MeanPurity', [], 'MeanFscore', [], 'MeanRuntime', []);
    all_results_mat = repmat(empty_struct, total_loops_per_dataset, 1);
    
    % 预分配 CSV 的内存元胞空间 (彻底移除循环内的 writetable!)
    result_cell = cell(total_loops_per_dataset, 11);
    
    % 加载当前数据集
    fprintf('\n======================================================\n');
    fprintf('>>> 正在加载并运行数据集编号: %d <<<\n', data_idx);
    fprintf('>>> 实验结束后，指标 CSV 将一次性保存至: %s\n', csvFileName);
    fprintf('>>> 实验结束后，全量 MAT 将一次性保存至: %s\n', matFileName);
    fprintf('======================================================\n');
    
    [X, Y] = loaddata2(data_idx);
    [num, dim] = size(X);
    X = X ./ max(X, [], 2); % 归一化
    c = length(unique(Y));
    
    %% 开始网格参数搜索
    for ar = param_anchors_rate
        for ord = param_order
            for ns = param_num_sampling
                for k_val = param_k
                    anchors = [];
                    for t = 1:ns
                        anchors = [anchors, (ar+(t-1)*delta)*c];
                    end
                    
                    fprintf('进度: %d / %d (%.2f%%) | Data: %d | AR=%d, Ord=%d, NS=%d, K=%d ... ', ...
                        loop_idx, total_loops_per_dataset, (loop_idx/total_loops_per_dataset)*100, data_idx, ar, ord, ns, k_val);
                    
                    % 对当前超参数组合，使用少量 seed 做平均
                    metric_list = zeros(length(select_seeds), 4); % [ACC NMI Purity Fscore]
                    runtime_list = zeros(length(select_seeds), 1);

                    for s = 1:length(select_seeds)
                        seed = select_seeds(s);
                        rng(seed);

                        % 1. 运行核心实验
                        [F, ~, runtime, ~] = AHD_EC(k_val, ord, X, anchors, c);
                        fprintf("Over ... ");
                    
                        % 2. 评估聚类结果
                        [ACC, MIhat, Purity, Fscore, P, R, RI] = ClusteringMeasure4(Y, F);

                        metric_list(s, :) = [ACC, MIhat, Purity, Fscore];
                        runtime_list(s) = runtime;
                    end

                    mean_ACC    = mean(metric_list(:,1));
                    mean_NMI    = mean(metric_list(:,2));
                    mean_Purity = mean(metric_list(:,3));
                    mean_Fscore = mean(metric_list(:,4));
                    mean_runtime = mean(runtime_list);

                    
                    % 3. 将矩阵转化为字符串 (仅供 CSV 使用)
                    anchors_str = mat2str(anchors);

                    result_cell(loop_idx, :) = {data_idx, ar, anchors_str, ord, ns, k_val, ...
                        mean_ACC, mean_NMI, mean_Purity, mean_Fscore, mean_runtime};
                    
                    % 直接往预先挖好的"坑"里填数据
                    all_results_mat(loop_idx).AnchorsRate = ar;
                    all_results_mat(loop_idx).Order = ord;
                    all_results_mat(loop_idx).NumSampling = ns;
                    all_results_mat(loop_idx).K = k_val;
                    all_results_mat(loop_idx).Anchors = anchors;
                    all_results_mat(loop_idx).MeanACC = mean_ACC;
                    all_results_mat(loop_idx).MeanNMI = mean_NMI;
                    all_results_mat(loop_idx).MeanPurity = mean_Purity;
                    all_results_mat(loop_idx).MeanFscore = mean_Fscore;
                    all_results_mat(loop_idx).MeanRuntime = mean_runtime;
                    
                    fprintf('Done.\n');
                    loop_idx = loop_idx + 1;
                    
                    % 清空不需要的临时大矩阵
                    clear F;
                end
            end
        end
    end
    
    % 根据 selection_metric 选最优参数
    switch upper(selection_metric)
        case 'ACC'
            score_vec = [all_results_mat.MeanACC];
        case 'NMI'
            score_vec = [all_results_mat.MeanNMI];
        case 'PURITY'
            score_vec = [all_results_mat.MeanPurity];
        case 'FSCORE'
            score_vec = [all_results_mat.MeanFscore];
        otherwise
            error('Unknown selection metric: %s', selection_metric);
    end

    [~, best_idx] = max(score_vec);
    best_cfg = all_results_mat(best_idx);

    fprintf('\n>>>最优参数如下：\n');
    fprintf('AR=%d, Ord=%d, NS=%d, K=%d, Anchors=%s\n', ...
        best_cfg.AnchorsRate, best_cfg.Order, best_cfg.NumSampling, ...
        best_cfg.K, mat2str(best_cfg.Anchors));
    fprintf('Mean ACC=%.4f, Mean NMI=%.4f, Mean Purity=%.4f, Mean Fscore=%.4f\n', ...
        best_cfg.MeanACC, best_cfg.MeanNMI, best_cfg.MeanPurity, best_cfg.MeanFscore);

    % 循环彻底结束，执行唯一一次集中式硬盘 I/O 写入！
    fprintf('\n>>> 正在将数据集 %d 的 平均指标 导出至 CSV...\n', data_idx);
    varNames = {'DatasetID', 'AnchorRate', 'Anchors', 'Order', 'NumSampling', 'K', ...
                'MeanACC', 'MeanNMI', 'MeanPurity', 'MeanFscore', 'MeanRuntime'};
    result_table = cell2table(result_cell, 'VariableNames', varNames);
    writetable(result_table, csvFileName); % 整个数据集只写 1 次！
    
    % 跑完一个数据集后，执行终极清理
    clear all_results_mat result_cell result_table; 
    fclose('all');         
    
    fprintf('>>> 数据集 %d 运行与保存完毕！内存已清空！\n', data_idx);
end

fprintf('\n 所有数据集实验全部结束！CSV 与 MAT 已稳妥保存！\n');