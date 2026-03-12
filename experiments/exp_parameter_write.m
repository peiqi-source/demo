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
dataset_list = 1; 

%% 参数定义 (网格搜索空间)

param_anchors_rate = [10 12];
param_order = 2:6;
param_num_sampling = 2:5;
param_k = [5 7 10 15];
param_rng = 2:2:16;
delta = 5;

% 大规模数据集
% param_anchors_rate = [10 20];
% param_order = 3:6;
% param_num_sampling = 3:5;
% param_k = [5 7 10 15];
% param_rng = 2:3:14;
% delta = 50;

% 计算单个数据集的总循环次数，用于重置进度提示与预分配
total_loops_per_dataset = length(param_anchors_rate) * length(param_order) * ...
                          length(param_num_sampling) * length(param_k) * length(param_rng);

%% 开始最外层数据集遍历
for data_idx = dataset_list
    
    % 强制回收上一轮的所有大型变量
    clear X Y all_results_mat result_cell F obj alphaA H B B1_cell C U;
    
    loop_idx = 1; 
    timestamp = datestr(now, 'yyyymmdd_HHMMSS'); 
    csvFileName = fullfile(resultsDir, sprintf('GridSearch_Data%d_%s.csv', data_idx, timestamp));
    matFileName = fullfile(resultsDir, sprintf('Data%d_%s.mat', data_idx, timestamp));
    
    % 预分配 .mat 的结构体空间
    empty_struct = struct('Anchors', [], 'Order', [], 'NumSampling', [], ...
        'K', [], 'Seed', [], 'ACC', [], 'NMI', [], 'Purity', [], 'Fscore', [], ...
        'Runtime', [], 'F_Labels', [], 'Obj_History', [], 'alphaA_History', []);
    all_results_mat = repmat(empty_struct, total_loops_per_dataset, 1);
    
    % 预分配 CSV 的内存元胞空间 (彻底移除循环内的 writetable!)
    result_cell = cell(total_loops_per_dataset, 16);
    
    % 加载当前数据集
    fprintf('\n======================================================\n');
    fprintf('>>> 正在加载并运行数据集编号: %d <<<\n', data_idx);
    fprintf('>>> 实验结束后，指标 CSV 将一次性保存至: %s\n', csvFileName);
    fprintf('>>> 实验结束后，全量 MAT 将一次性保存至: %s\n', matFileName);
    fprintf('======================================================\n');
    
    [X, Y] = loaddata(data_idx);
    [num, dim] = size(X);
    X = X ./ max(X, [], 2); % 归一化
    c = length(unique(Y));
    
    %% 开始网格参数搜索
    for ar = param_anchors_rate
        for ord = param_order
            for ns = param_num_sampling
                for k_val = param_k
                    for seed = param_rng
                        rng(seed);
                        anchors = [];
                        for t = 1:ns
                            anchors = [anchors, ar*c+(t-1)*delta*c];
                        end
                        
                        fprintf('进度: %d / %d (%.2f%%) | Data: %d | AR=%d, Ord=%d, NS=%d, K=%d, Seed=%d ... ', ...
                            loop_idx, total_loops_per_dataset, (loop_idx/total_loops_per_dataset)*100, data_idx, ar, ord, ns, k_val, seed);
                        
                        % 1. 运行核心实验
                        [F, obj, runtime, alphaA] = AHD_EC(k_val, ord, X, anchors, c);
                        fprintf("Over ... ");
                        
                        % 2. 评估聚类结果
                        [ACC, MIhat, Purity, Fscore, P, R, RI] = ClusteringMeasure4(Y, F);
                        
                        % 3. 将矩阵转化为字符串 (仅供 CSV 使用)
                        anchors_str = mat2str(anchors);
                        obj_str = mat2str(round(obj, 6)); 
                        alphaA_str = mat2str(round(alphaA, 6)); 
                        
                        result_cell(loop_idx, :) = {data_idx, anchors_str, ord, ns, k_val, seed, ...
                            ACC, MIhat, Purity, Fscore, P, R, RI, runtime, obj_str, alphaA_str};
                        
                        % 直接往预先挖好的"坑"里填数据
                        all_results_mat(loop_idx).Anchors = anchors;
                        all_results_mat(loop_idx).Order = ord;
                        all_results_mat(loop_idx).NumSampling = ns;
                        all_results_mat(loop_idx).K = k_val;
                        all_results_mat(loop_idx).Seed = seed;
                        all_results_mat(loop_idx).ACC = ACC;
                        all_results_mat(loop_idx).NMI = MIhat;
                        all_results_mat(loop_idx).Purity = Purity;
                        all_results_mat(loop_idx).Fscore = Fscore;
                        all_results_mat(loop_idx).Runtime = runtime;
                        all_results_mat(loop_idx).F_Labels = F;         
                        all_results_mat(loop_idx).Obj_History = obj;     
                        all_results_mat(loop_idx).alphaA_History = alphaA; 
                        
                        fprintf('Done.\n');
                        loop_idx = loop_idx + 1;
                        
                        % 清空不需要的临时大矩阵
                        clear F obj alphaA obj_str alphaA_str;
                    end
                end
            end
        end
    end
    
    % 循环彻底结束，执行唯一一次集中式硬盘 I/O 写入！
    fprintf('\n>>> 正在将数据集 %d 的指标导出至 CSV...\n', data_idx);
    varNames = {'DatasetID', 'Anchors', 'Order', 'NumSampling', 'K', 'Seed', ...
                'ACC', 'NMI', 'Purity', 'Fscore', 'P', 'R', 'RI', 'Runtime', ...
                'Obj_History', 'alphaA_History'};
    result_table = cell2table(result_cell, 'VariableNames', varNames);
    writetable(result_table, csvFileName); % 整个数据集只写 1 次！
    
    fprintf('>>> 正在打包保存数据集 %d 的全量 .mat 数据...\n', data_idx);
    save(matFileName, 'all_results_mat', 'X', 'Y', 'c', '-v7.3'); 
    
    % 跑完一个数据集后，执行终极清理
    clear all_results_mat result_cell result_table; 
    fclose('all');         
    
    fprintf('>>> 数据集 %d 运行与保存完毕！内存已清空！\n', data_idx);
end

fprintf('\n 所有数据集实验全部结束！CSV 与 MAT 已稳妥保存！\n');