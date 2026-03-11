%% exp_parameter_write.m 性能指标提升实验 (分数据集保存 CSV 与 MAT，修正进度提示)
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
dataset_list = 1:12; 

%% 参数定义 (网格搜索空间)
param_anchors_rate = [10 12];
param_order = 3:6;
param_num_sampling = 3:7;
param_k = [5 7 10 15];
param_rng = 2:2:16;
delta = 6;

% 计算单个数据集的总循环次数，用于重置进度提示
total_loops_per_dataset = length(param_anchors_rate) * length(param_order) * ...
                          length(param_num_sampling) * length(param_k) * length(param_rng);

%% 开始最外层数据集遍历
for data_idx = dataset_list

    % 清空上一轮数据集在内存中暂存的结构体
    clear X Y all_results_mat result_data F obj alphaA H B B1_cell C U;

    % 每次进入新数据集，进度清零
    loop_idx = 1; 
    
    % 为当前数据集生成专属的 CSV 和 MAT 文件名
    timestamp = datestr(now, 'yyyymmdd_HHMMSS'); 
    csvFileName = fullfile(resultsDir, sprintf('GridSearch_Data%d_%s.csv', data_idx, timestamp));
    matFileName = fullfile(resultsDir, sprintf('Data%d_%s.mat', data_idx, timestamp));
    
    % 是否更新表头
    isFirst = true; 
    
    % 加载当前数据集
    fprintf('\n======================================================\n');
    fprintf('>>> 正在加载并运行数据集编号: %d <<<\n', data_idx);
    fprintf('>>> 指标 CSV 将实时保存至: %s\n', csvFileName);
    fprintf('>>> 全量 MAT 将在跑完后保存至: %s\n', matFileName);
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
                        
                        % --- 进度提示 (按单个数据集计算) ---
                        fprintf('进度: %d / %d (%.2f%%) | Data: %d | AR=%d, Ord=%d, NS=%d, K=%d, Seed=%d ... ', ...
                            loop_idx, total_loops_per_dataset, (loop_idx/total_loops_per_dataset)*100, data_idx, ar, ord, ns, k_val, seed);
                        
                        % 1. 运行核心实验
                        [F, obj, runtime, alphaA] = AHD_EC(k_val, ord, X, anchors, c);
                        
                        % 2. 评估聚类结果
                        [ACC, MIhat, Purity, Fscore, P, R, RI] = ClusteringMeasure2(Y, F);
                        
                        % 3. 将矩阵转化为字符串 (仅供 CSV 使用)
                        obj_str = mat2str(round(obj, 6)); 
                        alphaA_str = mat2str(round(alphaA, 6)); 
                        
                        % 4. 构建单行表格
                        result_data = table(data_idx, ar, ord, ns, k_val, seed, ...
                            ACC, MIhat, Purity, Fscore, P, R, RI, runtime, ...
                            {obj_str}, {alphaA_str}, ...
                            'VariableNames', {'DatasetID', 'AnchorsRate', 'Order', 'NumSampling', 'K', 'Seed', ...
                            'ACC', 'NMI', 'Purity', 'Fscore', 'P', 'R', 'RI', 'Runtime', ...
                            'Obj_History', 'alphaA_History'});
                        
                        % 5. 追加写入当前数据集的 CSV
                        if isFirst
                            writetable(result_data, csvFileName, 'WriteMode', 'overwrite');
                            isFirst = false;
                        else
                            writetable(result_data, csvFileName, 'WriteMode', 'append', 'WriteVariableNames', false);
                        end
                        
                        % 将每一次实验的 "原生矩阵格式" 塞进结构体数组暂存
                        all_results_mat(loop_idx).AnchorsRate = ar;
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
                    end
                end
            end
        end
    end
    
    % 当一个数据集的所有参数组合跑完后，把结构体数组连同原始特征 X 和 Y 一把存进 .mat
    fprintf('>>> 正在打包保存数据集 %d 的全量 .mat 数据...\n', data_idx);
    save(matFileName, 'all_results_mat', 'X', 'Y', 'c', '-v7.3');
    clear all_results_mat; % 立刻释放可能高达几个G的结构体
    fclose('all');         % 强行释放所有底层文件读写句柄，防 I/O 阻塞
    fprintf('>>> 数据集 %d 运行与保存完毕！\n', data_idx);
    
end

fprintf('\n所有数据集实验全部结束！CSV 与 MAT 已稳妥保存！\n');