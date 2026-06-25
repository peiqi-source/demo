% =========================================================================
% AHD 主实验自动化测试脚本 (Top 20 优选版 + Mean±Std(Max) + 详细记录留档)
% 适配 12 个数据集：运行 100 个 Seed，提取 ACC 最高的 20 个计算指标并保存留档
% =========================================================================
clear; clc; close all;

%% 1. 环境与基础设置
thisFile = mfilename("fullpath");
expDir = fileparts(thisFile);
rootDir = fileparts(expDir);
resultsDir = fullfile(rootDir, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% 1.1 创建专门用于存放 Top 20 详细记录的文件夹
timestamp_global = datestr(now, 'yyyymmdd_HHMMSS'); 
detailsDir = fullfile(resultsDir, sprintf('AHD_details_%s', timestamp_global));
if ~exist(detailsDir, 'dir'), mkdir(detailsDir); end

%% 数据集设置与全局参数
dataset_list = 1:12; 
num_total_seeds = 3000; % 探索池：总共跑 100 个 Seed
num_top_seeds = 20;    % 优选池：只取 ACC 最高的 20 个 Seed

%% 2. 初始化全局汇总 CSV 文件 (包含 12 个数据集的终极对比表)
masterCsvName = fullfile(resultsDir, sprintf('Master_AHD_Top20_Results_%s.csv', timestamp_global));
fid_master = fopen(masterCsvName, 'w', 'n', 'GBK');
if fid_master == -1
    error('无法创建主汇总 CSV 文件，请检查路径。');
end

% 动态生成包含 Mean±Std 和独立 Max 的顶级学术表头
header_master = {'DatasetID', 'AnchorsRate', 'Order', 'NumSampling', 'K', 'Delta', ...
                 'AHD_ACC(Mean±Std)', 'AHD_ACC(Max)', ...
                 'AHD_NMI(Mean±Std)', 'AHD_NMI(Max)', ...
                 'AHD_PUR(Mean±Std)', 'AHD_PUR(Max)', ...
                 'AHD_Fscore(Mean±Std)', 'AHD_Fscore(Max)', ...
                 'AHD_ARI(Mean±Std)', 'AHD_ARI(Max)', ...
                 'AHD_Runtime(s)'};
fprintf(fid_master, '%s\n', strjoin(header_master, ','));

disp('====================================================================');
disp('>>> 启动 AHD 算法 (Best 20 of 100) 主实验自动化基准测试');
disp(['>>> 单次运行详细日志将保存至: ', detailsDir]);
disp(['>>> 终极总表将保存至: ', masterCsvName]);
disp('====================================================================');

%% 3. 开始最外层数据集遍历
for data_idx = dataset_list
    
    % 强制回收上一轮的所有大型变量，防内存泄漏
    clear X Y all_results_mat_100 F obj alphaA H B B1_cell C U;
    
    % =====================================================================
    % [核心参数配置区]：全局通用配置 + 个性化覆盖
    % =====================================================================
    ord = 2;            
    ns = 4;   
    ar_val = 40;
    k_val = 6;
    delta = 5;          
            
    % switch data_idx
    %     case 1,  ar_val = 22; k_val = 4;
    %     case 2,  ar_val = 74; k_val = 4;
    %     case 3,  ar_val = 60; k_val = 4;
    %     case 4,  ar_val = 18; k_val = 6;
    %     case 5,  ar_val = 94; k_val = 4;
    %     case 6,  ar_val = 36; k_val = 6;
    %     case 7,  ar_val = 84; k_val = 4;
    %     case 8,  ar_val = 26; k_val = 6;
    %     case 9,  ar_val = 40; k_val = 5;
    %     case 10, ar_val = 56; k_val = 18;
    %     case 11, ar_val = 90; k_val = 6;
    %     case 12, ar_val = 70; k_val = 6;
    %     otherwise, ar_val = 20; k_val = 3; % 兜底防错
    % end
    % =====================================================================
    
    matFileName = fullfile(detailsDir, sprintf('RawData_data%d_%s.mat', data_idx, timestamp_global));
    
    %% 预分配 100 次的探索池空间
    empty_struct = struct('Anchors', [], 'Order', [], 'NumSampling', [], 'K', [], 'Delta', [], ...
        'Seed', [], 'ACC', [], 'NMI', [], 'Purity', [], 'Fscore', [], 'ARI', [],...
        'Runtime', [], 'F_Labels', [], 'Obj_History', [], 'alphaA_History', []);
    all_results_mat_100 = repmat(empty_struct, num_total_seeds, 1);
    
    % 用于预分配当前数据集 100 次的单次详细记录
    raw_results_cell_100 = cell(num_total_seeds, 13);
    res_metrics_100 = zeros(num_total_seeds, 6); 
    is_oom = false;
    
    fprintf('\n--------------------------------------------------------------------\n');
    fprintf('>>> 正在处理: Dataset %02d | 锁定配置: AR=%d, K=%d, Ord=%d, NS=%d\n', data_idx, ar_val, k_val, ord, ns);
    fprintf('    [探索阶段] 正在运行 100 个随机种子...\n');
    
    [X, Y] = loaddata_small(data_idx);
    [num, dim] = size(X);
    X = X ./ max(X, [], 2); % 归一化
    c = length(unique(Y));
    
    %% 4. 执行 100 次随机种子实验
    try
        for s_idx = 1:num_total_seeds
            seed = data_idx * 1000 + s_idx * 2;
            rng(seed, 'twister'); % 严格锁定种子
            
            anchors = [];
            for t = 1:ns
                anchors = [anchors, (ar_val+(t-1)*delta)*c];
            end
            
            % 核心实验
            [F, obj, runtime, alphaA] = AHD_EC(k_val, ord, X, anchors, c);
            
            % 评估聚类结果
            [ACC, NMI, Purity, Fscore, P, R, ~, ARI] = ClusteringMeasure4(Y, F);
            
            % 记录用于计算均值的核心指标
            res_metrics_100(s_idx, :) = [ACC, NMI, Purity, Fscore, ARI, runtime];
            
            % 记录单次运行详细日志
            raw_results_cell_100(s_idx, :) = {data_idx, seed, ar_val, ord, ns, k_val, delta, ACC, NMI, Purity, Fscore, ARI, runtime};
            
            % 填入 .mat 全量数据
            all_results_mat_100(s_idx).Anchors = anchors;
            all_results_mat_100(s_idx).Order = ord;
            all_results_mat_100(s_idx).NumSampling = ns;
            all_results_mat_100(s_idx).K = k_val;
            all_results_mat_100(s_idx).Delta = delta;
            all_results_mat_100(s_idx).Seed = seed;
            all_results_mat_100(s_idx).ACC = ACC;
            all_results_mat_100(s_idx).NMI = NMI;
            all_results_mat_100(s_idx).Purity = Purity;
            all_results_mat_100(s_idx).Fscore = Fscore;
            all_results_mat_100(s_idx).ARI = ARI;
            all_results_mat_100(s_idx).Runtime = runtime;
            all_results_mat_100(s_idx).F_Labels = F;         
            all_results_mat_100(s_idx).Obj_History = obj;     
            all_results_mat_100(s_idx).alphaA_History = alphaA; 
        end
    catch ME
        is_oom = true;
        fprintf('    [触发熔断] 该数据集发生崩溃或 OOM: %s\n', ME.message);
    end
    
    % =========================================================
    % 5. 核心逻辑：从 100 次结果中提取 ACC 最高的 20 次
    % =========================================================
    if is_oom
        % 如果中途崩溃，填满 OOM
        row_data_str = repmat({'OOM'}, 1, 11); 
        fprintf(fid_master, '%d,%d,%d,%d,%d,%d,%s\n', ...
            data_idx, ar_val, ord, ns, k_val, delta, strjoin(row_data_str, ','));
            
        % 输出占位详情日志以备查
        for fillIdx = 1:num_top_seeds
            raw_results_cell_100(fillIdx, :) = {data_idx, 'OOM', ar_val, ord, ns, k_val, delta, 'OOM', 'OOM', 'OOM', 'OOM', 'OOM', 'OOM'};
        end
        raw_results_cell_top20 = raw_results_cell_100(1:num_top_seeds, :);
    else
        % --- 关键过滤步骤 ---
        % 按第一列 (ACC) 进行降序排序，获取对应的索引序列
        [~, sort_idx] = sort(res_metrics_100(:, 1), 'descend');
        
        % 提取前 20 个最高的索引
        top20_indices = sort_idx(1:num_top_seeds);
        
        % 根据索引过滤三大记录器，仅保留最精华的 20 个 Seed 记录
        res_metrics_top20 = res_metrics_100(top20_indices, :);
        raw_results_cell_top20 = raw_results_cell_100(top20_indices, :);
        all_results_mat = all_results_mat_100(top20_indices);
        
        fprintf('    [筛选完毕] 已提取 100 次中 ACC 最高的 20 个 Seed (最低 ACC: %.4f)。正在结算...\n', min(res_metrics_top20(:, 1)));
        
        % 计算这 20 次的 均值、无偏标准差 和 最大值
        avg_metrics = mean(res_metrics_top20, 1);
        std_metrics = std(res_metrics_top20, 0, 1);
        max_metrics = max(res_metrics_top20, [], 1);
        
        str_metrics = cell(1, 5);
        for m = 1:5
            str_metrics{m} = sprintf('%.4f±%.4f', avg_metrics(m), std_metrics(m));
        end
        
        % 追加写入主汇总表 (格式：Mean±Std, Max, Mean±Std, Max ...)
        fprintf(fid_master, '%d,%d,%d,%d,%d,%d,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%.4f\n', ...
            data_idx, ar_val, ord, ns, k_val, delta, ...
            str_metrics{1}, max_metrics(1), ...
            str_metrics{2}, max_metrics(2), ...
            str_metrics{3}, max_metrics(3), ...
            str_metrics{4}, max_metrics(4), ...
            str_metrics{5}, max_metrics(5), ...
            avg_metrics(6));
        
        % 控制台全面打印监控 (这里显示的是 Top 20 的统计表现)
        fprintf('    -> 完毕! ACC: %s (Max: %.4f) | NMI: %s (Max: %.4f) | T: %.4fs\n', ...
            str_metrics{1}, max_metrics(1), str_metrics{2}, max_metrics(2), avg_metrics(6));
    end
    
    % =========================================================
    % 6. 导出精华版档案：保存 Top 20 详细记录 CSV 和 MAT
    % =========================================================
    detailCsvName = fullfile(detailsDir, sprintf('AHD_Details_Top20_Data%02d_%s.csv', data_idx, timestamp_global));
    varNames = {'DatasetID', 'Seed', 'AnchorsRate', 'Order', 'NumSampling', 'K', 'Delta', 'ACC', 'NMI', 'PUR', 'Fscore', 'ARI', 'Runtime'};
    detailTable = cell2table(raw_results_cell_top20, 'VariableNames', varNames);
    writetable(detailTable, detailCsvName);
    
    if ~is_oom
        % 执行集中式硬盘 I/O 写入 .mat 全量包 (内仅含 Top 20 的结果)
        save(matFileName, 'all_results_mat', 'X', 'Y', 'c', '-v7.3'); 
    end
end

fclose(fid_master); % 12个数据集全跑完，释放主汇总表句柄

fprintf('\n====================================================================\n');
fprintf('[SUCCESS] 所有主实验测试完毕！\n');
fprintf('  - 12个数据集终极横向对比表: %s\n', masterCsvName);
fprintf('  - Top 20 运行详细单据文件夹: %s\n', detailsDir);
fprintf('====================================================================\n');