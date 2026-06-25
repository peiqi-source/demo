% =========================================================================
% PTA & PTGP 算法主实验脚本 (20 Seeds 标准对比版 + Mean±Std(Max) + 详细记录留档)
% 适配 12 个数据集：从池中抽取指定 M 个基聚类，运行 20 次 Seed 统计指标
% 论文来源: TKDE 2016 "Robust Ensemble Clustering Using Probability Trajectories"
% =========================================================================

clear; clc; close all;

%% 1. 环境与基础设置
folderPath = 'Base_LabelsPool_MAT_Allsample'; % 基聚类池路径
resultsDir = fullfile(pwd, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% 创建专门用于存放 20 次详细运行记录的文件夹
detailsDir = fullfile(resultsDir, 'PTA_PTGP_details');
if ~exist(detailsDir, 'dir'), mkdir(detailsDir); end

datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
numDatasets = length(datasetNames);

%% 全局超参数
M = 20;                % [接口]: 集成规模 (从 100 个池子中抽取的数量)
num_seeds = 20;        % 运行次数
timestamp_global = datestr(now, 'yyyymmdd_HHMMSS'); 

%% 2. 初始化全局汇总 CSV 文件 (包含 12 个数据集的终极对比表)
masterCsvName = fullfile(resultsDir, sprintf('Master_PTA_PTGP_Results_%s.csv', timestamp_global));
fid_master = fopen(masterCsvName, 'w', 'n', 'GBK');
if fid_master == -1
    error('无法创建主汇总 CSV 文件，请检查路径。');
end

% 动态生成包含 Mean±Std 和独立 Max 的顶级学术表头
header_master = {'Dataset_Name', 'Method', 'M_Size', ...
                 'ACC(Mean±Std)', 'ACC(Max)', 'NMI(Mean±Std)', 'NMI(Max)', ...
                 'PUR(Mean±Std)', 'PUR(Max)', 'Fscore(Mean±Std)', 'Fscore(Max)', ...
                 'ARI(Mean±Std)', 'ARI(Max)', 'Runtime(s)'};
fprintf(fid_master, '%s\n', strjoin(header_master, ','));

disp('====================================================================');
disp('>>> 启动 PTA & PTGP 算法标准主实验对比测试');
disp(['>>> 详细日志将保存至: ', detailsDir]);
disp(['>>> 终极总表将保存至: ', masterCsvName]);
disp('====================================================================');

%% 3. 开始遍历数据集
for d_idx = 1:numDatasets
    currentDatasetName = datasetNames{d_idx};
    fileName = sprintf('LabelsPool_%s.mat', currentDatasetName);
    fullPath = fullfile(folderPath, fileName);
    
    if ~exist(fullPath, 'file')
        warning('未找到文件: %s，跳过。', fullPath);
        continue;
    end
    
    % 加载数据
    data = load(fullPath);
    base_labels = data.base_labels; % 尺寸: 100 x N 或 N x 100
    Y = data.Y;                     % 真实标签
    
    if size(base_labels, 1) > size(base_labels, 2)
        base_labels = base_labels'; % 确保格式为 M_total x N
    end
    [poolSize, N] = size(base_labels); 
    trueK = numel(unique(Y)); 
    
    % 锁定真实类别数对应的索引
    clsNums = [2:20, 25:5:50];
    clsNums = unique([clsNums, trueK]); 
    trueKidx = find(clsNums == trueK);
    
    % 预分配 4 个算法的性能池: [ACC, NMI, PUR, Fscore, ARI, Runtime]
    res_AL = zeros(num_seeds, 6);
    res_CL = zeros(num_seeds, 6);
    res_SL = zeros(num_seeds, 6);
    res_PTGP = zeros(num_seeds, 6);
    
    % 用于详细记录留档的 Cell (4个算法 x 20行 = 80行)
    raw_detail_cell = cell(num_seeds * 4, 9);
    row_ptr = 1;
    
    fprintf('\n>>> 正在处理: Dataset %02d (%s) | N=%d, K=%d\n', d_idx, currentDatasetName, N, trueK);
    
    %% 4. 执行 20 次 Seed 实验
    try
        for s_idx = 1:num_seeds
            seed = d_idx * 100 + s_idx * 100;
            rng(seed, 'twister'); 
            
            % 从池子中随机挑选 M 个基聚类并转置为 N x M
            selected_idx = randperm(poolSize, M);
            baseCls = base_labels(selected_idx, :)'; 

            % % 要让每种对比算法都用相同的20个标签
            % baseCls = base_labels(1:4:80, :)';
            
            % ---------------- 核心算法步骤与计时 ----------------
            tic; [mcBaseCls, mcLabels] = computeMicroclusters(baseCls); t1 = toc;
            tilde_N = size(mcBaseCls, 1);
            
            tic; MCA = computeMCA(mcBaseCls); t2 = toc;
            
            para.K = min(20, floor(sqrt(tilde_N)/2));
            para.T = min(20, floor(sqrt(tilde_N)/2));
            
            tic; PTS = computePTS_fast_v3(MCA, mcLabels, para); t3 = toc;
            tic; [mcResultsAL, mcResultsCL, mcResultsSL] = runPTA_v2(PTS, clsNums); t4 = toc;
            tic; mcResultsPTGP = runPTGP_v2(mcBaseCls, PTS, clsNums); t5 = toc;
            
            tic;
            resultsAL = mapMicroclustersBackToObjects(mcResultsAL, mcLabels);
            resultsCL = mapMicroclustersBackToObjects(mcResultsCL, mcLabels);
            resultsSL = mapMicroclustersBackToObjects(mcResultsSL, mcLabels);
            resultsPTGP = mapMicroclustersBackToObjects(mcResultsPTGP, mcLabels);
            t6 = toc;
            
            % 结算各算法总耗时
            t_common = t1 + t2 + t3 + t6;
            time_AL = t_common + t4;
            time_PTGP = t_common + t5;
            
            % ---------------- 提取指标 (TrueK) ----------------
            [a1, n1, p1, f1, ~, ~, ~, ar1] = ClusteringMeasure4(Y, resultsAL(:, trueKidx));
            [a2, n2, p2, f2, ~, ~, ~, ar2] = ClusteringMeasure4(Y, resultsCL(:, trueKidx));
            [a3, n3, p3, f3, ~, ~, ~, ar3] = ClusteringMeasure4(Y, resultsSL(:, trueKidx));
            [a4, n4, p4, f4, ~, ~, ~, ar4] = ClusteringMeasure4(Y, resultsPTGP(:, trueKidx));
            
            % 存入统计池
            res_AL(s_idx, :)   = [a1, n1, p1, f1, ar1, time_AL];
            res_CL(s_idx, :)   = [a2, n2, p2, f2, ar2, time_AL];
            res_SL(s_idx, :)   = [a3, n3, p3, f3, ar3, time_AL];
            res_PTGP(s_idx, :) = [a4, n4, p4, f4, ar4, time_PTGP];
            
            % 存入单次详细记录
            raw_detail_cell(row_ptr:row_ptr+3, :) = { ...
                currentDatasetName, 'PTA-AL', seed, a1, n1, p1, f1, ar1, time_AL; ...
                currentDatasetName, 'PTA-CL', seed, a2, n2, p2, f2, ar2, time_AL; ...
                currentDatasetName, 'PTA-SL', seed, a3, n3, p3, f3, ar3, time_AL; ...
                currentDatasetName, 'PTGP',   seed, a4, n4, p4, f4, ar4, time_PTGP };
            row_ptr = row_ptr + 4;
        end
        
        %% 5. 结果结算与写入总表
        methods_name = {'PTA-AL', 'PTA-CL', 'PTA-SL', 'PTGP'};
        res_pool = {res_AL, res_CL, res_SL, res_PTGP};
        
        for m_idx = 1:4
            cur_res = res_pool{m_idx};
            avg_m = mean(cur_res, 1);
            std_m = std(cur_res, 0, 1);
            max_m = max(cur_res, [], 1);
            
            str_stats = cell(1, 5);
            for i = 1:5
                str_stats{i} = sprintf('%.4f±%.4f', avg_m(i), std_m(i));
            end
            
            % 写入汇总 CSV: Dataset, Method, M, 指标(Mean±Std, Max)..., Runtime
            fprintf(fid_master, '%s,%s,%d,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%.4f\n', ...
                currentDatasetName, methods_name{m_idx}, M, ...
                str_stats{1}, max_m(1), str_stats{2}, max_m(2), ...
                str_stats{3}, max_m(3), str_stats{4}, max_m(4), ...
                str_stats{5}, max_m(5), avg_m(6));
        end
        
        % 保存当前数据集的详细 CSV 记录
        detailCsv = fullfile(detailsDir, sprintf('PTA_PTGP_Details_%s_%s.csv', currentDatasetName, timestamp_global));
        varNames = {'Dataset', 'Method', 'Seed', 'ACC', 'NMI', 'PUR', 'Fscore', 'ARI', 'Runtime'};
        detailTable = cell2table(raw_detail_cell, 'VariableNames', varNames);
        writetable(detailTable, detailCsv);
        
        fprintf('    [OK] 20次运行统计完毕，已写入总表与详细日志。\n');
        
    catch ME
        fprintf('    [Error] 数据集 %s 处理失败: %s\n', currentDatasetName, ME.message);
        % 写入 OOM 占位
        for m_idx = 1:4
             fprintf(fid_master, '%s,%s,%d,%s\n', currentDatasetName, methods_name{m_idx}, M, repmat('OOM,', 1, 10));
        end
    end
end

fclose(fid_master);
disp('====================================================================');
fprintf('[SUCCESS] PTA/PTGP 实验全部结束！\n汇总表：%s\n', masterCsvName);
disp('====================================================================');