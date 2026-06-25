% =========================================================================
% FCAG 算法自动化测试脚本 (顶刊 Mean±Std 格式 + 5指标全面评估 + OOM防御)
% 适配 12 个数据集、候选锚点数遍历、ClusteringMeasure4 极限版、20次随机种子
% =========================================================================

clc; clear; close all;

%% 1. 全局超参数与环境配置
addpath('func');

datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                 'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
numDatasets = length(datasetNames);

numNeighbor = 10; % 原论文固定为 10
m_candidates = [128, 256, 512, 1024]; % 候选锚点数
numRuns = 20; % 独立运行次数 (严格对齐20次)

csvFileName = fullfile(pwd, 'Baseline_FCAG_5Metrics_20Seeds_MeanStd.csv');

%% 2. 初始化 CSV 文件 (全局句柄统一管理)
% 修改后的代码（强行指定 GBK 编码）
fid = fopen(csvFileName, 'w', 'n', 'GBK');
if fid == -1
    error('无法创建或打开 CSV 文件，请检查路径或文件是否被占用。');
end

% 动态生成包含 Mean±Std 的顶级学术表头
header = {'Dataset_Name', 'numNeighbor', 'm_Anchors', ...
          'FCAG_ACC(Mean±Std)', 'FCAG_NMI(Mean±Std)', 'FCAG_PUR(Mean±Std)', ...
          'FCAG_Fscore(Mean±Std)', 'FCAG_ARI(Mean±Std)', ...
          'FCAG_Time_iter(s)', 'FCAG_Time_total(s)'};
fprintf(fid, '%s\n', strjoin(header, ','));

disp('====================================================================');
disp('>>> 启动 FCAG 自动化基准测试 (带 Mean±Std 统计计算)');
disp(['>>> 共计 ', num2str(numDatasets), ' 个数据集，每个锚点配置独立运行 ', num2str(numRuns), ' 个种子']);
disp('====================================================================');

%% 3. 主实验 Pipeline 遍历数据集
for d_idx = 1:numDatasets
    dataname = datasetNames{d_idx};
    
    try
        % 1. 接入数据加载器
        [X, Y] = loaddata(d_idx); 
    catch ME
        warning('加载数据集 %s 失败，请检查 ec_data 文件夹。跳过...', dataname);
        continue;
    end
    
    [N, dim] = size(X);
    nC = length(unique(Y));

    X = double(X);
    % 增加 [0,1] 归一化
    X_min = min(X, [], 1);
    X_max = max(X, [], 1);
    X_range = X_max - X_min;
    X_range(X_range == 0) = 1; 
    X = (X - X_min) ./ X_range;
    
    fprintf('\n>>> 正在处理: Dataset %02d (%s) | 样本数 N=%d, 类别数 K=%d\n', d_idx, dataname, N, nC);
    
    for m = m_candidates
        if m >= N
            fprintf('  [跳过] 候选锚点数 m=%d 大于等于样本数 N=%d\n', m, N);
            continue;
        end        
        
        numAnchor = fix(log2(m));
        generateAnchor = 1;  % BKHK
        
        is_oom = false;
        
        try
            % 提前生成图 
            T_graph_start = tic; 
            [B, M_anchor] = AnchorGEN(X, numAnchor, numNeighbor, generateAnchor);
            time_graph = toc(T_graph_start); 
            
            % 存放 20 次随机种子的结果：[ACC, NMI, PUR, Fscore, ARI, Time_iter]
            res_fcag = zeros(numRuns, 6);
            
            for runIdx = 1:numRuns
                % 严格控制每次循环的随机种子
                rng(runIdx * 1000 + d_idx, 'twister'); 
                
                Tstart = tic;
                ym = kmeans(M_anchor, nC, 'EmptyAction', 'singleton'); 
                U0 = n2nc(ym);
                
                [y_pred, ~, ~, ~, ~, ~] = FCAG(B, U0);
                time_iter = toc(Tstart);
                
                % 接入极限版测评函数
                [ACC, NMI, PUR, Fscore, ~, ~, ~, ARI] = ClusteringMeasure4(Y, y_pred); 
                
                res_fcag(runIdx, :) = [ACC, NMI, PUR, Fscore, ARI, time_iter];
            end
            
        catch ME
            is_oom = true;
            fprintf('    [触发熔断] m=%d 时发生崩溃或内存溢出: %s\n', m, ME.message);
        end
        
        % =========================================================
        % 4. 统计与写入 (Mean ± Std 核心逻辑)
        % =========================================================
        if is_oom
            % 如果发生异常，指标全填 OOM
            row_data_str = repmat({'OOM'}, 1, 7); 
            fprintf(fid, '%s,%d,%d,%s\n', dataname, numNeighbor, m, strjoin(row_data_str, ','));
        else
            % 正常跑完 20 个 seed，计算均值 (Mean) 和无偏标准差 (Std)
            avg_metrics = mean(res_fcag, 1); 
            std_metrics = std(res_fcag, 0, 1); % 0 表示无偏估计(除以N-1)
            
            avg_time_total = time_graph + avg_metrics(6); % 加上恒定的建图时间
            
            % 将 5 个指标格式化为 'Mean±Std' 的字符串数组
            str_metrics = cell(1, 5);
            for i = 1:5
                str_metrics{i} = sprintf('%.4f±%.4f', avg_metrics(i), std_metrics(i));
            end
            
            % 写入 CSV (指标用 %s 占位接收拼接好的字符串，时间用 %.4f)
            fprintf(fid, '%s,%d,%d,%s,%s,%s,%s,%s,%.4f,%.4f\n', ...
                    dataname, numNeighbor, m, ...
                    str_metrics{1}, str_metrics{2}, str_metrics{3}, str_metrics{4}, str_metrics{5}, ...
                    avg_metrics(6), avg_time_total);
            
            % 控制台打印时也展示方差，让你对算法稳定性心中有数
            fprintf('  -> m=%-4d | ACC: %s | NMI: %s | T_total: %.4fs\n', ...
                    m, str_metrics{1}, str_metrics{2}, avg_time_total);
        end
    end
end

fclose(fid);

fprintf('\n====================================================================\n');
fprintf('[SUCCESS] 所有实验运行完毕！结果已保存至:\n%s\n', csvFileName);
fprintf('====================================================================\n');

rmpath('func');