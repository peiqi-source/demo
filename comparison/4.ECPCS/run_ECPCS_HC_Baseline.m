% =========================================================================
% ECPCS-HC 对比算法自动化测试脚本 (顶刊 Mean±Std 格式 + 5指标全面评估 + OOM防御)
% 适配 12 个数据集、100基聚类池、ClusteringMeasure4 极限版
% =========================================================================

clear all;
close all;
clc;

%% 1. 全局超参数与接口设置
M = 20;             % [接口]: 从100个基聚类池中挑选的基聚类数量 M
numRuns = 20;       % 独立运行次数 (Seed数量，用于稳健性测试)
t = 20;             % ECPCS-HC 内部随机游走步长 (论文默认 20)

% 路径设置对齐，确保使用统一的数据集文件命名格式
folderPath = 'Base_LabelsPool_MAT_Allsample';
datasetNames = {'Umist', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                'MNIST', 'OpticDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
numDatasets = length(datasetNames);

% 更新了输出文件后缀，标明使用了 MeanStd 统计
csvFileName = sprintf('Baseline_ECPCS_HC_5Metrics_M%d_20Seeds_MeanStd.csv', M);

%% 2. 初始化 CSV 文件 (强制 GBK 编码防止 ± 乱码)
fid = fopen(csvFileName, 'w', 'n', 'GBK');
if fid == -1
    error('无法创建或打开 CSV 文件，请检查路径或文件是否被占用。');
end

% 动态生成包含 Mean±Std 的顶级学术表头
header = {'Dataset_Name', 'ECPCS_ACC(Mean±Std)', 'ECPCS_NMI(Mean±Std)', ...
          'ECPCS_PUR(Mean±Std)', 'ECPCS_Fscore(Mean±Std)', 'ECPCS_ARI(Mean±Std)', 'ECPCS_Runtime(s)'};
fprintf(fid, '%s\n', strjoin(header, ','));

disp('====================================================================');
disp(['>>> 启动 ECPCS-HC 自动化基准测试 (带 Mean±Std 统计) | 选取 M = ', num2str(M)]);
disp(['>>> 共计 ', num2str(numDatasets), ' 个数据集，每个独立运行 ', num2str(numRuns), ' 个种子']);
disp('====================================================================');

%% 3. 遍历 12 个数据集
for d_idx = 1:numDatasets
    currentDatasetName = datasetNames{d_idx};
    fileName = sprintf('LabelsPool_%s.mat', currentDatasetName);
    filePath = fullfile(folderPath, fileName);
    
    if ~exist(filePath, 'file')
        warning('未找到文件: %s，已跳过。', filePath);
        continue;
    end
    
    % 加载数据
    load(filePath, 'base_labels', 'Y');
    
    % 数据维度自适应处理 (转置为 N * M)
    if size(base_labels, 1) > size(base_labels, 2)
        all_baseCls = base_labels;   
    else
        all_baseCls = base_labels';  
    end
    
    [N, total_M] = size(all_baseCls);
    K = numel(unique(Y));
    
    fprintf('\n>>> 正在处理: Dataset %02d (%s) | 样本数 N=%d, 类别数 K=%d\n', d_idx, currentDatasetName, N, K);
    
    % 用于存放单数据集的各个 seed 结果：[ACC, NMI, PUR, Fscore, ARI, Runtime]
    res_ecpcs = zeros(numRuns, 6);
    is_oom = false; % OOM 及异常标志位
    
    %% 4. 独立重复实验与基聚类抽样 (带 try-catch 保护)
    try
        for runIdx = 1:numRuns
            % 严格控制每次循环的随机种子，保证可复现
            rng(runIdx * 1000 + d_idx, 'twister'); 

            % 从 total_M (100) 个基聚类中无放回随机挑选 M 个
            selected_indices = randperm(total_M, M);
            selected_baseCls = all_baseCls(:, selected_indices);
            
            % 运行 ECPCS-HC 算法并独立计时
            tic;
            Label = ECPCS_HC(selected_baseCls, K, t);
            run_time = toc;
            
            % 接入极限版测评函数 (提取 5 个指标)
            [ACC, NMI, PUR, Fscore, ~, ~, ~, ARI] = ClusteringMeasure4(Y, Label);
            
            res_ecpcs(runIdx, :) = [ACC, NMI, PUR, Fscore, ARI, run_time];
        end
    catch ME
        % 捕捉到内存溢出或计算异常，直接熔断
        is_oom = true;
        fprintf('    [触发熔断] 该数据集发生崩溃或内存溢出: %s\n', ME.message);
    end
    
    %% 5. 结果结算与 CSV 写入 (Mean ± Std 核心逻辑)
    if is_oom
        % 如果发生异常，全行写入 OOM
        row_data_str = repmat({'OOM'}, 1, 6); 
        fprintf(fid, '%s,%s\n', currentDatasetName, strjoin(row_data_str, ','));
    else
        % 正常跑完 20 个 seed，计算均值 (Mean) 和无偏标准差 (Std)
        avg_metrics = mean(res_ecpcs, 1);
        std_metrics = std(res_ecpcs, 0, 1);
        
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
                
        % 控制台精简输出，便于观察算法稳定性
        fprintf('    完成! ACC: %s | NMI: %s | T: %.4fs\n', ...
            str_metrics{1}, str_metrics{2}, avg_metrics(6));
    end
end

fclose(fid);
disp('====================================================================');
fprintf('[SUCCESS] 所有数据集测试完毕！\n对比基准已输出至文件: %s\n', fullfile(pwd, csvFileName));
disp('====================================================================');