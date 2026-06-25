% =========================================================================
% run_AHD_Ablation_Frontend12.m
% Front-end ablation study for AHD-EC on 12 datasets.
%
% Variants:
%   1) w/o HO  : remove high-order graph generation, only first-order graphs
%   2) w/o MS  : remove multi-anchor/multi-scale sampling, only first anchor scale
%   3) w/o AW  : remove adaptive weighting in ADCF, use uniform weights
%   4) Full    : complete AHD-EC
%
% Output:
%   - One folder containing detailed per-dataset running records
%   - One master CSV summarizing ACC, NMI, ARI and runtime
%
% Note:
%   This script follows the automation/logging style of
%   run_AHD_HyperparametersTuning.m, but reports all repeated runs instead
%   of selecting the best/top seeds, which is more appropriate for ablation.
% =========================================================================

clear; clc; close all;

%% 1. Environment and output folders
thisFile = mfilename('fullpath');
if isempty(thisFile)
    expDir = pwd;
else
    expDir = fileparts(thisFile);
end
rootDir = fileparts(expDir);

resultsDir = fullfile(pwd, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

timestamp_global = datestr(now, 'yyyymmdd_HHMMSS');
detailsDir = fullfile(resultsDir, sprintf('AHD_Frontend_NULL_Ablation_details_%s', timestamp_global));
if ~exist(detailsDir, 'dir'), mkdir(detailsDir); end

masterCsvName = fullfile(resultsDir, sprintf('Master_AHD_Frontend_NULL_Ablation_%s.csv', timestamp_global));

%% 2. Dataset information
% The order is consistent with loaddata_small.m and your experimental setting table.
dataset_list = 1:12;
dataset_names = {'UMIST', 'VS(vehicle)', 'COIL20', 'SPF(steel plate)', ...
                 'IS(image segmentation)', 'FCT(forest)', 'MNIST', ...
                 'OptDigits', 'LS(Landsat)', 'ISOLET', 'USPS', 'PenDigits'};

%% 3. Repeated runs and ablation variants
num_runs = 20;  % Recommended for paper tables. Change to 10/30 if needed.

% Columns: [use_HO, use_MS, use_AW]
variant_names = {'w/o HO', 'w/o MS', 'w/o AW', 'NULL'};
variant_tags  = {'wo_HO',  'wo_MS',  'wo_AW',  'NULL'};
flags = [
    true, false,  false;   % w/o high-order: only first-order graphs
    false,  true, false;   % w/o multi-anchor: only one anchor scale
    false,  false,  true;  % w/o adaptive weighting: uniform alpha
    false,  false,  false    % full model
];
num_variants = numel(variant_names);

%% 4. Master CSV header
fid_master = fopen(masterCsvName, 'w', 'n', 'GBK');
if fid_master == -1
    error('无法创建总览 CSV 文件：%s', masterCsvName);
end

% 已移除 AnchorsRate, Order, NumSampling, K, Delta
header_master = {'DatasetID','DatasetName','Variant','use_HO','use_MS','use_AW', ...
                 'NumOriginalGraphs','NumBaseClusterings','NumRepeatedBaseClusterings','NumRuns', ...
                 'ACC_mean±std','ACC_max', ...
                 'NMI_mean±std','NMI_max', ...
                 'ARI_mean±std','ARI_max', ...
                 'Runtime_mean±std','Runtime_max'};
fprintf(fid_master, '%s\n', strjoin(header_master, ','));

fprintf('====================================================================\n');
fprintf('>>> 启动 AHD-EC 前端消融实验：Base9 公平版本。\n');
fprintf('>>> 关键约束：Full、w/o HO、w/o MS、w/o AW 均生成 9 个 base clusterings。\n');
fprintf('>>> 详细记录文件夹: %s\n', detailsDir);
fprintf('>>> 总览结果 CSV   : %s\n', masterCsvName);
fprintf('====================================================================\n');

%% 5. Main loop over 12 datasets
for data_idx = dataset_list

    clear X Y all_runs detail_rows;

     % ---------------------------------------------------------------------
    % 5.1 锁定主实验超参数
    % 这些参数与主实验 / 超参数搜索后的设置保持一致，避免消融实验重新调参。
    % ---------------------------------------------------------------------
    opt_order = 4;
    opt_ns = 4;
    delta = 5;
    target_num_base = opt_order * opt_ns;  % Full = 3 阶 × 3 个锚点尺度 = 9 个基聚类

    switch data_idx
        case 1,  anchors_rate = 22; k_val = 4;
        case 2,  anchors_rate = 74; k_val = 4;
        case 3,  anchors_rate = 60; k_val = 4;
        case 4,  anchors_rate = 18; k_val = 6;
        case 5,  anchors_rate = 94; k_val = 4;
        case 6,  anchors_rate = 36; k_val = 6;
        case 7,  anchors_rate = 84; k_val = 4;
        case 8,  anchors_rate = 26; k_val = 6;
        case 9,  anchors_rate = 40; k_val = 5;
        case 10, anchors_rate = 56; k_val = 18;
        case 11, anchors_rate = 90; k_val = 6;
        case 12, anchors_rate = 70; k_val = 6;
        otherwise, anchors_rate = 20; k_val = 3;
    end

    fprintf('\n--------------------------------------------------------------------\n');
    fprintf('>>> Dataset %02d (%s) | AR=%d, K=%d, Order=%d, NS=%d, Delta=%d\n', ...
        data_idx, dataset_names{data_idx}, anchors_rate, k_val, opt_order, opt_ns, delta);

    % ---------------------------------------------------------------------
    % 5.2 加载数据并构造完整锚点列表
    % ---------------------------------------------------------------------
    [X, Y] = loaddata_small(data_idx);
    X = double(X);
    Y = double(Y);

    % 与主实验保持一致的行归一化；加入 0 行保护，避免除 0。
    rowMax = max(X, [], 2);
    rowMax(rowMax == 0) = 1;
    X = X ./ rowMax;

    [num, dim] = size(X);
    c = length(unique(Y));

    anchors_full = zeros(1, opt_ns);
    for t = 1:opt_ns
        anchors_full(t) = (anchors_rate + (t - 1) * delta) * c;
    end

    % 详细记录表：每一行对应一个数据集、一个变体、一次 seed 运行。（共19列）
    detail_rows = cell(num_variants * num_runs, 14);
    all_runs = struct([]);
    row_ptr = 0;

     % ---------------------------------------------------------------------
    % 5.3 遍历消融版本与随机种子
    % ---------------------------------------------------------------------
    for v = 1:num_variants
        cur_use_HO = flags(v, 1);
        cur_use_MS = flags(v, 2);
        cur_use_AW = flags(v, 3);

        metrics_mat = nan(num_runs, 4); % ACC, NMI, ARI, Runtime
        info_cell = cell(num_runs, 1);

        fprintf('    >>> Variant %-7s | HO=%d, MS=%d, AW=%d | Target #Base=%d\n', ...
            variant_names{v}, cur_use_HO, cur_use_MS, cur_use_AW, target_num_base);

        for run_idx = 1:num_runs
            % 重要：四个消融版本使用相同 seed 序列，降低随机初始化差异的影响。
            seed = data_idx * 1000 + run_idx * 10;
            rng(seed, 'twister');

            try
                [F, obj, runtime, alphaA, runInfo] = AHD_EC_fix(k_val, opt_order, X, anchors_full, c, ...
                    cur_use_HO, cur_use_MS, cur_use_AW, target_num_base);

                [ACC, NMI, ~, ~, ~, ~, ~, ARI] = ClusteringMeasure4(Y, F);
                metrics_mat(run_idx, :) = [ACC, NMI, ARI, runtime];

            catch ME
                F = [];
                obj = [];
                alphaA = [];
                runtime = NaN;
                ACC = NaN; NMI = NaN; ARI = NaN;
                runInfo = struct('NumOriginalGraphs', NaN, 'NumBaseClusterings', NaN, ...
                                 'NumRepeatedBaseClusterings', NaN, 'ClusterNumbers', [], ...
                                 'RepresentationSourceIndex', []);
                fprintf('        [ERROR] Seed %d failed: %s\n', seed, ME.message);
            end

            info_cell{run_idx} = runInfo;

            row_ptr = row_ptr + 1;
            
            % 已移除 anchors_rate, opt_order, opt_ns, k_val, delta
            detail_rows(row_ptr, :) = {data_idx, dataset_names{data_idx}, variant_tags{v}, seed, ...
                cur_use_HO, cur_use_MS, cur_use_AW, ...
                runInfo.NumOriginalGraphs, runInfo.NumBaseClusterings, runInfo.NumRepeatedBaseClusterings, ...
                ACC, NMI, ARI, runtime};

            all_runs(row_ptr).DatasetID = data_idx;
            all_runs(row_ptr).DatasetName = dataset_names{data_idx};
            all_runs(row_ptr).Variant = variant_tags{v};
            all_runs(row_ptr).Seed = seed;
            all_runs(row_ptr).use_HO = cur_use_HO;
            all_runs(row_ptr).use_MS = cur_use_MS;
            all_runs(row_ptr).use_AW = cur_use_AW;
            all_runs(row_ptr).Anchors = anchors_full;
            % 已移除 all_runs 结构体中的 AnchorsRate, Order, NumSampling, K, Delta
            all_runs(row_ptr).NumOriginalGraphs = runInfo.NumOriginalGraphs;
            all_runs(row_ptr).NumBaseClusterings = runInfo.NumBaseClusterings;
            all_runs(row_ptr).NumRepeatedBaseClusterings = runInfo.NumRepeatedBaseClusterings;
            all_runs(row_ptr).ACC = ACC;
            all_runs(row_ptr).NMI = NMI;
            all_runs(row_ptr).ARI = ARI;
            all_runs(row_ptr).Runtime = runtime;
            all_runs(row_ptr).F_Labels = F;
            all_runs(row_ptr).Obj_History = obj;
            all_runs(row_ptr).alphaA_History = alphaA;
        end

        % -----------------------------------------------------------------
        % 5.4 写入当前变体的汇总结果
        % -----------------------------------------------------------------
        valid_rows = ~isnan(metrics_mat(:, 1));
        valid_metrics = metrics_mat(valid_rows, :);
        valid_runs = size(valid_metrics, 1);

        if valid_runs > 0
            avg_metrics = mean(valid_metrics, 1);
            std_metrics = std(valid_metrics, 0, 1);
            max_metrics = max(valid_metrics, [], 1);

            % 由于同一变体的 base 数量信息一致，取第一个有效 runInfo 即可。
            firstInfo = info_cell{find(valid_rows, 1, 'first')};
        else
            avg_metrics = nan(1, 4);
            std_metrics = nan(1, 4);
            max_metrics = nan(1, 4);
            firstInfo = struct('NumOriginalGraphs', NaN, 'NumBaseClusterings', NaN, 'NumRepeatedBaseClusterings', NaN);
        end

        % 调整了 fprintf 占位符，删去了 5 个 %d
        fprintf(fid_master, '%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.4f±%.4f,%.4f,%.4f±%.4f,%.4f,%.4f±%.4f,%.4f,%.4f±%.4f,%.4f\n', ...
            data_idx, dataset_names{data_idx}, variant_tags{v}, cur_use_HO, cur_use_MS, cur_use_AW, ...
            firstInfo.NumOriginalGraphs, firstInfo.NumBaseClusterings, firstInfo.NumRepeatedBaseClusterings, valid_runs, ...
            avg_metrics(1), std_metrics(1), max_metrics(1), ... % ACC
            avg_metrics(2), std_metrics(2), max_metrics(2), ... % NMI
            avg_metrics(3), std_metrics(3), max_metrics(3), ... % ARI
            avg_metrics(4), std_metrics(4), max_metrics(4));   % Runtime

        fprintf('        ACC %.4f±%.4f | NMI %.4f±%.4f | ARI %.4f±%.4f | Time %.4fs | OriginalGraphs=%d, FinalBase=%d\n', ...
            avg_metrics(1), std_metrics(1), avg_metrics(2), std_metrics(2), ...
            avg_metrics(3), std_metrics(3), avg_metrics(4), ...
            firstInfo.NumOriginalGraphs, firstInfo.NumBaseClusterings);
    end

    % ---------------------------------------------------------------------
    % 5.5 保存当前数据集的详细运行记录
    % ---------------------------------------------------------------------
    % 已移除 AnchorsRate, Order, NumSampling, K, Delta
    detailVarNames = {'DatasetID','DatasetName','Variant','Seed','use_HO','use_MS','use_AW', ...
                      'NumOriginalGraphs','NumBaseClusterings','NumRepeatedBaseClusterings', ...
                      'ACC','NMI','ARI','Runtime'};
    detailTable = cell2table(detail_rows, 'VariableNames', detailVarNames);

    detailCsvName = fullfile(detailsDir, sprintf('AHD_Frontend_Ablation_Base9_Data%02d_%s.csv', data_idx, timestamp_global));
    writetable(detailTable, detailCsvName);

    detailMatName = fullfile(detailsDir, sprintf('AHD_Frontend_Ablation_Base9_Data%02d_%s.mat', data_idx, timestamp_global));
    % save 参数中移除了 'opt_order', 'opt_ns', 'anchors_rate', 'k_val', 'delta'
    save(detailMatName, 'all_runs', 'detailTable', 'anchors_full', ...
        'target_num_base', 'X', 'Y', 'c', '-v7.3');

    fprintf('    已保存当前数据集详细记录: %s\n', detailCsvName);
end

fclose(fid_master);

fprintf('\n====================================================================\n');
fprintf('[SUCCESS] Base9 公平消融实验全部完成！\n');
fprintf('  总览 CSV    : %s\n', masterCsvName);
fprintf('  详细文件夹  : %s\n', detailsDir);
fprintf('====================================================================\n');