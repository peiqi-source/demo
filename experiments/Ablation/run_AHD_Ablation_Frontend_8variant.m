% =========================================================================
% run_AHD_Ablation_Frontend12.m
% 前端消融实验脚本（组件叠加拓展版）：针对 AHD-EC 在 12 个数据集上进行测试。
%
% 拓展后的 8 种变体说明（false 代表关闭/不使用，true 代表打开/激活该组件）：
%   1) Base           : 基础模型，三个组件全部关闭 (flags: [false, false, false])
%   2) Base+HO        : 仅打开高阶图生成 (flags: [true, false, false])
%   3) Base+MS        : 仅打开多锚点/多尺度采样 (flags: [false, true, false])
%   4) Base+AW        : 仅打开自适应权重优化 (flags: [false, false, true])
%   5) Base+HO+MS     : 同时打开高阶图与多锚点采样 (flags: [true, true, false])
%   6) Base+HO+AW     : 同时打开高阶图与自适应权重 (flags: [true, false, true])
%   7) Base+MS+AW     : 同时打开多锚点采样与自适应权重 (flags: [false, true, true])
%   8) Full           : 三个组件全开的完整模型 (flags: [true, true, true])
%
% 输出说明：
%   - 详细记录文件夹：包含每个数据集独立运行的详细测算数据 (.csv & .mat)
%   - 总览 Master CSV：汇总 12 个数据集在 8 种组件叠加变体下的 ACC, NMI, ARI 和运行时间
% =========================================================================

clear; clc; close all;

%% 1. 环境配置与输出路径初始化
thisFile = mfilename('fullpath');
if isempty(thisFile)
    expDir = pwd;
else
    expDir = fileparts(thisFile);
end
rootDir = fileparts(expDir);

% 创建结果根目录
resultsDir = fullfile(pwd, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% 生成全局时间戳，防止多次实验结果相互覆盖
timestamp_global = datestr(now, 'yyyymmdd_HHMMSS');
detailsDir = fullfile(resultsDir, sprintf('AHD_Frontend_Addition_Ablation_details_%s', timestamp_global));
if ~exist(detailsDir, 'dir'), mkdir(detailsDir); end

% 总览 CSV 文件路径
masterCsvName = fullfile(resultsDir, sprintf('Master_AHD_Frontend_Addition_Ablation_%s.csv', timestamp_global));

%% 2. 评测数据集配置
% 保持与标准实验矩阵一致的 12 个基准数据集
dataset_list = 1:12;
dataset_names = {'UMIST', 'VS', 'COIL20', 'SPF', ...
                 'IS', 'FCT', 'MNIST', ...
                 'OptDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};

%% 3. 独立重复运行次数与 8 种组件叠加变体配置
num_runs = 20;  % 学术论文图表推荐的独立重复实验次数

% 完整拓展至 2^3 = 8 种实验组合（按组件累加逻辑命名）
% 矩阵列对应关系: [use_HO, use_MS, use_AW] (注：true代表打开，false代表关闭)
variant_names = {'Base', 'Base+HO', 'Base+MS', 'Base+AW', ...
                 'Base+HO+MS', 'Base+HO+AW', 'Base+MS+AW', 'Full'};
             
variant_tags  = {'Base', 'Base_HO', 'Base_MS', 'Base_AW', ...
                 'Base_HO_MS', 'Base_HO_AW', 'Base_MS_AW', 'Full'};

flags = [
    false, false, false;   % 1) Base: 基础对照组 (0个true)
    true,  false, false;   % 2) Base+HO: 仅打开高阶图 (1个true)
    false, true,  false;   % 3) Base+MS: 仅打开多尺度 (1个true)
    false, false, true;    % 4) Base+AW: 仅打开自适应权重 (1个true)
    true,  true,  false;   % 5) Base+HO+MS: 打开高阶和多尺度 (2个true)
    true,  false, true;    % 6) Base+HO+AW: 打开高阶和自适应权重 (2个true)
    false, true,  true;    % 7) Base+MS+AW: 打开多尺度和自适应权重 (2个true)
    true,  true,  true     % 8) Full: 三个组件全部打开 (3个true)
];
num_variants = numel(variant_names);

%% 4. 构建总览 Master CSV 表头
fid_master = fopen(masterCsvName, 'w', 'n', 'GBK');
if fid_master == -1
    error('无法创建总览 CSV 文件：%s', masterCsvName);
end

header_master = {'DatasetID','DatasetName','Variant','use_HO','use_MS','use_AW', ...
                 'NumOriginalGraphs','NumBaseClusterings','NumRepeatedBaseClusterings','NumRuns', ...
                 'ACC_mean±std','ACC_max', ...
                 'NMI_mean±std','NMI_max', ...
                 'ARI_mean±std','ARI_max', ...
                 'Runtime_mean±std','Runtime_max'};
fprintf(fid_master, '%s\n', strjoin(header_master, ','));

fprintf('====================================================================\n');
fprintf('>>> 启动 AHD-EC 前端组件增量/叠加消融实验（共 8 种变体）。\n');
fprintf('>>> 关键约束：所有变体均在公平基准下生成指定数量的 base clusterings。\n');
fprintf('>>> 详细记录文件夹: %s\n', detailsDir);
fprintf('>>> 总览结果 CSV   : %s\n', masterCsvName);
fprintf('====================================================================\n');

%% 5. 遍历 12 个基准数据集的主循环
for data_idx = dataset_list

    clear X Y all_runs detail_rows;

    % ---------------------------------------------------------------------
    % 5.1 锁定主实验核心超参数（避免消融实验内引入二次调参偏差）
    % ---------------------------------------------------------------------
    opt_order = 4;
    opt_ns = 4;
    delta = 5;
    target_num_base = opt_order * opt_ns;  % 3阶 × 3个尺度 = 9个基聚类平衡基准

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
    % 5.2 数据载入与标准化预处理
    % ---------------------------------------------------------------------
    [X, Y] = loaddata_small(data_idx);
    X = double(X);
    Y = double(Y);

    % 行归一化及防除零异常保护
    rowMax = max(X, [], 2);
    rowMax(rowMax == 0) = 1;
    X = X ./ rowMax;

    [num, dim] = size(X);
    c = length(unique(Y));

    % 构造完整锚点采样序列
    anchors_full = zeros(1, opt_ns);
    for t = 1:opt_ns
        anchors_full(t) = (anchors_rate + (t - 1) * delta) * c;
    end

    % 初始化当前数据集的详细行记录矩阵 (8种变体 × 20次运行)
    detail_rows = cell(num_variants * num_runs, 14);
    all_runs = struct([]);
    row_ptr = 0;

    % ---------------------------------------------------------------------
    % 5.3 纵向遍历 8 种消融变体与横向随机种子独立运行
    % ---------------------------------------------------------------------
    for v = 1:num_variants
        cur_use_HO = flags(v, 1);
        cur_use_MS = flags(v, 2);
        cur_use_AW = flags(v, 3);

        metrics_mat = nan(num_runs, 4); % 缓存当前变体的 [ACC, NMI, ARI, Runtime]
        info_cell = cell(num_runs, 1);

        fprintf('    >>> Variant %-16s | HO=%d, MS=%d, AW=%d | Target #Base=%d\n', ...
            variant_names{v}, cur_use_HO, cur_use_MS, cur_use_AW, target_num_base);

        for run_idx = 1:num_runs
            % 严格控制随机种子序列，确保不同变体之间初始条件完全可比
            seed = data_idx * 1000 + run_idx * 10;
            rng(seed, 'twister');

            try
                % 调用核心算法接口（将配置好的当前开关 true/false 传入函数中）
                [F, obj, runtime, alphaA, runInfo] = AHD_EC_fix(k_val, opt_order, X, anchors_full, c, ...
                    cur_use_HO, cur_use_MS, cur_use_AW, target_num_base);

                % 测算聚类指标
                [ACC, NMI, ~, ~, ~, ~, ~, ARI] = ClusteringMeasure4(Y, F);
                metrics_mat(run_idx, :) = [ACC, NMI, ARI, runtime];

            catch ME
                % 异常捕获机制，确保个别种子偶发不收敛时不中断整体实验
                F = []; obj = []; alphaA = []; runtime = NaN;
                ACC = NaN; NMI = NaN; ARI = NaN;
                runInfo = struct('NumOriginalGraphs', NaN, 'NumBaseClusterings', NaN, ...
                                 'NumRepeatedBaseClusterings', NaN, 'ClusterNumbers', [], ...
                                 'RepresentationSourceIndex', []);
                fprintf('        [ERROR] Seed %d 运行失败: %s\n', seed, ME.message);
            end

            info_cell{run_idx} = runInfo;
            row_ptr = row_ptr + 1;
            
            % 组合成明细单行数据存入 Cell 数组中
            detail_rows(row_ptr, :) = {data_idx, dataset_names{data_idx}, variant_tags{v}, seed, ...
                cur_use_HO, cur_use_MS, cur_use_AW, ...
                runInfo.NumOriginalGraphs, runInfo.NumBaseClusterings, runInfo.NumRepeatedBaseClusterings, ...
                ACC, NMI, ARI, runtime};

            % 归档至大结构体以便后续导出复杂的 .mat 文件
            all_runs(row_ptr).DatasetID = data_idx;
            all_runs(row_ptr).DatasetName = dataset_names{data_idx};
            all_runs(row_ptr).Variant = variant_tags{v};
            all_runs(row_ptr).Seed = seed;
            all_runs(row_ptr).use_HO = cur_use_HO;
            all_runs(row_ptr).use_MS = cur_use_MS;
            all_runs(row_ptr).use_AW = cur_use_AW;
            all_runs(row_ptr).Anchors = anchors_full;
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
        % 5.4 统计并整合当前变体的多轮均值与标准差，写入全局 CSV
        % -----------------------------------------------------------------
        valid_rows = ~isnan(metrics_mat(:, 1));
        valid_metrics = metrics_mat(valid_rows, :);
        valid_runs = size(valid_metrics, 1);

        if valid_runs > 0
            avg_metrics = mean(valid_metrics, 1);
            std_metrics = std(valid_metrics, 0, 1);
            max_metrics = max(valid_metrics, [], 1);
            firstInfo = info_cell{find(valid_rows, 1, 'first')};
        else
            avg_metrics = nan(1, 4); std_metrics = nan(1, 4); max_metrics = nan(1, 4);
            firstInfo = struct('NumOriginalGraphs', NaN, 'NumBaseClusterings', NaN, 'NumRepeatedBaseClusterings', NaN);
        end

        % 输出整定好的统计行数据至 Master CSV 
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
    % 5.5 定向存储当前数据集的细粒度明细表格与二进制 Mat 数据
    % ---------------------------------------------------------------------
    detailVarNames = {'DatasetID','DatasetName','Variant','Seed','use_HO','use_MS','use_AW', ...
                      'NumOriginalGraphs','NumBaseClusterings','NumRepeatedBaseClusterings', ...
                      'ACC','NMI','ARI','Runtime'};
    detailTable = cell2table(detail_rows, 'VariableNames', detailVarNames);

    detailCsvName = fullfile(detailsDir, sprintf('AHD_Frontend_Addition_Ablation_Base9_Data%02d_%s.csv', data_idx, timestamp_global));
    writetable(detailTable, detailCsvName);

    detailMatName = fullfile(detailsDir, sprintf('AHD_Frontend_Addition_Ablation_Base9_Data%02d_%s.mat', data_idx, timestamp_global));
    save(detailMatName, 'all_runs', 'detailTable', 'anchors_full', ...
        'target_num_base', 'X', 'Y', 'c', '-v7.3');

    fprintf('    [SAVE] 当前数据集细粒度记录已导出: %s\n', detailCsvName);
end

fclose(fid_master);

fprintf('\n====================================================================\n');
fprintf('[SUCCESS] 组件叠加全矩阵消融实验全部顺利完成！\n');
fprintf('  主控总览 CSV 路径 : %s\n', masterCsvName);
fprintf('  明细数据文件夹路径 : %s\n', detailsDir);
fprintf('====================================================================\n');