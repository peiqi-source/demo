% =========================================================================
% run_generate_AHD_base_pool9_12datasets.m
% -------------------------------------------------------------------------
% 功能：
%   一次性为 12 个真实数据集生成并保存 AHD-EC 前端的 9 个基聚类池。
%   后续 Table 2/3 的解耦验证中，ADCF / CSPA / MCLA / PTA / PTGP 等后端方法
%   必须读取同一个保存好的基聚类池，从而保证“只改变后端，不改变输入”。
%
% 输出：
%   results/AHD_BasePool9_时间戳/
%       Data01_UMIST/
%           AHD_BasePool9_Data01_UMIST_Run001_Seed1002.mat
%           ...
%           AHD_BasePool9_Details_Data01_UMIST.csv
%       ...
%       Master_AHD_BasePool9_时间戳.csv
%
% 每个 .mat 文件包含：
%   H_pool      : 1 × 9 cell，ADCF 直接使用的 sparse indicator matrices
%   label_pool  : n × 9 标签矩阵，CSPA/MCLA/PTA/PTGP 等传统共识函数可直接使用
%   c_base      : 9 个基聚类对应的簇数，默认为 c:c+8
%   Y           : 真实标签，用于后续评价 ACC/NMI/ARI
%   meta        : 本次前端生成的参数和运行细节
%
% 说明：
%   该脚本只生成并保存 AHD-EC 的前端基聚类池，不运行 ADCF。
% =========================================================================

clear; clc; close all;

%% 1. 路径设置
saveDir = fullfile(pwd, 'AHD_BasePool9_MAT');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end

%% 2. 数据集与实验参数
% 数据集名称与论文 Experimental Settings 中 12 个数据集保持一致。
datasetNames = {'UMIST', 'VS', 'COIL20', 'SPF', ...
                'IS', 'FCT', 'MNIST', ...
                'OptDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
dataset_list = 1:12;

% 每个数据集生成多少个 AHD-EC 9-base pool。
% 论文中如果后端消融跑 20 个 seed，这里保持 num_pool_runs=20 即可。
% 如果你希望后续像 simple base generation 一样构造更大的候选池，可以改成 100。
num_pool_runs = 20;

% AHD-EC 主实验固定设置：3 个 anchor scales × 3 个 graph orders = 9 个基聚类。
order = 3;
ns = 3;
delta = 5;

% 是否额外保存 X。后端共识函数通常只需要 H_pool/label_pool/Y，不需要 X。
% 为了节省空间，默认 false；如果后续想完整复现实验，可以改成 true。
save_X = false;

%% 3. 主汇总表初始化

fprintf('====================================================================\n');
fprintf('>>> 开始生成 AHD-EC 前端 9 个基聚类池\n');
fprintf('>>> 保存目录: %s\n', saveDir);
fprintf('>>> 每个数据集生成 %d 个 pool，每个 pool 含 9 个 base clusterings\n', num_pool_runs);
fprintf('====================================================================\n');

%% 4. 遍历 12 个数据集
for data_idx = dataset_list
    clear X Y H_pool label_pool c_base meta;
    
    dataname = datasetNames{data_idx};

    % -------------------------------------------------------------
    % 4.1 与 run_AHD_HyperparametersTuning.m 保持一致的参数配置
    % -------------------------------------------------------------
    switch data_idx
        case 1,  ar = 22; k = 4;
        case 2,  ar = 74; k = 4;
        case 3,  ar = 60; k = 4;
        case 4,  ar = 18; k = 6;
        case 5,  ar = 94; k = 4;
        case 6,  ar = 36; k = 6;
        case 7,  ar = 84; k = 4;
        case 8,  ar = 26; k = 6;
        case 9,  ar = 40; k = 5;
        case 10, ar = 56; k = 18;
        case 11, ar = 90; k = 6;
        case 12, ar = 70; k = 6;
        otherwise, ar = 20; k = 3;
    end

    fprintf('\n--------------------------------------------------------------------\n');
    fprintf('>>> Dataset %02d/%02d: %s | AR=%d, K=%d, Order=%d, NS=%d\n', ...
        data_idx, length(dataset_list), dataname, ar, k, order, ns);

    % -------------------------------------------------------------
    % 4.2 加载并预处理数据
    % -------------------------------------------------------------
    [X, Y] = loaddata_small(data_idx);
    [num, dim] = size(X);
    c = length(unique(Y));

    % 与主实验保持同类归一化处理，并增加防除零保护。
    rowMax = max(X, [], 2);
    rowMax(rowMax == 0) = 1;
    X = X ./ rowMax;
    X(isnan(X) | isinf(X)) = 0;

    anchors = zeros(1, ns);
    for t = 1:ns
        anchors(t) = (ar + (t - 1) * delta) * c;
    end

    fprintf('    样本数 n=%d, 维度 d=%d, 类别数 c=%d, anchors=[%s]\n', ...
        num, dim, c, num2str(anchors));
    % -------------------------------------------------------------
    % 4.3 生成多个 AHD-EC 9-base pool
    % -------------------------------------------------------------
    for run_idx = 1:num_pool_runs
        seed = data_idx * 1000 + run_idx * 10;   % 与 AHD 主实验 seed 规则保持一致
        rng(seed, 'twister');

        fprintf('    Run %03d/%03d | Seed=%d ... ', run_idx, num_pool_runs, seed);

        try
            %% 1. 构造多锚点一阶二部图 B1_cell
            % 注意：该部分与原始 AHD_EC.m 保持一致。
            B1_cell = cell(1, ns);
            actual_anchors = anchors;
            anchor_index_cell = cell(1, ns);  % 记录每个尺度实际选中的锚点索引，便于复现检查
            
            for t = 1:ns
                if actual_anchors(t) >= num
                    actual_anchors(t) = 9 * c;          % 保底一：退回到 9*c
                    if actual_anchors(t) >= num
                        actual_anchors(t) = num - 2;    % 保底二：最多取 num-2，避免锚点数越界
                    end
                end
                [~, ind, ~] = graphgen_anchor(X, actual_anchors(t));
                centers = X(ind, :);
                anchor_index_cell{t} = ind;
            
                % 只取每个样本最近的 k+1 个锚点，避免完整 n × m 距离矩阵造成内存压力。
                [D_knn_T, idx_knn_T] = pdist2(centers, X, 'squaredeuclidean', 'Smallest', k + 1);
                D_knn = D_knn_T';
                col_idx = idx_knn_T';
            
                % 向量化构造一阶稀疏二部图。
                di_k1 = D_knn(:, end);
                denominator = k * di_k1 - sum(D_knn(:, 1:k), 2) + eps;
                vals = (repmat(di_k1, 1, k + 1) - D_knn) ./ repmat(denominator, 1, k + 1);
                row_idx = repmat((1:num)', 1, k + 1);
            
                B1_cell{1, t} = sparse(row_idx(:), col_idx(:), vals(:), num, actual_anchors(t));
            end
            
            %% 2. 构造多阶 anchor graph 表示 B
            % 当前主实验 order=3, ns=3，因此最终产生 3 × 3 = 9 个图表示。
            B = cell(order, ns);
            for t = 1:ns
                B_raw = B1_cell{1, t};
                dx = sum(B_raw, 2);
                dz = sum(B_raw, 1)';
                Dx_inv_sqrt = spdiags(1 ./ sqrt(dx + eps), 0, size(B_raw, 1), size(B_raw, 1));
                Dz_inv_sqrt = spdiags(1 ./ sqrt(dz + eps), 0, size(B_raw, 2), size(B_raw, 2));
            
                % 一阶归一化二部图。
                B{1, t} = Dx_inv_sqrt * B_raw * Dz_inv_sqrt;
            
                % 高阶图生成：与原始 AHD_EC.m 一致，基于 SVD 构造更高阶传播表示。
                [U, sigma, Vt] = svd(full(B{1, t}), 'econ');
                for d_order = 2:order
                    temp = U * (sigma .^ (2 * d_order - 1) * Vt');
                    temp(temp < eps) = 0;
                    B{d_order, t} = temp ./ (sum(temp, 2) + eps);  % 行归一化为转移概率形式
                end
            end
            
            %% 3. 对每个图表示执行 Tcut，生成 9 个基聚类
            % 依据：Full AHD-EC 的前端多样性来源为
            %       3 个 anchor scales × 3 个 graph orders = 9 个结构表示。
            %       每个结构表示配合一个递增的过聚类簇数 c_i，形成最终的 9 个基聚类。
            c_base = c:1:(c + order * ns - 1);
            B = reshape(B, [], 1);
            num_base = length(B);
            
            H_pool = cell(1, num_base);
            label_pool = zeros(num, num_base);
            
            % 与原始 AHD_EC.m 一致的 Tcut 重启设置。
            rep_times = 10;
            for i = 1:num_base
                [labels, ~] = Tcut_for_bipartite_graph(B{i}, c_base(i), 100, rep_times);
                label_pool(:, i) = labels(:);
                H_pool{i} = sparse(1:num, labels(:), 1, num, c_base(i));
            
                % 用完即清，减少多阶图堆积导致的内存压力。
                B{i} = [];
            end
            
            %% 4. 保存运行细节
            meta = struct();
            meta.k = k;
            meta.anchors_input = anchors;
            meta.anchors_actual = actual_anchors;
            meta.anchor_index_cell = anchor_index_cell;
            meta.c = c;
            meta.c_base = c_base;
            meta.num_base_clusterings = num_base;
            meta.rep_times = rep_times;
            meta.dataset_id = data_idx;
            meta.dataset_name = dataname;
            meta.seed = seed;
            meta.n = num;
            meta.d = dim;
            meta.ar_val = ar;
            meta.delta = delta;
            meta.order = order;
            meta.num_sampling = ns;

            matFileName = fullfile(saveDir, sprintf('AHD_BasePool9_%s.mat', dataname));

            if save_X
                save(matFileName,'dataname', 'X', 'Y', 'c', ...
                    'H_pool', 'label_pool', 'c_base', 'anchors', 'meta', '-v7.3');
            else
                save(matFileName,'dataname', 'Y', 'c', ...
                    'H_pool', 'label_pool', 'c_base', 'anchors', 'meta', '-v7.3');
            end
            fprintf('完成 \n');

        catch ME
            % 出错时不中断整个 12 数据集流程，记录错误并继续下一个 run。
            fprintf('失败: %s\n', ME.message);
        end
    end
end

fprintf('\n====================================================================\n');
fprintf('[SUCCESS] AHD-EC 9-base pool 生成完毕！\n');
fprintf('  - pool 文件夹: %s\n', saveDir);
fprintf('====================================================================\n');
