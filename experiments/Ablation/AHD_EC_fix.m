function [F, obj, runtime, alphaA, info] = AHD_EC_fix(k, order, X, anchors, c, use_HO, use_MS, use_AW, target_num_base)
% =========================================================================
% AHD_EC_fix_Base9
% 用于前端消融实验的 AHD-EC 修正版。
%
% 与原始 AHD_EC.m 的关系：
%   1) 锚点选择、二部图构造、高阶图构造、Tcut 基聚类、ADCF 共识学习等核心流程保持一致；
%   2) 只额外加入三个消融开关：use_HO / use_MS / use_AW；
%   3) 额外加入 target_num_base，用于保证 w/o HO 和 w/o MS 也输出 9 个 base clusterings；
%   4) 当原始表示数不足 target_num_base 时，不引入被删除模块，只复用已有表示，
%      并通过不同 c_base 生成额外基聚类，从而控制 ensemble size。
%
% 输入：
%   k               : anchor graph 的近邻数
%   order           : Full 模型的高阶阶数，默认实验中为 3
%   X               : 数据矩阵，n × d
%   anchors         : 完整多锚点配置，例如 [(ar)c, (ar+delta)c, (ar+2delta)c]
%   c               : 真实簇数 / 目标共识聚类簇数
%   use_HO          : 是否使用 high-order graph
%   use_MS          : 是否使用 multi-anchor / multi-scale
%   use_AW          : 是否在 ADCF 中使用自适应权重
%   target_num_base : 目标 base clustering 数量；Full=order*num_sampling=9
%
% 输出：
%   F       : 最终聚类标签
%   obj     : ADCF 目标函数历史
%   runtime : 总运行时间
%   alphaA  : ADCF 权重历史
%   info    : 实验细节信息，用于日志记录
% =========================================================================

%% 0. 默认参数保护
if nargin < 6 || isempty(use_HO), use_HO = true; end
if nargin < 7 || isempty(use_MS), use_MS = true; end
if nargin < 8 || isempty(use_AW), use_AW = true; end
if nargin < 9 || isempty(target_num_base), target_num_base = order * length(anchors); end

tic;
[num, ~] = size(X);

%% 1. 多锚点 / 多尺度开关
% Full 或 w/o HO：保留全部锚点尺度；
% w/o MS：只使用第一个锚点尺度，确保没有 multi-anchor 信息泄漏。
if use_MS
    num_sampling = length(anchors);
else
    num_sampling = 1;
end

%% 2. 构造一阶 anchor bipartite graph
% 这一段与原始 AHD_EC.m 保持一致，只是 num_sampling 由 use_MS 控制。
B1_cell = cell(1, num_sampling);
for t = 1:num_sampling
    if anchors(t) >= num
        anchors(t) = 9 * c;              % 保底一：退回到 9*c
        if anchors(t) >= num
            anchors(t) = num - 2;        % 保底二：最多取 num-2，避免锚点数越界
        end
    end

    [~, ind, ~] = graphgen_anchor(X, anchors(t));
    centers = X(ind, :);

    % 使用 pdist2 的 Smallest 选项，只取 k+1 个最近锚点，避免构造完整距离矩阵。
    [D_knn_T, idx_knn_T] = pdist2(centers, X, 'squaredeuclidean', 'Smallest', k + 1);
    D_knn = D_knn_T';
    col_idx = idx_knn_T';

    % 向量化构造一阶稀疏二部图。
    di_k1 = D_knn(:, end);
    denominator = k * di_k1 - sum(D_knn(:, 1:k), 2) + eps;
    vals = (repmat(di_k1, 1, k + 1) - D_knn) ./ repmat(denominator, 1, k + 1);
    row_idx = repmat((1:num)', 1, k + 1);

    B1_cell{1, t} = sparse(row_idx(:), col_idx(:), vals(:), num, anchors(t));
end

%% 3. 高阶图开关
% Full 或 w/o MS：保留 order 阶高阶图；
% w/o HO：只保留一阶图，不执行 SVD 与高阶传播。
if use_HO
    actual_order = order;
else
    actual_order = 1;
end

B = cell(actual_order, num_sampling);
for t = 1:num_sampling
    B_raw = B1_cell{1, t};
    dx = sum(B_raw, 2);
    dz = sum(B_raw, 1)';
    Dx_inv_sqrt = spdiags(1 ./ sqrt(dx + eps), 0, size(B_raw, 1), size(B_raw, 1));
    Dz_inv_sqrt = spdiags(1 ./ sqrt(dz + eps), 0, size(B_raw, 2), size(B_raw, 2));
    B{1, t} = Dx_inv_sqrt * B_raw * Dz_inv_sqrt;

    % 只有 use_HO=true 时才生成高阶图；w/o HO 完全跳过该部分。
    if use_HO && actual_order > 1
        [U, sigma, Vt] = svd(full(B{1, t}), 'econ');
        for d = 2:actual_order
            temp = U * (sigma .^ (2 * d - 1) * Vt');
            temp(temp < eps) = 0;
            B{d, t} = temp ./ (sum(temp, 2) + eps); % 行归一化为转移概率形式
        end
    end
end

%% 4. 将已有图表示整理成候选表示池
% Full：actual_order * num_sampling = 9，无需补齐；
% w/o HO：1 * 3 = 3，只含三个尺度的一阶图，需要复用补齐到 9；
% w/o MS：3 * 1 = 3，只含单一锚点配置下的 1/2/3 阶图，需要复用补齐到 9；
% w/o AW：3 * 3 = 9，只关闭权重学习，无需补齐。
B_core = reshape(B, [], 1);
num_original_graphs = length(B_core);

num_base_clusterings = max(target_num_base, num_original_graphs);
num_repeated_base_clusterings = num_base_clusterings - num_original_graphs;

% RepresentationSourceIndex 记录第 i 个 base clustering 使用的是第几个原始图表示。
% 如果 i > num_original_graphs，则循环复用已有图表示，但使用新的 c_base(i)。
representation_source_index = zeros(num_base_clusterings, 1);
for i = 1:num_base_clusterings
    representation_source_index(i) = mod(i - 1, num_original_graphs) + 1;
end

%% 5. 多分辨率基聚类生成
% 为了公平比较，所有变体都使用相同长度的 c_base：c, c+1, ..., c+8。
% w/o HO / w/o MS 的额外基聚类来自"已有图表示 + 不同聚类簇数"，而不是重新引入被删除模块。
c_base = c:1:(c + num_base_clusterings - 1);
H = cell(num_base_clusterings, 1);

rep_times = 20;

for i = 1:num_base_clusterings
    src_idx = representation_source_index(i);
    B_current = B_core{src_idx};

    [labels, ~] = Tcut_for_bipartite_graph(B_current, c_base(i), 100, rep_times);
    H{i} = sparse(1:num, labels, 1, num, c_base(i));
end

% 释放中间图，降低内存压力。
clear B B_core B1_cell B_raw B_current;

%% 6. ADCF 共识融合
% 使用第一个基聚类初始化，这一点与原始 AHD_EC.m 保持一致。
F_init = H{1};
[F, obj, ~, alphaA] = ADCF_fix(H, F_init, use_AW);

runtime = toc;

%% 7. 输出日志信息
info = struct();
info.use_HO = use_HO;
info.use_MS = use_MS;
info.use_AW = use_AW;
info.ActualOrder = actual_order;
info.ActualNumSampling = num_sampling;
info.NumOriginalGraphs = num_original_graphs;
info.TargetNumBase = target_num_base;
info.NumBaseClusterings = num_base_clusterings;
info.NumRepeatedBaseClusterings = num_repeated_base_clusterings;
info.ClusterNumbers = c_base;
info.RepresentationSourceIndex = representation_source_index;
info.RepTimes = rep_times;
end
