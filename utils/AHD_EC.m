function [F, obj, runtime, alphaA] = AHD_EC(k, order, X, anchors, c)
tic;
[num, ~] = size(X);
[~, num_sampling] = size(anchors);

B1_cell = cell(1, num_sampling);
for t = 1:num_sampling
    if anchors(t) >= num
        anchors(t) = 9 * c; % 保底一：退回到 9*c
        if anchors(t) >= num 
            anchors(t) = num - 2; % 保底二：如果 9*c 依然比总样本大，最多取 num-2
        end
    end
    % 1. 快速锚点选择与距离计算
    if num > 9000
        ind = randperm(num, anchors(t)); 
        centers = X(ind, :);
    else
        [~, ind, ~] = graphgen_anchor(X, anchors(t)); 
        centers = X(ind, :);
    end
    % 使用底层优化的 pdist2 极速获取最小的 k+1 个平方距离，彻底消灭 O(N*m) 内存核弹！
    [D_knn_T, idx_knn_T] = pdist2(centers, X, 'squaredeuclidean', 'Smallest', k+1);
    D_knn = D_knn_T';       
    col_idx = idx_knn_T';   
    
    % 极限向量化构建一阶稀疏二部图
    di_k1 = D_knn(:, end);
    denominator = k * di_k1 - sum(D_knn(:, 1:k), 2) + eps;
    vals = (repmat(di_k1, 1, k+1) - D_knn) ./ repmat(denominator, 1, k+1);
    row_idx = repmat((1:num)', 1, k+1);
    
    % 直接生成 N x m 的稀疏矩阵！
    B1_cell{1,t} = sparse(row_idx(:), col_idx(:), vals(:), num, anchors(t)); 
end

%% 高阶图生成
B = cell(order, num_sampling); 
for t = 1:num_sampling
    B{1,t} = B1_cell{1,t} ./ max(max(B1_cell{1,t}, [], 2)); 
    [U, sigma, Vt] = svd(full(B{1,t}), 'econ'); % 保持 econ
    for d = 2:order
        temp = U * (sigma.^(2*d-1) * Vt'); 
        temp(temp < eps) = 0;
        B{d,t} = temp ./ (sum(temp, 2) + eps); % 行归一化转移概率
    end
end

%% 多分辨率基聚类
c_base = c:1:(c+order*num_sampling-1);
B = reshape(B, [], 1);
H = cell(length(B), 1); % 指示矩阵为 H

if num > 10000
    rep_times = 1; % 降维打击：大图极易聚类，不需要多次重启，省下 90% 时间！
else
    rep_times = 10; % 小图流形复杂：保持 10 次重启以榨干最高精度。
end

for i = 1:length(B)
    [labels, ~] = Tcut_for_bipartite_graph(B{i}, c_base(i), 100, rep_times); 
    % 直接生成 N x c 的稀疏指示矩阵
    H{i} = sparse(1:num, labels, 1, num, c_base(i)); 
    B{i} = []; % 用完即毁！防止几十个高阶图堆积撑爆内存！
end

%%
% 使用较优质的一阶二部图的基聚类结果代替纯随机初始化 
F_init = H{1};
% 传入 H 而不是 S
[F, obj, ~, alphaA] = MDC(H, F_init); 

runtime = toc;
end