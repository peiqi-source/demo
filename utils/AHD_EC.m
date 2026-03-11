function [F, obj, runtime, alphaA] = AHD_EC(k, order, X, anchors, c)
tic;
[num, ~] = size(X);
[~, num_sampling] = size(anchors);

B1_cell = cell(1, num_sampling);
for t = 1:num_sampling
    % 1. 快速锚点选择与距离计算
    [~, ind, ~] = graphgen_anchor(X, anchors(t)); 
    centers = X(ind, :);
    if num <= anchors(t), anchors(t) = 9*c; end
    
    D = L2_distance_1(X', centers'); 
    [D_sort, idx] = sort(D, 2);
    
    % 2. 极限向量化构建一阶稀疏二部图
    D_knn = D_sort(:, 1:k+1);
    di_k1 = D_knn(:, k+1);
    denominator = k * di_k1 - sum(D_knn(:, 1:k), 2) + eps;
    vals = (repmat(di_k1, 1, k+1) - D_knn) ./ repmat(denominator, 1, k+1);
    
    row_idx = repmat((1:num)', 1, k+1);
    col_idx = idx(:, 1:k+1);
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
        B{d,t} = temp ./ (sum(temp, 2) + eps);
    end
end

%% 多分辨率基聚类
c_base = c:1:(c+order*num_sampling-1);
B = reshape(B, [], 1);
H = cell(length(B), 1); % 指示矩阵为 H

for i = 1:length(B)
    [labels, ~] = Tcut_for_bipartite_graph(B{i}, c_base(i), 100, 10); 
    % 直接生成 N x c 的稀疏指示矩阵
    H{i} = sparse(1:num, labels, 1, num, c_base(i)); 
end

%%
% 使用纯随机初始化 
F_init = H{1};
% 传入 H 而不是 S
[F, obj, ~, alphaA] = MDC(H, F_init); 

runtime = toc;
end