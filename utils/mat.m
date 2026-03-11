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
    
    % 2. 极限向量化构建一阶稀疏二部图 (快如闪电，省去 for ii = 1:num)
    D_knn = D_sort(:, 1:k+1);
    di_k1 = D_knn(:, k+1);
    denominator = k * di_k1 - sum(D_knn(:, 1:k), 2) + eps;
    vals = (repmat(di_k1, 1, k+1) - D_knn) ./ repmat(denominator, 1, k+1);
    
    row_idx = repmat((1:num)', 1, k+1);
    col_idx = idx(:, 1:k+1);
    % 直接生成 N x m 的稀疏矩阵！
    B1_cell{1,t} = sparse(row_idx(:), col_idx(:), vals(:), num, anchors(t)); 
end

%%
B = cell(order, num_sampling); 
for t = 1:num_sampling
    B{1,t} = B1_cell{1,t} ./ max(max(B1_cell{1,t}, [], 2)); 
    [U, sigma, Vt] = svd(full(B{1,t}), 'econ'); % 保持 econ
    for d = 2:order
        temp = U * (sigma.^(2*d-1) * Vt'); 
        temp(temp < 1e-4) = 0; % 截断极小值
        B{d,t} = sparse(temp ./ max(max(temp, [], 2))); % 强制转为稀疏矩阵省内存
    end
end

%%
% 多分辨率基聚类
c_base = c:1:(c+order*num_sampling-1);
B = reshape(B, [], 1);
H = cell(length(B), 1); % 【关键】改名为 H，用来存指示矩阵

for i = 1:length(B)
    [labels, ~] = Tcut_for_bipartite_graph(B{i}, c_base(i)); 
    % 【关键】直接生成 N x c 的稀疏指示矩阵！彻底干掉 F_d * F_d'
    H{i} = sparse(1:num, labels, 1, num, c_base(i)); 
end

%%
% 使用纯随机初始化 (速度最快，最配 MDC)
F_init = sparse(1:num, randi(c, num, 1), 1, num, c);
% 传入 H 而不是 S！
[F, obj, ~, alphaA] = MDC(H, F_init); 

runtime = toc;
end




function [F, obj, runtime, alphaA] = MDC(H, F)
tic 
NITR = 100; 
[num, c] = size(F); 
V = length(H); 
mu = 1e-4; rho = 1.01; 
alphaA = []; alpha = ones(V, 1) / V; alphaA = [alphaA, alpha]; 

%% 预计算：极速图间相似度 (基于迹技巧)
B1 = zeros(V, V);
for u = 1:V
    for v = u:V
        % Tr(Hu Hu' Hv Hv') 等价于下式，仅需 O(N * c^2)
        val = sum((H{u}' * H{v}).^2, 'all'); 
        B1(u, v) = val; B1(v, u) = val;
    end
end

%% 提取所有视图的聚类标签 (避免循环内的 find 耗时)
h_v = zeros(num, V);
for v = 1:V
    [~, h_v(:, v)] = max(H{v}, [], 2);
end

%% 初始化相交矩阵 C (这是速度起飞的核心魔法)
C = cell(1, V);
for v = 1:V
    C{v} = H{v}' * F; % c_v x c 大小的极小矩阵
end

%% 初始化全局统计量
ff = sum(F, 1); 
fsf = zeros(1, c);
for v = 1:V
    fsf = fsf + alpha(v) * sum(C{v}.^2, 1); % 极其优雅的等价计算
end
sii = sum(alpha); 

% 初始 obj
sum_S2 = alpha' * B1 * alpha;
obj(1) = sum_S2 - 2 * sum(fsf ./ (ff + eps)) + c;
changed = zeros(NITR, 10); 

%% 开始极速优化迭代
for iter = 1:NITR
    for it = 1:10
        converged = true; 
        for i = 1:num 
            m = find(F(i, :)); 
            if isempty(m), continue; end
            
            % 【降维核弹】用极小的 C 矩阵直接拼出 ui，复杂度 O(V * c)
            ui = zeros(1, c);
            for v = 1:V
                ui = ui + alpha(v) * C{v}(h_v(i, v), :);
            end
            
            % 增量计算与决策
            del = (fsf + 2 * ui + sii) ./ (ff + 1 + eps) - (fsf ./ (ff + eps));
            f0_m = (fsf(m) - 2 * ui(m) + sii) / (ff(m) - 1 + eps);
            del(m) = fsf(m) / (ff(m) + eps) - f0_m;
            
            [~, p] = max(del); 
            
            if p ~= m 
                converged = false;
                changed(iter, it) = changed(iter, it) + 1;
                
                % O(1) 状态同步
                ff(m) = ff(m) - 1;
                ff(p) = ff(p) + 1;
                fsf(m) = fsf(m) - 2 * ui(m) + sii;
                fsf(p) = fsf(p) + 2 * ui(p) + sii;
                
                % 更新极小的相交矩阵 C
                for v = 1:V
                    q = h_v(i, v);
                    C{v}(q, m) = C{v}(q, m) - 1;
                    C{v}(q, p) = C{v}(q, p) + 1;
                end
                
                F(i, m) = 0; F(i, p) = 1;
            end
        end
        if converged, break; end
    end
    
    % ALM 权重更新 (无需拉平任何大矩阵)
    b = zeros(V, 1);
    for v = 1:V
        b(v) = 2 * sum(sum(C{v}.^2, 1) ./ (ff + eps)); 
    end
    [alpha, ~, ~] = ALM(B1, b, mu, rho);
    alphaA = [alphaA, alpha]; 
    
    % 同步新权重下的指标
    fsf = zeros(1, c);
    for v = 1:V
        fsf = fsf + alpha(v) * sum(C{v}.^2, 1);
    end
    sii = sum(alpha);
    sum_S2 = alpha' * B1 * alpha;
    obj(iter+1) = sum_S2 - 2 * sum(fsf ./ (ff + eps)) + c;
    
    if iter > 1 && abs((obj(iter+1) - obj(iter)) / obj(iter+1)) < 1e-10
        break;
    end
    if iter > 30 && sum(abs(obj(iter-9:iter-5) - obj(iter-4:iter))) < 1e-10
        break;
    end
end
[~, F] = max(F, [], 2);
runtime = toc;
end

