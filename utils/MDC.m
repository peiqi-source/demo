function [f_label, obj, runtime, alphaA] = MDC(H, F_init)
tic;
NITR = 100; 
[num, c] = size(F_init); 
V = length(H); 
mu = 1e-4; rho = 1.01; 
alphaA = []; alpha = ones(V, 1) / V; alphaA = [alphaA, alpha]; 

%% 数据结构剥离：剥离稀疏矩阵操作，转换为纯 1D 数组 O(1) 极速更新
if size(F_init, 2) > 1
    [~, f_label] = max(F_init, [], 2);
    F_sparse = F_init;
else
    f_label = F_init;
    F_sparse = sparse(1:num, f_label, 1, num, c);
end
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

%% 初始化相交矩阵 C 
C = cell(1, V);
for v = 1:V
    C{v} = H{v}' * F_sparse; % c_v x c 大小的极小矩阵
end

%% 初始化全局统计量
ff = sum(F_sparse, 1); 
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
            m = f_label(i); % 直接从 1D 数组读，摒弃 find(F(i,:)) 的巨额开销
            if m == 0, continue; end
            
            % 用极小的 C 矩阵直接拼出 ui，复杂度 O(V * c)
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
                
                f_label(i) = p;
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
runtime = toc;
end