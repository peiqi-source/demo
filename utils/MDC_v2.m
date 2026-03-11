function [F,obj,runtime,alphaA] = MDC(W,F)
tic % 开始计时
%% 初始化参数设置
NITR = 100; 
[num, c] = size(F); 
V = size(W, 2); 
mu = 1e-4; 
rho = 1.01; 
alphaA = [];
alpha = ones(V, 1) / V; 
alphaA = [alphaA, alpha]; 

%% 终极降维：提前预计算 ALM 需要的 B1 矩阵 (O(V^2 * nnz) 替代 O(N^2))
% 彻底干掉原本 M=[M, W(:)] 的内存核弹
B1 = zeros(V, V);
for u = 1:V
    for v = u:V
        % Tr(W_u^T W_v) 完美等价于拉平后的点乘求和
        val = sum(W{u} .* W{v}, 'all'); 
        B1(u, v) = val;
        B1(v, u) = val;
    end
end

%% init S
S = sparse(num, num); 
for v = 1:V
    S = S + alpha(v) * W{v};
end

%% 初始化向量化所需的中间变量 (大幅降低循环内计算量)
U = S * F;            % N x c 矩阵，U(i,k) 代表点 i 与簇 k 的连接总和
fsf = sum(F .* U, 1); % 1 x c 向量，每个簇的内部紧密度
ff = sum(F, 1);       % 1 x c 向量，每个簇的节点数
s_diag = diag(S);     % N x 1 向量，自身的度

% 极速计算初始 obj，利用迹的性质，完全避免构建 N x N 稠密投影矩阵 F1
obj(1) = sum(S.^2, 'all') - 2 * sum(fsf ./ (ff + eps)) + c;
changed = zeros(NITR, 10); 

%% 开始优化迭代
for iter = 1:NITR
    % ==========================================
    % 1. 极限向量化的坐标下降法 (更新 F)
    % ==========================================
    for it = 1:10
        converged = true; 
        for i = 1:num 
            m = find(F(i, :)); % 找到当前点所属的簇
            if isempty(m), continue; end
            
            sii = s_diag(i);
            ui = U(i, :); % 1 x c，点 i 与所有簇的连接和
            
            % 【降维打击】一次性并行计算点 i 跳槽到所有簇 k 的目标增量 Δ(h)
            del = (fsf + 2 * ui + sii) ./ (ff + 1 + eps) - (fsf ./ (ff + eps));
            
            % 单独修正 k == m（原地不动）的真实增量
            f0_m = (fsf(m) - 2 * ui(m) + sii) / (ff(m) - 1 + eps);
            del(m) = fsf(m) / (ff(m) + eps) - f0_m;
            
            [~, p] = max(del); % 极速找到最佳跳槽目标 p
            
            if p ~= m 
                converged = false;
                changed(iter, it) = changed(iter, it) + 1;
                
                % 状态同步更新 (极其廉价的常数级 O(1) 操作)
                ff(m) = ff(m) - 1;
                ff(p) = ff(p) + 1;
                
                fsf(m) = fsf(m) - 2 * ui(m) + sii;
                fsf(p) = fsf(p) + 2 * ui(p) + sii;
                
                F(i, m) = 0;
                F(i, p) = 1;
                
                % 动态维护 U 矩阵，完美避免下一次重复计算耗时的 S*F
                si = S(:, i); 
                U(:, m) = U(:, m) - si;
                U(:, p) = U(:, p) + si;
            end
        end
        if converged 
            break;
        end
    end
    
    % ==========================================
    % 2. 降维魔法 ALM 更新 (更新 alpha)
    % ==========================================
    b = zeros(V, 1);
    for v = 1:V
        % 利用迹变换极速计算原本 b = 2*M'*f1 的等价结果
        U_v = W{v} * F;
        b(v) = 2 * sum(sum(F .* U_v, 1) ./ (ff + eps));
    end
    
    [alpha, ~, ~] = ALM(B1, b, mu, rho);
    alphaA = [alphaA, alpha]; 
    
    % ==========================================
    % 3. 同步最新 S 矩阵与全局状态
    % ==========================================
    S = sparse(num, num);
    for v = 1:V
        S = S + alpha(v) * W{v};
    end
    
    U = S * F;
    fsf = sum(F .* U, 1);
    s_diag = diag(S);
    
    % 【再次降维】替代原始的 norm((S-F1),'fro')^2
    obj(iter+1) = sum(S.^2, 'all') - 2 * sum(fsf ./ (ff + eps)) + c;
    
    % ==========================================
    % 4. 收敛判定
    % ==========================================
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

