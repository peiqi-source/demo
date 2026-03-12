function [labels, U_x] = Tcut_for_bipartite_graph(B, Nseg, MAXiter, REPlic)
% Tcut_for_bipartite_graph: 极速二部图谱聚类 (Transfer Cut)
% 该版本已彻底消灭所有 NxN 的矩阵操作与全量特征分解，复杂度严格为 O(N)

% 设置 K-means 默认参数
if nargin < 3, MAXiter = 100; end
if nargin < 4, REPlic = 1; end

[Nx, Ny] = size(B); % Nx: 样本数, Ny: 锚点数 (m)

% =========================================================================
% 第一步：极限降维计算转移矩阵 Wy (彻底消灭 NxN 的 Dx 矩阵)
% =========================================================================
dx = sum(B, 2);
dx(dx == 0) = 1e-10; % 防除零保护

% 隐式扩展直接相除：完全等价于 Wy = B' * Dx^-1 * B
% 产生一个极其轻量的 m x m 矩阵
Wy = B' * (B ./ dx); 

% =========================================================================
% 第二步：计算 Y 侧度矩阵并归一化 (彻底消灭 m x m 的 Dy 矩阵)
% =========================================================================
d = sum(Wy, 2);
d(d == 0) = eps;
d_inv_sqrt = 1 ./ sqrt(d);

% 向量外积广播：完全等价于 nWy = Dy^-0.5 * Wy * Dy^-0.5
nWy = Wy .* (d_inv_sqrt * d_inv_sqrt');
nWy = (nWy + nWy') / 2; % 强制对称，防止浮点误差导致复数特征值

% =========================================================================
% 第三步：极速特征值分解 (从 O(m^3) 降维到 O(m^2 * Nseg))
% =========================================================================
opts.disp = 0;   % 关闭控制台输出
opts.isreal = 1; % 确保是实数矩阵
opts.issym = 1;  % 确保是对称矩阵

try
    % 使用 Lanczos 算法只求最大的 Nseg 个代数特征值 ('la')
    [evec, eval] = eigs(nWy, Nseg, 'la', opts);
catch
    % 极小概率容错：如果极度稀疏导致 eigs 不收敛，退化回全量 eig 保底
    [evec, eval] = eig(full(nWy));
end

% 提取前 Nseg 个最大的特征向量
[~, idx] = sort(diag(eval), 'descend');
Ncut_evec = evec(:, idx(1:Nseg));

% =========================================================================
% 第四步：将特征向量从锚点侧 (Y) 转移回样本侧 (X) -- Transfer Cut 核心
% =========================================================================
% 1. 还原 Y 侧真实特征向量
U_y = d_inv_sqrt .* Ncut_evec; 

% 2. 转移回 X 侧 (等价于 U_x = Dx^-1 * B * U_y)
% 极速矩阵乘法，瞬间得出 N x Nseg 的特征表示
U_x = (B ./ dx) * U_y;

% =========================================================================
% 第五步：特征行归一化与离散化 (K-means)
% =========================================================================
% 谱聚类标准步骤：映射到单位超球面，消除度(degree)不平衡的影响
U_x = U_x ./ (sqrt(sum(U_x.^2, 2)) + eps);

% 屏蔽由于孤立点偶尔导致的 KMeans 警告
warning('off', 'stats:kmeans:FailedToConverge');
warning('off', 'stats:kmeans:MissingDataRemoved');

% 执行极速 K-means，EmptyAction=singleton 确保不会出现空簇崩溃
labels = kmeans(U_x, Nseg, ...
    'MaxIter', MAXiter, ...
    'Replicates', REPlic, ...
    'EmptyAction', 'singleton');

warning('on', 'stats:kmeans:MissingDataRemoved');
warning('on', 'stats:kmeans:FailedToConverge');

end