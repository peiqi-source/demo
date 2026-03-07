function F0 = Y_Initialize_SVD(S1, c)
[~, order] = size(S1);
S = zeros(size(S1{1}));
for o = 1:order
    S = S + S1{o};
end
% S = S1{1};
S = S / order;
S = (S + S')/2;                  % 强制对称
S(1:size(S,1)+1:end) = 0;        % 对角线置 0
S = max(S,0);                    % 截断成非负（数值误差可能出负）

% 归一化拉普拉斯 L = I - D^{-1/2} S D^{-1/2}
d = sum(S,2);
D_inv_sqrt = diag(1./sqrt(d + 1e-12));
L = eye(size(S,1)) - D_inv_sqrt * S * D_inv_sqrt;

% 取最小的c个特征向量（谱嵌入）
[U,~] = eigs(L, c, 'smallestreal');

% kmeans 得到初始标签 （来代替 one-hot ）
initLabel = kmeans(U, c, "Replicates", 10, "MaxIter", 200);
F0 = full(ind2vec(initLabel', c))';
end