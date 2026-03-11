function [labels,evec] = Tcut_for_bipartite_graph(B,Nseg,maxKmIters,cntReps)
% B - |X|-by-|Y|, cross-affinity-matrix
% 输入：
% B：num x anchor 的二部图相似度矩阵
% Nseg：要聚类的簇数
% maxKmIters：K-means的最大迭代次数

if nargin < 4
    cntReps = 3;
end
if nargin < 3
    maxKmIters = 100;
end

[Nx,Ny] = size(B);
if Ny < Nseg
%     error('Need more columns!');
    B(1,Nseg)=0;
    Ny = Nseg;
end

dx = sum(B,2);
dx(dx==0) = 1e-10; % Just to make 1./dx feasible.
Dx = sparse(1:Nx,1:Nx,1./dx); 
clear dx
Wy = B'*Dx*B;

%%% compute Ncut eigenvectors
% normalized affinity matrix
d = sum(Wy,2);
d(d==0) = eps;
D = sparse(1:Ny,1:Ny,1./sqrt(d)); 
clear d
nWy = D*Wy*D; 
clear Wy
nWy = (nWy+nWy')/2;

% computer eigenvectors
[evec,eval] = eig(full(nWy)); 
clear nWy   
[~,idx] = sort(diag(eval),'descend');
Ncut_evec = D*evec(:,idx(1:Nseg)); 
clear D

%%% compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
evec = Dx * B * Ncut_evec; 
clear B Dx Ncut_evec

% 强制固定特征向量的符号（消除 eig 的随机翻转）
for j = 1:size(evec, 2)
    % 找到该列绝对值最大的元素，强制让它的符号为正
    [~, max_idx] = max(abs(evec(:, j)));
    if evec(max_idx, j) < 0
        evec(:, j) = -evec(:, j);
    end
end

% normalize each row to unit norm
evec = bsxfun( @rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10 );
%rng(Nseg); % 用当前的簇数 Nseg 作为种子，既保证了确定性，又保留了不同簇数下的多样性！

labels = litekmeans(evec,Nseg,'MaxIter',maxKmIters,'Replicates',cntReps,'Distance','cosine');