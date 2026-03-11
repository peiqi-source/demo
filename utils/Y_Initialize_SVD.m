function F0 = Y_Initialize_SVD(S, c)
    [~, graph_num] = size(S);
    
    % =========================================================
    % 关键改进 1：自适应初始权重评估 (彻底消除等权噪声污染)
    % =========================================================
    sim_matrix = zeros(graph_num, graph_num);
    for i = 1:graph_num
        norm_i = norm(S{i}, 'fro');
        for j = i:graph_num
            if i == j
                sim_matrix(i, j) = 1;
            else
                norm_j = norm(S{j}, 'fro');
                % 利用矩阵内积极速计算图与图之间的结构相似度
                sim = sum(sum(S{i} .* S{j})) / (norm_i * norm_j + eps);
                sim_matrix(i, j) = sim;
                sim_matrix(j, i) = sim;
            end
        end
    end
    
    % 计算每个图的“合群度”，并归一化为初始权重 w
    w = sum(sim_matrix, 2);
    w = w / sum(w); 
    
    % 构建去噪后的加权初始共识图 S0
    S0 = zeros(size(S{1}));
    for i = 1:graph_num
        S0 = S0 + w(i) * S{i};  % 弃用粗暴的 S0 = S0 / graph_num!
    end
    % =========================================================

    S0 = (S0 + S0')/2;                  % 强制对称
    S0(1:size(S0,1)+1:end) = 0;         % 对角线置 0
    S0 = max(S0,0);                     % 截断成非负

    % =========================================================
    % 关键改进 2：目标对齐与符号锁定
    % =========================================================
    % 直接对 S0 取最大的 c 个特征向量 (对齐 MDC 的 Trace 最大化)
    S0_sparse = sparse(S0); 
    [U, ~] = eigs(S0_sparse, c, 'largestreal');

    % 特征向量符号对齐 (消除随机翻转)
    for j = 1:c
        [~, max_idx] = max(abs(U(:, j)));
        if U(max_idx, j) < 0
            U(:, j) = -U(:, j);
        end
    end

    % 行归一化 (投射到单位超球面)
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)) + 1e-12);

    % 锁定随机种子，进行 K-means 划分
    %rng(c); 
    initLabel = litekmeans(U, c, 'Replicates', 10, 'MaxIter', 200, 'Distance', 'cosine');
    
    % 转化为离散指示矩阵
    F0 = full(ind2vec(initLabel', c))';
end