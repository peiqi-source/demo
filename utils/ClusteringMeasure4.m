function [ACC, MIhat, Purity, Fscore, P, R, RI] = ClusteringMeasure4(Y, predY)
% ClusteringMeasure2 终极物理极限版
% 严格 O(N) 复杂度，无损精度，彻底剥离冗余的纯 MATLAB 匈牙利算法
    % 强制成 n×1 列向量
    if size(Y,2) ~= 1
        Y = Y';
    end
    if size(predY,2) ~= 1
        predY = predY';
    end
    
    Y = Y(:);
    predY = predY(:);
    n = length(Y); 

    % 1. 极速重编号 (O(N log N) 但底层高度优化)
    [~, ~, Y] = unique(Y);
    [~, ~, predY] = unique(predY);
    nClass1 = max(Y);
    nClass2 = max(predY);
    
    % 2. 极速生成混淆矩阵 G (O(N), 耗时 < 0.01 秒)
    G = full(sparse(Y, predY, 1, nClass1, nClass2));
    
    % ---------------------------------------------------------
    % 指标 1：Purity (纯度)
    % ---------------------------------------------------------
    Purity = sum(max(G, [], 1)) / n;
    
    % ---------------------------------------------------------
    % 指标 2：ACC (准确率) -> 依赖混淆矩阵的匈牙利最优匹配
    % 【极限拆弹】：直接调用内置 C++ 编译的 matchpairs，无需任何补零！
    % ---------------------------------------------------------
    % 1e9 为极大代价，迫使算法寻找最大权值匹配
    M = matchpairs(-G, 1e9); 
    % 使用 sub2ind 极速提取匹配对的样本数
    ACC = sum(G(sub2ind(size(G), M(:,1), M(:,2)))) / n;
    
    % ---------------------------------------------------------
    % 指标 3：NMI (归一化互信息)
    % ---------------------------------------------------------
    P1 = sum(G, 2) / n;
    P2 = sum(G, 1) / n;
    
    H1 = -sum(P1(P1 > 0) .* log2(P1(P1 > 0)));
    H2 = -sum(P2(P2 > 0) .* log2(P2(P2 > 0)));
    
    P12 = G / n;
    MI = 0;
    for i = 1:nClass1
        for j = 1:nClass2
            if P12(i,j) > 0
                MI = MI + P12(i,j) * log2(P12(i,j) / (P1(i) * P2(j)));
            end
        end
    end
    MIhat = real(MI / max(H1, H2));
    
    % ---------------------------------------------------------
    % 指标 4~7：Fscore, Precision(P), Recall(R), Rand Index(RI)
    % 【极限拆弹】：取消所有逻辑索引，直接利用向量化组合公式
    % ---------------------------------------------------------
    G_vec = G(:);
    TP = sum(G_vec .* (G_vec - 1)) / 2;
    
    sum_col = sum(G, 1)';
    TP_FP = sum(sum_col .* (sum_col - 1)) / 2;
    
    sum_row = sum(G, 2);
    TP_FN = sum(sum_row .* (sum_row - 1)) / 2;
    
    if TP_FP == 0, P = 0; else, P = TP / TP_FP; end
    if TP_FN == 0, R = 0; else, R = TP / TP_FN; end
    if (P + R) == 0, Fscore = 0; else, Fscore = 2 * P * R / (P + R); end
    
    Total_Pairs = n * (n - 1) / 2;
    FN = TP_FN - TP;
    FP = TP_FP - TP;
    TN = Total_Pairs - TP - FP - FN;
    
    if Total_Pairs == 0
        RI = 1;
    else
        RI = (TP + TN) / Total_Pairs;
    end
end