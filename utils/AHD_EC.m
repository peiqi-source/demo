function [labels, obj, runtime, alphaA] = AHD_EC(k, order, anchors)
%%
disp('----------Anchor Selection----------');
disp('----------Generate 1-st order 2P Graphs----------');
[~, centers{t}, ~, ~, ~] = litekmeans(X, anchors(t)); % 对 X 做 kmeans 得 m_t 个 anchors
D = L2_distance_1(X', centers{t}'); % n x m ，算每个样本到每个 anchor 距离
[~, idx] = sort(D, 2);
B1 = zeros(num,anchors(t));
for ii = 1:num
    id = idx(ii,1:k+1);
    di = D(ii, id);
    B1(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps); % 构建一阶相似度矩阵
end
B1_cell{1,t} = B1; % 在实验：选取锚点的数量对聚类结果的影响

%%
disp('----------Generate high order 2P Graphs----------');
B = cell(order,num_sampling); % cell: 1 x num_sampling -> order x num_sampling
B{1,t} =B1_cell{1,t}./max(max(B1_cell{1,t},[],2)); % 归一化
[U,sigma,Vt] = svd(B{1,t}); % 奇异值分解F
for d = 2:order
    temp = U*sigma.^(2*d-1)*Vt'; % 构建高阶二部图
    temp(temp<eps)=0;
    temp = temp./max(max(temp,[],2));
    %temp = temp./sum(temp,2);
    B{d,t} = temp;
end

%%
disp('----------Generate base cluastering----------');
c_base = c:1:(c+order*num_sampling-1);
B = reshape(B, [], 1);
for i = 1:order*num_sampling
    [labels,F] = Tcut_for_bipartite_graph(B{i},c_base(i)); % 对每个二部图 B 做一次 Tcut（谱聚类+kmeans），得到 labels
    F_d = ind2vec(labels')'; % labels one-hot 成 F_d
    S{i} = F_d*F_d'; % 再得到相似度矩阵
end

F_init = Y_Initialize(num, c); % 初始化一个随机 one-hot 指示矩阵（离散聚类指示矩阵F）
[labels,obj,runtime,alphaA] = MDC(S,F_init); % MDC 会学习每个基础聚类的权重 alpha，并更新最终聚类 F

result = ClusteringMeasure1(Y, labels)
