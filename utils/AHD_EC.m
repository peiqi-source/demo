function [labels_all, obj_all, runtime, alphaA_all] = AHD_EC(k, order, X, anchors, c, anchor_select)
%%
tic;
[~, num_sampling] = size(anchors);
[num, ~] = size(X);
for t = 1:num_sampling
    disp('---Anchor Selection---');
    if anchor_select == 1 % rand sample
        vec = randperm(num); % randperm(n)：1..n 的随机排列
        ind = vec(1:anchors(t));
        centers{t} = X(ind, :);
    elseif anchor_select == 2 % kmeans sample
        [~, centers{t}, ~, ~, ~] = litekmeans(X, anchors(t));  % 直接用 kmeans 的中心作为锚点（锚点不一定是原样本点）
    elseif anchor_select == 3 % 离中心最近的样本作为锚点
        [~, ~, ~, ~, dis] = litekmeans(X,anchors(t));
        [~,ind] = min(dis,[],1); 
        ind = sort(ind,'ascend');
        centers{t} = X(ind, :);
    elseif anchor_select == 4 % DAS
        [~,ind,~] = graphgen_anchor(X,anchors(t));
        centers{t} = X(ind, :);% anchors x dim
    end
    if num <= anchors(t) % 防止锚点数超过样本数
        anchors(t) = 9*c % 设置锚点上限
    end
    disp('---Generate 1-st order 2P Graphs---');
    D = L2_distance_1(X', centers{t}'); % n x m ，算每个样本到每个 anchor 距离
    [~, idx] = sort(D, 2);
    B1 = zeros(num,anchors(t));
    for ii = 1:num
        id = idx(ii,1:k+1);
        di = D(ii, id);
        B1(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps); % 构建一阶相似度矩阵
    end
    B1_cell{1,t} = B1; % 在实验：选取锚点的数量对聚类结果的影响
end

%%
disp('---Generate high order 2P Graphs---');
B = cell(order,num_sampling); % cell: 1 x num_sampling -> order x num_sampling
for t = 1:num_sampling
    B{1,t} =B1_cell{1,t}./max(max(B1_cell{1,t},[],2)); % 归一化
    [U,sigma,Vt] = svd(B{1,t}); % 奇异值分解
    for d = 2:order
        temp = U*sigma.^(2*d-1)*Vt'; % 构建高阶二部图
        temp(temp<eps)=0;
        temp = temp./max(max(temp,[],2));
        %temp = temp./sum(temp,2); % 严格归一化：使行和为1（更像概率分布）
        B{d,t} = temp; % 在
    end
end

%%
disp('---Generate base cluastering---');
% 多分辨率（multi-resolution）基聚类集成：故意让每次基础聚类的粒度不一样，后面再把这些结果融合成更稳的相似度/共识
c_base = c:1:(c+order*num_sampling-1);
B = reshape(B, [], 1);
for i = 1:order*num_sampling
    [labels,evec] = Tcut_for_bipartite_graph(B{i},c_base(i)); % 对每个二部图 B 做一次 Tcut（谱聚类+kmeans），得到 labels
    F_d = ind2vec(labels')'; % labels one-hot 成 F_d
    S{i} = F_d*F_d'; % 连续的基聚类结果
end

%%
disp('---Generate consensus clustering---')
<<<<<<< HEAD
F_init = Y_Initialize_SVD(S, c); % 初始化一个随机 one-hot 指示矩阵（离散聚类指示矩阵F）
[labels,obj,~,alphaA] = MDC(S,F_init); % MDC 会学习每个基础聚类的权重 alpha，并更新最终聚类 F
=======
labels_all = cell(1,2);
obj_all    = cell(1,2);
alphaA_all  = cell(1,2);
for run_id = 1:2
    if run_id == 1
        F_init = Y_Initialize_SVD(S, c); % 初始化 指示矩阵（离散聚类指示矩阵F）
    elseif run_id == 2
        F_init = Y_Initialize_rand(S, c); % 初始化一个随机 one-hot 指示矩阵（离散聚类指示矩阵F）
    end
    [labels_i, obj_i, ~, alphaA_i] = MDC(S,F_init); % MDC 会学习每个基础聚类的权重 alpha，并更新最终聚类 F
    labels_all{run_id} = labels_i;
    obj_all{run_id}    = obj_i;
    alphaA_all{run_id}  = alphaA_i;
end

>>>>>>> b2
runtime = toc;
