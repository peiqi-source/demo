function [F,obj,runtime,alphaA] = MDC_v1(W,F)
% X: 1 by V cell, X{1,v} n by d, where n is the number of objects and d is
% the dimensionality, V is the view number
% c: number of clusters
% 2^numAnchors: number of anchors
% numNearestAnchors: number of neighborhoods
% NITR: max iteratios to run
% F：the final clustering result（离散聚类指示矩阵）
% W：构建的原始每个视图的相似度矩阵

tic % 开始计时
%% 初始化参数设置
NITR=100; % 最大迭代次数
[num,c] = size(F); % num: 数据点个数 (n), c: 聚类簇数
V=size(W,2); % V: 视图的个数
mu=1e-4; % ALM 算法的初始惩罚参数
rho=1.01; % 步长
alphaA=[];
alpha=ones(V,1)/V; % 初始权重设为 1/V (等权平均)
alphaA=[alphaA,alpha]; % 记录第一轮的权值，用于画收敛图

%% init S（综合相似度矩阵，每个视图的加权和）
S=sparse(zeros(num)); % 使用稀疏矩阵节省内存
for v=1:V
    S=S+alpha(v,1)*W{1,v};
end

%% 计算初始目标函数（“理论上的相似度矩阵”（应该是一个块对角矩阵））
F1=F*(diag(diag(F'*F+eps*eye(c)).^-1))*F'; % 这就是理想图 S' = Y(Y^T Y)^{-1}Y^T。加上 eps 是为了防止除以 0

obj(1)=norm((S-F1),'fro')^2; % 真实图和理想图之间的误差
changed = zeros(NITR,10); % 追踪器：记录每次用坐标下降法有多少个点会跳槽
% tic

%% 开始优化迭代
for iter=1:NITR

    % fix alpha, update F
    fsf=sum(F.*(S*F)); % fsf 记录了每个簇的内部连接紧密度 (y_l^T S y_l)，公式（30）的分子
    ff=sum(F.*F); % ff 记录了每个簇包含的节点数量 (y_l^T y_l)，公式（30）的分母

    % 坐标下降法
    for it=1:10
        converged=true; % 标记符：数据点是否需要分去别的簇（true：呆在原簇，false：要去别的簇）
        for i=1:num % 遍历每一个数据点
            F_0=F;  F_0(i,:)=zeros(1,c); % 逐行更新，先选中第 i 行，将聚类标签清零
            fi=F(i,:); [~,m]=max(fi); % m：点 i 当前属于哪个簇
            si=S(:,i); sii=S(i,i); % 抽出第 i 个点与其他点的连接 si，以及自身的连接 sii
            del=zeros(1,c); % 用来记录 Δ(h)

            for k=1:c % 穷举，把点 i 尝试分给每一个簇 k
                fk=F(:,k); f0k=F_0(:,k);
                % 采用增量的方法更新
                if k==m % 情况1：如果正好是它原本所属的簇 m
                    f1=fsf(k)/ff(k); % 包含它时的得分：原本的得分直接拿来用
                    f0=(fsf(k)-2*f0k'*si-sii)/(ff(k)-1); % 剔除它后的得分 (利用数学展开极速计算，无需全图重算)
                else % 情况2：如果要试探其他簇 k
                    f1=(fsf(k)+2*fk'*si+sii)/(ff(k)+1); % 放入它后的得分 (只需加上相关的边权重)
                    f0=fsf(k)/ff(k); % 不含它时的得分：就是原本的得分
                end
                del(1,k)=f1-f0; % 计算得分差 Δ(h)
            end

            [~,p]=max(del); % p：能带来最大增量 Δ(h) 的那个最佳簇

            if p~=m % 如果最佳簇 p 和原本的簇 m 不一样，说明点 i 需要跳槽
                converged=false;
                changed(iter,it)=changed(iter,it)+1;
                F_p=F_0; F_p(i,p)=1; % 生成新的矩阵 F_p
                f0m=F_0(:,m); fpp=F_p(:,p);
                % 同步更新 ff 和 fsf，为下一次循环点 i+1 做准备
                % update ff
                ff(m)=f0m'*f0m;   ff(p)=fpp'*fpp;
                % update fsf
                fsf(m)=f0m'*S*f0m;   fsf(p)=fpp'*S*fpp;
                % update F
                F=F_p;
            end
        end
        if converged % 如果遍历了所有的点，大家都不跳槽了，提前结束内循环
            break;
        end
    end

    % fix F, update alpha
    F1=F*(diag(diag(F'*F+eps*eye(c)).^-1))*F'; % 拿到最新算出来的理想图 S'
    f1=F1(:); % f1 = vec(S')，将矩阵拉平成一维长向量 (也就是论文里等价公式的 y)
    M=[];
    for v=1:V
        M=[M,W{1,v}(:)]; % 把每一个视图也拉平，拼成一个矩阵 M (也就是论文里的 hat{S})
    end
    B1=M'*M; % hat{A} = hat{S}^T hat{S}
    b=2*M'*f1; % b = 2*hat{S}^T y
    % 调用外部 ALM 函数求解标准二次规划问题 (返回的最优权重就是 alpha)
    [alpha,obj1,obj2]=ALM(B1,b,mu,rho);
    alphaA=[alphaA,alpha]; % 记录新的权重

    % check objective value
    S=sparse(zeros(num));
    for v=1:V
        S=S+alpha(v,1)*W{1,v}; % 按照刚算出的新 alpha，重新计算综合相似度矩阵 S
    end
    obj(iter+1)=norm((S-F1),'fro')^2; % 计算最新的整体目标函数误差

    % 收敛判断 1：如果误差相对变化量小于 1e-10，说明收敛，跳出大循环
    if iter>2 && abs((obj(iter)-obj(iter-1))/obj(iter))<1e-10
        break;
    end

    % 收敛判断 2：如果循环次数很多了，且最近 10 次的波动之和极小，也认为收敛
    if iter>30 && sum(abs(obj(iter-9:iter-5)-obj(iter-5+1:iter)))<1e-10
        break;
    end
end
% 把离散矩阵 F (Y) 转化为最终的一维标签输出
[~,F]=max(F,[],2);
runtime=toc;
end

