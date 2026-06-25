%% exp_ablation.m - 物理开关控制的消融实验
clear; clc; close all;

%% 数据加载与基础参数锁定 (锁定完全体的最优配置)
[X, Y] = loaddata(15);
X = X ./ max(X, [], 2);
c = length(unique(Y));

% 调参得出的 SOTA 全局最优配置
opt_k = 7;
opt_order = 2;   % 完全体的目标阶数
opt_ns = 3;      % 完全体的目标采样数
anchors_rate = 10;
delta = 5;

% 预先生成好完全体需要的 anchor 数组
anchors_full = zeros(1, opt_ns);
for t = 1:opt_ns
    anchors_full(t) = anchors_rate * c + (t - 1) * delta * c;
end

num_seeds = 10;

%% 2. 顶刊级变体开关矩阵 (Flag Matrix)
% 列含义：[use_HO, use_MS, use_AW]
variants_name = {'w/o HO', 'w/o MS', 'w/o AW', 'Full'};

% 极其清晰的布尔控制：0 代表物理关闭该模块，1 代表开启
flags = [
    false, true,  true;  % 变体 1: 物理关闭高阶 (无 SVD)
    true,  false, true;  % 变体 2: 物理关闭多采样 (只跑一次锚点)
    true,  true,  false; % 变体 3: 物理关闭自适应加权 (权重始终为 1/V)
    true,  true,  true   % 变体 4: 三键全开 (满血完全体)
];

num_variants = length(variants_name);
final_results = zeros(num_variants, 5); 

%% 3. 执行物理隔离的控制变量测试
fprintf('==================================================\n');
fprintf('>>>开始基于物理开关的消融实验 ...\n');
fprintf('==================================================\n');

for v = 1:num_variants
    cur_use_HO = flags(v, 1);
    cur_use_MS = flags(v, 2);
    cur_use_AW = flags(v, 3);
    
    fprintf('测试 [%-8s] | 开关状态: HO=%d, MS=%d, AW=%d\n', ...
            variants_name{v}, cur_use_HO, cur_use_MS, cur_use_AW);
    
    temp_metrics = zeros(num_seeds, 5);
    for seed = 1:num_seeds
        rng(seed);
        
        % 将完全体的参数和当前的开关数组一并传进去！
        [F, ~, runtime, ~] = AHD_EC_fix(opt_k, opt_order, X, anchors_full, c, ...
                                    cur_use_HO, cur_use_MS, cur_use_AW);
        
        [ACC, MIhat, Purity, Fscore, ~, ~, ~] = ClusteringMeasure4(Y, F);
        temp_metrics(seed, :) = [ACC, MIhat, Purity, Fscore, runtime];
    end
    
    final_results(v, :) = mean(temp_metrics, 1);
end

%% 4. 生成结果
varNames = {'ACC', 'NMI', 'Purity', 'Fscore', 'Runtime'};
result_table = array2table(final_results, 'VariableNames', varNames, 'RowNames', variants_name);

fprintf('\n=== 物理消融实验结果 ===\n');
disp(result_table);