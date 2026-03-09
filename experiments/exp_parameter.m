%% exp_parameter.m  带超参数网格搜索
clear;
clc;
close all;

%% setup paths
thisFile = mfilename("fullpath");
expDir = fileparts(thisFile);
rootDir = fileparts(expDir);
resultsDir = fullfile(rootDir, 'results');
addpath(genpath(rootDir));

%% load data
[X, Y] = loaddata(8);
X = X./max(X, [], 2);
c = length(unique(Y));

%% 参数定义
param_anchors_rate = [10 11];
param_order = 2:5;
param_num_sampling = 2:5;
param_k = [5 7 10 15];
param_rng = 0:2:10;
delta = 5;

% 使用结构体数组收集结果，方便后期转换为表格
results_struct = []; 
idx = 1;

%% 开始网格搜索
for ar = param_anchors_rate
    for ord = param_order
        for ns = param_num_sampling
            for k_val = param_k
                for seed = param_rng
                    
                    % 1. 设置随机种子
                    rng(seed);
                    
                    % 2. 生成锚点
                    anchors = [];
                    for t = 1:ns
                        anchors = [anchors, ar*c+(t-1)*delta*c];
                    end
                    
                    % 3. 遍历锚点选择方式 (a)
                    for a = 1:4
                        fprintf("Processing: AR=%d, Ord=%d, NS=%d, K=%d, Seed=%d, Method=%d\n", ...
                            ar, ord, ns, k_val, seed, a);
                            
                        [F, obj, runtime, alphaA] = AHD_EC(k_val, ord, X, anchors, c, a);
                        
                        % 4. 记录结果
                        for i = 1:2
                            [ACC, MIhat, Purity, Fscore, ~, ~, ~] = ClusteringMeasure2(Y, F{i});
                            
                            % 存入结构体
                            results_struct(idx).AnchorsRate = ar;
                            results_struct(idx).Order = ord;
                            results_struct(idx).NumSampling = ns;
                            results_struct(idx).K = k_val;
                            results_struct(idx).Seed = seed;
                            results_struct(idx).Method = a;
                            results_struct(idx).F_MadeStyle = i;
                            results_struct(idx).ACC = ACC;
                            results_struct(idx).NMI = MIhat;
                            results_struct(idx).Purity = Purity;
                            results_struct(idx).Fscore = Fscore;
                            results_struct(idx).Runtime = runtime;
                            
                            idx = idx + 1;
                        end
                    end
                end
            end
        end
    end
end

%% 汇总结果并保存
result_table = struct2table(results_struct);

fprintf('\n=== 实验结果汇总（显示前10行） ===\n');
disp(head(result_table, 10));

timestamp = datestr(now, 'yyyymmdd_HHMMSS'); 
csvFileName = sprintf('grid_search_results_%s.csv', timestamp);
savePath = fullfile(resultsDir, csvFileName);
writetable(result_table, savePath);
fprintf('完整实验结果已保存至:\n -> %s\n', savePath);