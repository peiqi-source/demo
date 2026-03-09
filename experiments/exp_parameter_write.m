%% exp_measure.m 性能指标提升实验 (带断点续传/边算边存)
clear;
clc;
close all;

%% setup paths
thisFile = mfilename("fullpath");
expDir = fileparts(thisFile);
rootDir = fileparts(expDir);
resultsDir = fullfile(rootDir, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end % 确保结果目录存在
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

% 初始化存储文件
timestamp = datestr(now, 'yyyymmdd_HHMMSS'); 
csvFileName = fullfile(resultsDir, sprintf('grid_search_results_%s.csv', timestamp));
isFirst = true; % 用于控制是否写入表头

%% 开始网格搜索
for ar = param_anchors_rate
    for ord = param_order
        for ns = param_num_sampling
            for k_val = param_k
                for seed = param_rng
                    
                    rng(seed);
                    anchors = [];
                    for t = 1:ns
                        anchors = [anchors, ar*c+(t-1)*delta*c];
                    end
                    
                    for a = 1:4
                        % 1. 运行核心实验
                        [F, obj, runtime, alphaA] = AHD_EC(k_val, ord, X, anchors, c, a);
                        
                        % 2. 处理每一次运行的两个结果 (run_id 1 和 2)
                        for i = 1:2
                            [ACC, MIhat, Purity, Fscore, ~, ~, ~] = ClusteringMeasure2(Y, F{i});
                            
                            % 3. 构建单行表格
                            result_data = table(ar, ord, ns, k_val, seed, a, i, ACC, MIhat, Purity, Fscore, runtime, ...
                                'VariableNames', {'AnchorsRate', 'Order', 'NumSampling', 'K', 'Seed', 'Method', 'F_MadeStyle', 'ACC', 'NMI', 'Purity', 'Fscore', 'Runtime'});
                            
                            % 4. 追加写入磁盘
                            if isFirst
                                writetable(result_data, csvFileName, 'WriteMode', 'overwrite');
                                isFirst = false;
                            else
                                writetable(result_data, csvFileName, 'WriteMode', 'append', 'WriteVariableNames', false);
                            end
                        end
                        fprintf("Saved: AR=%d, Ord=%d, NS=%d, K=%d, Seed=%d, Method=%d\n", ar, ord, ns, k_val, seed, a);
                    end
                end
            end
        end
    end
end

fprintf('\n实验结束，结果已实时保存至: %s\n', csvFileName);