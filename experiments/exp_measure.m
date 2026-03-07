%% exp_measure.m 性能指标提升实验
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
[X, Y] = loaddata(9);

X = X./max(X, [], 2);
c = length(unique(Y));

%% set parameter
rng(1);
k = 5;
order = 3;
num_sampling = 3;
anchors_rate = 10;
delta = 5;

anchors = [];
for t = 1:num_sampling
    anchors = [anchors, anchors_rate*c+(t-1)*delta*c];
end

%% run experiment and record result
total_exp = num_sampling * 4;
result_matrix = zeros(total_exp, 7); 
row_idx = 1;
for t = 1:num_sampling
    for a = 1:4
        fprintf("\n=== %d 种情况，锚点数为：%d，锚点选择方式：%d ===\n", t, anchors(t), a);
        [F, obj, runtime, alphaA] = AHD_EC(k, order, X, anchors(t), c, a);
        [ACC, MIhat, Purity,  Fscore, ~, ~, ~] = ClusteringMeasure2(Y, F);
        result_matrix(row_idx, :) = [anchors(t), a, ACC, MIhat, Purity, Fscore, runtime];
        row_idx = row_idx + 1;
    end
end
varNames = {'NumAnchors', 'Method', 'ACC', 'NMI', 'Purity', 'Fscore', 'Runtime'};
result_table = array2table(result_matrix, 'VariableNames', varNames);
fprintf('\n=== 实验结果汇总 ===\n');
disp(result_table);

timestamp = datestr(now, 'yyyymmdd_HHMMSS'); 
csvFileName = sprintf('exp_results_9_%s.csv', timestamp);
savePath = fullfile(resultsDir, csvFileName);
writetable(result_table, savePath);
fprintf('实验结果已成功保存至:\n -> %s\n', savePath);