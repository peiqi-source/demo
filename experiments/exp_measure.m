%% exp_measure_speed.m 性能指标提升实验
clear;
clc;
close all;

%% setup paths
thisFile = mfilename("fullpath");
expDir = fileparts(thisFile);
rootDir = fileparts(expDir);
resultsDir = fullfile(rootDir, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end % 确保结果目录存在

%% load data
[X, Y] = loaddata2(11);

X = X./max(X, [], 2);
c = length(unique(Y));

%% set parameter
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
total_exp =  5;
result_matrix = zeros(total_exp, 5); 
row_idx = 1;
for seed = 1:5
    rng(seed);
    fprintf(">>> Seed：%d | ", seed);
    [F, obj, runtime, alphaA] = AHD_EC(k, order, X, anchors, c);
    fprintf("Over ... ");
    [ACC, MIhat, Purity,  Fscore, ~, ~, ~] = ClusteringMeasure4(Y, F);
    fprintf("Done ... \n");
    result_matrix(row_idx, :) = [ACC, MIhat, Purity, Fscore, runtime];
    row_idx = row_idx + 1;
end
varNames = {'ACC', 'NMI', 'Purity', 'Fscore', 'Runtime'};
result_table = array2table(result_matrix, 'VariableNames', varNames);
fprintf('\n=== 实验结果汇总 ===\n');
disp(result_table);

timestamp = datestr(now, 'yyyymmdd_HHMMSS'); 
csvFileName = sprintf('results_MNIST_full_%s.csv', timestamp);
savePath = fullfile(resultsDir, csvFileName);
writetable(result_table, savePath);
fprintf('实验结果已成功保存至:\n -> %s\n', savePath);