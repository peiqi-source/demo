%% exp_measure.m 性能指标提升实验
clear;
clc;
close all;

%% setup paths
thisFile = mfilename("fullpath");
expDir = fileparts(thisFile);
rootDir = fileparts(expDir);
addpath(genpath(rootDir));

%% load data
[X, Y] = loaddata(8);

X = X./max(X, [], 2);
c = length(unique(Y));

%% set parameter
rng(1);
k = 5;
order = 3;
num_sampling = 3;
anchors_init = 10;
delta = 5;

anchors = [];
for t = 0:num_sampling
    anchors = [anchors_init, anchors_init*c+t*delta*c];
end

%% 
result = [];
for t = 1:num_sampling
   [F, obj, runtime, alphaA] = AHD_EC(k, order, X, anchors(t), c);
   res = ClusteringMeasure1(Y, F);
   ressult = [result, res];
end