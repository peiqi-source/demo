function [X,Y] = loaddata(ind)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
if ind == 1
    load('coil-20_1000.mat');
elseif ind ==2
    load('coil-100_1000.mat');
elseif ind ==3
    load('MSRA25_1024.mat');
elseif ind ==4
    load('PIE_vec_rate20.mat');
elseif ind ==5
    load('minist_5000_50.mat');
elseif ind ==6
    load('PIE_vec.mat');
elseif ind ==11
    data = load('OpticDigits.mat');
    X = data.fea;
    Y = data.gnd;
elseif ind ==12
    data = load('USPS.mat');
    X = data.fea;
    Y = data.gnd;
elseif ind ==13
    data = load('MNIST_full.mat');
    X = data.fea;
    Y = data.gt;
elseif ind ==14
    data = load('PenDigits.mat');
    X = data.fea;
    Y = data.gnd;
elseif ind == 15
    data = load('yeast.mat');
    X = data.fea;
    Y = data.gnd;
end


