function [X,Y] = loaddata(ind)
%   此处显示详细说明
if ind == 1
    data = load('COIL20.mat'); % 1,440
    X = data.fea;
    Y = data.gnd;
elseif ind ==2
    data = load('COIL100.mat'); % 7,200
    X = data.fea;
    Y = data.gnd;
elseif ind ==3
    data = load('yeast.mat'); % 1,484
    X = data.fea;
    Y = data.gnd;
elseif ind ==4
    load('MSRA25.mat'); % 1,799
elseif ind ==5
    data = load('OpticDigits.mat'); % 5,620
    X = data.fea;
    Y = data.gnd;
elseif ind ==6
    load('ISOLET.mat'); % 7,797
elseif ind ==7
    data = load('USPS.mat'); % 9,298
    X = data.fea;
    Y = data.gnd;
elseif ind ==8
    data = load('PenDigits.mat'); % 10,992
    X = data.fea;
    Y = data.gnd;
elseif ind ==9
    load('LetterRecognition.mat'); % 20,000
elseif ind ==10
    data = load('MNIST_1w.mat'); % 10,000
    X = data.fea;
    Y = data.gt;
elseif ind ==11
    data = load('MNIST_full.mat'); % 70,000
    X = data.fea;
    Y = data.gt;
elseif ind ==12
    data = load('covtype.mat'); % 581,012
    X = data.fea;
    Y = data.gnd;
end


