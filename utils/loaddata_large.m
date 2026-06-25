function [X,Y] = loaddata_large(ind)
if ind ==1
    load('LetterRecognition.mat'); % 20,000
elseif ind ==2
    load('mnist_1w.mat'); % 10,000
elseif ind ==3
    load('mnist_2w.mat'); % 20,000
elseif ind ==4
    load('mnist_3w.mat'); % 30,000
elseif ind ==5
    load('mnist_4w.mat'); % 40,000
elseif ind ==6
    load('mnist_5w.mat'); % 50,000
elseif ind ==7
    load('mnist_6w.mat'); % 60,000
elseif ind ==8
    data = load('mnist_7w.mat'); % 70,000
    X = data.fea;
    Y = data.gt;
elseif ind ==9
    data = load('covtype.mat'); % 581,012
    X = data.fea;
    Y = data.gnd;
end
X=double(X);
Y=double(Y);
end

