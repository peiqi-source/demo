function [X,Y] = loaddata(ind)
%   此处显示详细说明
if ind == 1
    data = load('Umist.mat'); % 575 x 1024
    X = data.fea;
    Y = data.gnd;
elseif ind ==2
    load('VS.mat'); % 5500 x 100
elseif ind ==3
    data = load('COIL20.mat'); % 1,440
    X = data.fea;
    Y = data.gnd;
elseif ind ==4
    load('SPF.mat'); % 1941 x 27
elseif ind ==5
    load('IS.mat'); 
elseif ind ==6
    load('FCT.mat');
elseif ind == 7
    load('MNIST.mat'); % 7,200
elseif ind ==8
    data = load('OpticDigits.mat'); % 5,620
    X = data.fea;
    Y = data.gnd;
elseif ind ==9
    load('LS.mat'); % 6435
elseif ind ==10
    load('ISOLET.mat'); % 7,797
elseif ind ==11
    load('USPS.mat'); % 9,298
elseif ind ==12
    load('PenDigits.mat'); % 10,992
end
X=double(X);
Y=double(Y);
end


