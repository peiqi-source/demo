function [X,Y] = loaddata(ind)
%   此处显示详细说明
if ind == 1
    load('ecoli.mat'); % 336
elseif ind ==2
    data = load('LS.mat'); % 6435 x 100
    X = data.members;
    Y = data.gt;
elseif ind ==3
    data = load('Texture.mat'); % 5500 x 100
    X = data.members;
    Y = data.gt;
elseif ind ==4
    data = load('Caltech20.mat'); % 2386 x 100
    X = data.members;
    Y = data.gt;
elseif ind ==5
    data = load('Umist.mat'); % 575 x 1024
    X = data.fea;
    Y = data.gnd;
elseif ind ==6
    load('SPF.mat'); % 1941 x 27
end


