function Y = Y_Initialize(n, c)
% 随机 one-hot 初始化
labels = 1:c;
labels = [labels, randi(c, 1, n - c)];
labels = labels(randperm(n));
Y = ind2vec(labels)';
Y = full(Y);
end