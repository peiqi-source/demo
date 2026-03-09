function Y = Y_Initialize_rand(S, c)
% 随机 one-hot 初始化
[n, ~] = size(S{1});
labels = 1:c;
labels = [labels, randi(c, 1, n - c)];
labels = labels(randperm(n));
Y = ind2vec(labels)';
Y = full(Y);
end