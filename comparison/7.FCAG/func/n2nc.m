function membership_matrix = n2nc(y)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
% 示例标签数组（n*1）

% 获取类别数量
num_classes = max(y);

% 获取样本数量
n = length(y);

% 初始化隶属度矩阵为n*c的零矩阵
membership_matrix = zeros(n, num_classes);

% 遍历每个样本的标签，设置对应的隶属度
for i = 1:n
    class_index = y(i);  % 获取当前样本的类别标签
    membership_matrix(i, class_index) = 1;  % 将对应位置的隶属度设为1
end
end