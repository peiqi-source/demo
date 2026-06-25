function [ anchor, ind2, score ] = graphgen_anchor(X, m )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[n,d] = size(X);
% 对每一列（每一维特征）求最小值，得到 1×d
vm = min(X,[],1);
% 最小值行向量复制成 n×d
Xm = ones(n,1)*vm;
% 减去买一列的最小值，让每一维变非负
X = X-Xm;

% 除以该维最大值，让每维落在 [0,1]
% 如果不同 view 的尺度差异很大时，建议打开
% for i=1:d
%     maxd=max(X(:,i));
%     X(:,i)=X(:,i)./maxd;
% end

% 对每一行求和（所有维度的特征信息求和），得到 n×1 的分数向量
score = sum(X, 2);
% 先归一化再 argmax，分数最大的为第一个锚点
score(:,1) = score/max(score);
[~,ind(1)] = max(score);
% 交替更新锚点选择
for i=2:m
   % 极小/极大都会被压到 0，最大在 0.5：中等分数会被放大
   % 有一定差异的但是差异又不是最大的选为锚点
   score(:,i) = score(:,i-1).*(ones(n,1)-score(:,i-1));
   score(:,i) = score(:,i)/max(score(:,i));
   [~,ind(i)] = max(score(:,i));
end
ind2 = sort(ind,'ascend');
anchor = X(ind2,:);

% for i=1:4
% % idd=find(score(:,i)>0.98);
% % figure;
% % plot(X(:,1),X(:,2),'.b', 'MarkerSize', 10); hold on;
% % plot(X(idd,1),X(idd,2),'.r', 'MarkerSize', 10); hold on;
% figure; plot(score(:,i),'-o');
% % axis equal;
% set(gcf,'Position',[400,100,700,600],'color','w');
% set(gca,'fontsize',16);
% set(gca,'linewidth',0.8);
% % saveas(gcf,strcat('C:\Users\opt\Desktop\mydisk\ongoing\Fast multiview CLR\Latex_revised2\response\s',num2str(i),'.pdf'));
% end


% id=zeros(1,m);
% for i=1:m
%     if ind(i)<=200
%         id(i)=1;
%     elseif ind(i)<=400
%         id(i)=2;
%     else
%         id(i)=3;
%     end
% end
end
