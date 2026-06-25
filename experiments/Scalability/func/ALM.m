function [alpha, obj1,obj2,mu] = ALM(A1,b,mu,rho)
% min_{d'*1=1, d>=0}  ||p - Md||^2
% min_{d'*1=1, d>=0, d=x}  d'Ax-d'B
NITER=10000;
Eta = ones(size(b));
x = ones(size(b));
cnt=0;
val=0;

obj1 = zeros(NITER,1);
obj2 = zeros(NITER,1);
for iter = 1: NITER
    C=1/mu*(b-Eta-A1*x)+x;
    alpha = EProjSimplex_new(C);
    x=alpha+1/mu*(Eta-A1'*alpha);
    
    Eta = Eta+mu*(alpha-x);
    mu = rho*mu;
    
    val_old=val;
    val=alpha'*A1*alpha-alpha'*b;
    %update objective value
    obj1(iter)=val;
    obj2(iter)=norm(alpha-x,'fro');
    if abs(val - val_old) < 1e-8
        if cnt >= 5
            break;
        else
            cnt = cnt + 1;
        end
    else
        cnt = 0;
    end
    
end
obj1 = obj1(1:iter);
obj2 = obj2(1:iter);
end