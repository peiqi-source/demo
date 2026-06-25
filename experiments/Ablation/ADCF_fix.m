function [f_label, obj, runtime, alphaA] = ADCF_fix(H, F_init, use_AW)
% ADCF_fix
% Checked against the original ADCF.m.
% Only one ablation switch is added:
%   use_AW: if false, alpha is fixed as uniform weights 1/V.

if nargin < 3
    use_AW = true;
end

tic;
NITR = 100;
[num, c] = size(F_init);
V = length(H);
mu = 1e-4;
rho = 1.01;
alpha = ones(V, 1) / V;
alphaA = alpha;

%% Convert sparse indicator matrix into label vector for O(1) updates
if size(F_init, 2) > 1
    [~, f_label] = max(F_init, [], 2);
    F_sparse = F_init;
else
    f_label = F_init;
    F_sparse = sparse(1:num, f_label, 1, num, c);
end

%% Precompute graph-level similarities by trace trick
B1 = zeros(V, V);
for u = 1:V
    for v = u:V
        val = sum((H{u}' * H{v}).^2, 'all');
        B1(u, v) = val;
        B1(v, u) = val;
    end
end

%% Extract labels of all base partitions
h_v = zeros(num, V);
for v = 1:V
    [~, h_v(:, v)] = max(H{v}, [], 2);
end

%% Initialize intersection matrices
C = cell(1, V);
for v = 1:V
    C{v} = H{v}' * F_sparse;
end

%% Initialize global statistics
ff = sum(F_sparse, 1);
fsf = zeros(1, c);
for v = 1:V
    fsf = fsf + alpha(v) * sum(C{v}.^2, 1);
end
sii = sum(alpha);

sum_S2 = alpha' * B1 * alpha;
obj(1) = sum_S2 - 2 * sum(fsf ./ (ff + eps)) + c;
changed = zeros(NITR, 10); %#ok<NASGU>

%% Discrete coordinate-wise optimization
for iter = 1:NITR
    for it = 1:10
        converged = true;
        for i = 1:num
            m = f_label(i);
            if m == 0, continue; end

            ui = zeros(1, c);
            for v = 1:V
                ui = ui + alpha(v) * C{v}(h_v(i, v), :);
            end

            del = (fsf + 2 * ui + sii) ./ (ff + 1 + eps) - (fsf ./ (ff + eps));
            f0_m = (fsf(m) - 2 * ui(m) + sii) / (ff(m) - 1 + eps);
            del(m) = fsf(m) / (ff(m) + eps) - f0_m;

            [~, p] = max(del);

            if p ~= m
                converged = false;

                ff(m) = ff(m) - 1;
                ff(p) = ff(p) + 1;
                fsf(m) = fsf(m) - 2 * ui(m) + sii;
                fsf(p) = fsf(p) + 2 * ui(p) + sii;

                for v = 1:V
                    q = h_v(i, v);
                    C{v}(q, m) = C{v}(q, m) - 1;
                    C{v}(q, p) = C{v}(q, p) + 1;
                end

                f_label(i) = p;
            end
        end
        if converged, break; end
    end

    %% Adaptive weighting switch
    if use_AW
        b = zeros(V, 1);
        for v = 1:V
            b(v) = 2 * sum(sum(C{v}.^2, 1) ./ (ff + eps));
        end
        [alpha, ~, ~] = ALM(B1, b, mu, rho);
    end

    alphaA = [alphaA, alpha]; %#ok<AGROW>

    fsf = zeros(1, c);
    for v = 1:V
        fsf = fsf + alpha(v) * sum(C{v}.^2, 1);
    end
    sii = sum(alpha);
    sum_S2 = alpha' * B1 * alpha;
    obj(iter + 1) = sum_S2 - 2 * sum(fsf ./ (ff + eps)) + c;

    if iter > 1 && abs((obj(iter + 1) - obj(iter)) / obj(iter + 1)) < 1e-10
        break;
    end
    if iter > 30 && sum(abs(obj(iter-9:iter-5) - obj(iter-4:iter))) < 1e-10
        break;
    end
end

runtime = toc;
end
