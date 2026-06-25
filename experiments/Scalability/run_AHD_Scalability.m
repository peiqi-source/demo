% =========================================================================
% 大规模可拓展性研究 (Scalability Study) 自动化测试脚本
% 数据集: MNIST (5000 到 70000，步长 5000)
% 对比算法: LWEA, LWGP, ECPCS_HC, DREC, PTA-CL, PTGP
% 提出算法: AHD (直接处理原始数据)
% =========================================================================

clear; clc; close all;
warning('off', 'all'); % 关闭由于数据过大可能产生的各种警告

%% 1. 环境与基础设置
addpath('func');
addpath('MNIST_Scalability_Data');
resultsDir = fullfile(pwd, 'results_scalability');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

detailsDir = fullfile(resultsDir, 'Scalability_Details');
if ~exist(detailsDir, 'dir'), mkdir(detailsDir); end

% ==========================================
% [升级点 1]: 数据集规模设定 5000:5000:70000
% ==========================================
data_sizes = 50000:5000:70000;

% ==========================================
% [升级点 2]: 将 LWGP 和 PTA-CL 加入算法序列
% ==========================================
methods_name = {'LWEA', 'LWGP', 'ECPCS_HC', 'DREC', 'PTA-CL', 'PTGP', 'AHD'};
num_methods = length(methods_name);

%% 全局超参数
total_M = 100; % 大池子容量
M = 20;                % [接口]: 集成规模 (为对比算法统一生成 20 个基聚类)
num_seeds = 20;        % 每个算法运行次数 (测试寻优稳定性)
timestamp_global = datestr(now, 'yyyymmdd_HHMMSS'); 

% [AHD 算法专属参数] (参考自你 MNIST 的配置，可按需微调)
ahd_ar = 80; 
ahd_k  = 5; 
ahd_ord = 3; 
ahd_ns = 3; 
ahd_delta = 5;

% [LWEA 算法专属参数] (参考自你 MNIST 的配置，可按需微调)
para_theta = 0.4;                     % 参数 theta 
clsNums = 2:30;                       % 聚类数目搜索范围 

% [ECPCS_HC 算法专属参数] (参考自你 MNIST 的配置，可按需微调)
t = 20;             % ECPCS-HC 内部随机游走步长 (论文默认 20)

% [DREC 算法专属参数] (参考自你 MNIST 的配置，可按需微调)
lambda = 100;         % DREC 的固定正则化参数


%% 2. 初始化全局汇总 CSV 文件
masterCsvName = fullfile(resultsDir, sprintf('Master_Scalability_MNIST_7Methods_%s.csv', timestamp_global));
fid_master = fopen(masterCsvName, 'w', 'n', 'GBK');
if fid_master == -1
    error('无法创建主汇总 CSV 文件，请检查路径。');
end

header_master = {'Dataset_Size', 'Method', 'N_Samples', ...
                 'ACC(Mean±Std)', 'ACC(Max)', 'NMI(Mean±Std)', 'NMI(Max)', ...
                 'PUR(Mean±Std)', 'PUR(Max)', 'Fscore(Mean±Std)', 'Fscore(Max)', ...
                 'ARI(Mean±Std)', 'ARI(Max)', 'Runtime(s)'};
fprintf(fid_master, '%s\n', strjoin(header_master, ','));

disp('====================================================================');
disp('>>> 启动可拓展性研究: MNIST 5000 -> 70000 (步长 5000)');
disp(['>>> 对比算法包含: ', strjoin(methods_name, ', ')]);
disp(['>>> 终极总表将保存至: ', masterCsvName]);
disp('====================================================================');

%% 3. 开始遍历数据规模 (5000 到 70000)
for idx = 1:length(data_sizes)
    d_size = data_sizes(idx);
    size_str = sprintf('%d', d_size);
    fileName = sprintf('MNIST_%d.mat', d_size);
    
    fprintf('\n--------------------------------------------------------------------\n');
    fprintf('>>> 正在处理: MNIST 规模 N = %s ...\n', size_str);
    
    if ~exist(fileName, 'file')
        warning('未找到文件: %s，请确认它在当前文件夹下！跳过...', fileName);
        continue;
    end

    data = load(fileName);
    X = double(data.X);
    Y = double(data.Y);

    [num, dim] = size(X);
    c = length(unique(Y)); % 真实簇数

    clsNums = unique([2:30, c]); % 供 LWEA/LWGP 搜索使用
    
    % --- 预处理防坑 ---
    X_norm = X ./ max(X, [], 2); % 归一化
    X_norm(isnan(X_norm)) = 0;        % 将是Nan的地方赋值为0，防止除以0产生NaN导致 kmeans 报错
    
    % =====================================================================
    % [预处理]: 使用 litekmeans 极速生成 100 个基聚类池 (供 6 个基线算法共享)
    % =====================================================================
    fprintf('    [预处理] 正在对 %d 个样本执行 litekmeans 生成 %d 个基聚类大池...\n', num, total_M);
    baseCls_Pool = zeros(num, total_M);
    for m = 1:total_M
        baseCls_Pool(:, m) = litekmeans(X_norm, c);
    end
    fprintf('    [预处理] 基聚类大池生成完毕！进入同步重采样评估阶段...\n');
    
    res_pool = cell(1, num_methods);
    for i = 1:num_methods
        res_pool{i} = zeros(num_seeds, 6);
    end
    raw_detail_cell = cell(num_seeds * num_methods, 9);
    row_ptr = 1;
    
    %% 4. 执行 20 次 Seed 实验 
    for s_idx = 1:num_seeds
        fprintf('   ========== seed_idx:%d ==========\n', s_idx);
        seed = d_size + s_idx * 100; % 使用规模值 + 序号作为种子，确保不同规模种子不同
        rng(seed, 'twister'); 

        % ---------------------------------------------------------
        % 【核心】：同步重采样！从 100 个里面抽出 20 个。
        % 在当前的 seed 循环下，所有的对比算法都将共享这 20 个基聚类！
        % ---------------------------------------------------------
        sampled_indices = randperm(total_M, M);
        baseCls = baseCls_Pool(:, sampled_indices);
        
        for m_idx = 1:num_methods
            algo = methods_name{m_idx};
            is_oom = false;
            
            try
                switch algo
                    case 'LWEA'
                        tic; 
                        [bcs, baseClsSegs] = getAllSegs(baseCls);
                        ECI = computeECI(bcs, baseClsSegs, para_theta);
                        LWCA = computeLWCA(baseClsSegs, ECI, M);       
                        predY_all = runLWEA(LWCA, clsNums);
                        [~, loc] = ismember(c, clsNums);
                        pred_y = predY_all(:, loc);
                        t_run = toc;
                        fprintf('    >>> LWEA done ...\n')

                    case 'LWGP'
                        % [新增]: LWGP 调用逻辑
                        tic; 
                        [bcs, baseClsSegs] = getAllSegs(baseCls);
                        ECI = computeECI(bcs, baseClsSegs, para_theta);
                        LWCA = computeLWCA(baseClsSegs, ECI, M);       
                        predY_all = runLWGP(bcs, baseClsSegs, ECI, clsNums);
                        [~, loc] = ismember(c, clsNums);
                        pred_y = predY_all(:, loc);
                        t_run = toc;
                        fprintf('    >>> LWGP done ...\n')
                        
                    case 'ECPCS_HC'
                        tic; 
                        pred_y = ECPCS_HC(baseCls, c, t); 
                        t_run = toc;
                        fprintf('    >>> ECPCS_HC done ...\n')
                        
                    case 'DREC'
                        tic; 
                        OutD = DREC(baseCls, c, lambda); 
                        pred_y = OutD.Blable;
                        t_run = toc;
                        fprintf('    >>> DREC done ...\n')

                    case 'PTA-CL'
                        % [新增]: PTA-CL 调用逻辑
                        tic;
                        [mcBaseCls, mcLabels] = computeMicroclusters(baseCls);
                        tilde_N = size(mcBaseCls, 1);
                        MCA = computeMCA(mcBaseCls);
                        
                        para.K = min(20, floor(sqrt(tilde_N)/2));
                        para.T = min(20, floor(sqrt(tilde_N)/2));
                        
                        PTS = computePTS_fast_v3(MCA, mcLabels, para);
                        % 第二个返回值即为 Complete-Link (PTA-CL)
                        [~, mcResultsCL, ~] = runPTA_v2(PTS, c);
                        resultsCL = mapMicroclustersBackToObjects(mcResultsCL, mcLabels);
                        pred_y = resultsCL(:, 1); 
                        t_run = toc;
                        
                    case 'PTGP'
                        tic;
                        [mcBaseCls, mcLabels] = computeMicroclusters(baseCls);
                        tilde_N = size(mcBaseCls, 1);
                        MCA = computeMCA(mcBaseCls);
                        
                        para.K = min(20, floor(sqrt(tilde_N)/2));
                        para.T = min(20, floor(sqrt(tilde_N)/2));
                        
                        PTS = computePTS_fast_v3(MCA, mcLabels, para);
                        mcResultsPTGP = runPTGP_v2(mcBaseCls, PTS, c);
                        resultsPTGP = mapMicroclustersBackToObjects(mcResultsPTGP, mcLabels);
                        pred_y = resultsPTGP(:, 1); 
                        t_run = toc;
                        fprintf('    >>> PTGP done ...\n')
                        
                    case 'AHD'
                        % 构建当前规模下的 Anchors
                        anchors = [];
                        for t = 1:ahd_ns
                            anchors = [anchors, (ahd_ar+(t-1)*ahd_delta)*c];
                        end
                        % 调用你自己的 AHD 算法，直接使用其内部精准返回的 t_run
                        [pred_y, ~, t_run, ~] = AHD_EC(ahd_k, ahd_ord, X, anchors, c);
                        fprintf('    >>> AHD done ...\n')
                end
                
                % 统一评估 5 大指标
                [acc, nmi, pur, fsc, ~, ~, ~, ari] = ClusteringMeasure4(Y, pred_y);
                
                res_pool{m_idx}(s_idx, :) = [acc, nmi, pur, fsc, ari, t_run];
                raw_detail_cell(row_ptr, :) = {size_str, algo, seed, acc, nmi, pur, fsc, ari, t_run};
                row_ptr = row_ptr + 1;
                
            catch ME
                fprintf('    [熔断] %s 算法在 %s 数据上发生异常: %s\n', algo, size_str, ME.message);
                res_pool{m_idx}(s_idx, :) = NaN(1, 6); 
                raw_detail_cell(row_ptr, :) = {size_str, algo, seed, NaN, NaN, NaN, NaN, NaN, NaN};
                row_ptr = row_ptr + 1;
            end
        end % end methods
    end % end seed
    
    %% 5. 结果结算与写入总表
    for m_idx = 1:num_methods
        cur_res = res_pool{m_idx};
        
        if all(isnan(cur_res(:, 1)))
            fprintf(fid_master, '%s,%s,%d,%s\n', size_str, methods_name{m_idx}, num, repmat('OOM,', 1, 10));
        else
            avg_m = mean(cur_res, 1, 'omitnan');
            std_m = std(cur_res, 0, 1, 'omitnan');
            max_m = max(cur_res, [], 1, 'omitnan');
            
            str_stats = cell(1, 5);
            for i = 1:5
                str_stats{i} = sprintf('%.4f±%.4f', avg_m(i), std_m(i));
            end
            
            fprintf(fid_master, '%s,%s,%d,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%.4f\n', ...
                size_str, methods_name{m_idx}, num, ...
                str_stats{1}, max_m(1), str_stats{2}, max_m(2), ...
                str_stats{3}, max_m(3), str_stats{4}, max_m(4), ...
                str_stats{5}, max_m(5), avg_m(6));
            
            fprintf('    -> [%-6s] ACC: %s | T: %.4fs\n', methods_name{m_idx}, str_stats{1}, avg_m(6));
        end
    end
    
    % 保存详情
    detailCsv = fullfile(detailsDir, sprintf('Scalability_Details_%s_%s.csv', size_str, timestamp_global));
    varNames = {'Dataset_Size', 'Method', 'Seed', 'ACC', 'NMI', 'PUR', 'Fscore', 'ARI', 'Runtime'};
    detailTable = cell2table(raw_detail_cell, 'VariableNames', varNames);
    writetable(detailTable, detailCsv);

end

fclose(fid_master);
disp('====================================================================');
fprintf('[SUCCESS] 可拓展性与对比实验全部结束！\n汇总表：%s\n', masterCsvName);
disp('====================================================================');