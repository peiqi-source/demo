% =========================================================================
% 前后端解耦消融实验 - 全细节过程记录版 (完美适配 ADCF.m)
% 目的: 1. 验证前端生成策略 (AHD Front vs K-means) 对后端共识函数的影响
%       2. 完整记录 AHD 后端的优化细节 (obj, alpha) 及所有算法的 20 次 Seed 表现
% 保存: 每一个数据集生成一个详细过程 CSV，最后生成一个 Master 汇总表
% =========================================================================

clear; clc; close all;
warning('off', 'all');
addpath('func');

%% 1. 环境与路径设置
dir_Kmeans = 'Base_LabelsPool_MAT_TrueC'; % 基聚类池路径
dir_AHD = 'AHD_BasePool9_MAT'; % 基聚类池路径

% dir_Kmeans = fullfile(pwd, 'Base_LabelsPool_MAT_TrueC');  
% dir_AHD    = fullfile(pwd, 'AHD_BasePool9_MAT');          

% 创建本次实验的专属文件夹
timestamp_root = datestr(now, 'yyyymmdd_HHMMSS');
mainResultsDir = fullfile(pwd, ['AHD_Backend_Ablation_details_', timestamp_root]);
if ~exist(mainResultsDir, 'dir'), mkdir(mainResultsDir); end

datasetNames = {'UMIST', 'VS', 'COIL20', 'SPF', 'IS', 'FCT', ...
                'MNIST', 'OptDigits', 'LS', 'ISOLET', 'USPS', 'PenDigits'};
data_list = 1:12;

% 实验维度
frontends = {'Kmeans_TrueC', 'AHD_Front'};
backends  = {'LWEA', 'ECPCS_HC', 'DREC', 'PTGP', 'ADCF'};
num_seeds = 20;
M = 9; % 统一 9 个基聚类

% 后端参数
para_theta = 0.4;    % LWEA
t_ecpcs = 20;        % ECPCS_HC
lambda = 100;        % DREC

%% 2. 初始化 Master 汇总 CSV (存 Mean±Std)
masterCsvName = fullfile(pwd, sprintf('Master_Ablation_Summary_MeanStd_%s.csv', timestamp_root));
fid_master = fopen(masterCsvName, 'w', 'n', 'GBK');
header_master = {'Dataset', 'Front_End', 'Back_End', 'N', 'K', ...
                 'ACC(Mean±Std)', 'ACC(Max)', 'NMI(Mean±Std)', 'NMI(Max)', ...
                 'PUR(Mean±Std)', 'PUR(Max)', 'Fscore(Mean±Std)', 'Fscore(Max)', ...
                 'ARI(Mean±Std)', 'ARI(Max)', 'Avg_Runtime(s)'};
fprintf(fid_master, '%s\n', strjoin(header_master, ','));

disp('====================================================================');
disp(['>>> 启动全细节消耦消融实验 | 完美适配 ADCF 后端']);
disp(['>>> 保存目录: ', mainResultsDir]);
disp('====================================================================');

%% 3. 数据集主循环
for data_idx = data_list
    dataname = datasetNames{data_idx};
    fprintf('\n>>> 处理数据集: %02d/12 [%s] <<<\n', data_idx, dataname);
    
    % --- 加载数据池 ---
    file_K = fullfile(dir_Kmeans, sprintf('LabelsPool_%s.mat', dataname));
    file_A = fullfile(dir_AHD, sprintf('AHD_BasePool9_%s.mat', dataname));
    if ~exist(file_K, 'file') || ~exist(file_A, 'file')
        warning('数据池文件缺失，跳过 %s', dataname);
        continue; 
    end
    
    dataK = load(file_K); Y = double(dataK.Y(:)); c = length(unique(Y)); N = length(Y);
    if size(dataK.base_labels, 1) > size(dataK.base_labels, 2), pool_K = dataK.base_labels; else, pool_K = dataK.base_labels'; end
    dataA = load(file_A); pool_A = dataA.label_pool;
    
    % 每个数据集准备一个过程记录 Cell (2个前端 * 5个后端 * 20个Seed = 200行)
    dataset_detail_cell = cell(length(frontends)*length(backends)*num_seeds, 14);
    detail_row_ptr = 1;

    for fe_idx = 1:length(frontends)
        fe_name = frontends{fe_idx};
        for be_idx = 1:length(backends)
            be_name = backends{be_idx};
            
            current_group_metrics = zeros(num_seeds, 6); 
            is_oom = false;

            for s_idx = 1:num_seeds
                seed = data_idx*1000 + fe_idx*100 + be_idx*10 + s_idx;
                rng(seed, 'twister');
                
                % 准备 baseCls (N x 9 标签矩阵)
                if strcmp(fe_name, 'Kmeans_TrueC')
                    sel_idx = randperm(size(pool_K, 2), M);
                    baseCls = pool_K(:, sel_idx);
                else
                    baseCls = pool_A(:, 1:M); 
                end
                
                % 初始化 AHD 特有变量占位符
                obj_str = 'N/A'; alpha_str = 'N/A';

                try
                    switch be_name
                        case 'LWEA'
                            tic; [bcs, baseClsSegs] = getAllSegs(baseCls);
                            ECI = computeECI(bcs, baseClsSegs, para_theta);
                            LWCA = computeLWCA(baseClsSegs, ECI, M);       
                            pred_y_all = runLWEA(LWCA, unique([2:30, c]));
                            [~, loc] = ismember(c, unique([2:30, c]));
                            pred_y = pred_y_all(:, loc); t_run = toc;
                            
                        case 'ECPCS_HC'
                            tic; pred_y = ECPCS_HC(baseCls, c, t_ecpcs); t_run = toc;
                            
                        case 'DREC'
                            tic; OutD = DREC(baseCls, c, lambda); pred_y = OutD.Blable; t_run = toc;
                            
                        case 'PTGP'
                            tic; [mcB, mcL] = computeMicroclusters(baseCls); 
                            MCA = computeMCA(mcB); para.K = 20; para.T = 20;
                            PTS = computePTS_fast_v3(MCA, mcL, para);
                            mcRes = runPTGP_v2(mcB, PTS, c);
                            res = mapMicroclustersBackToObjects(mcRes, mcL);
                            pred_y = res(:, 1); t_run = toc;
                            
                        case 'ADCF'
                            % --------------------------------------------------
                            % [完美适配] 将 N x 9 转化为 ADCF 专属的稀疏 H_pool
                            % --------------------------------------------------
                            H_pool = cell(1, M);
                            for v = 1:M
                                max_c_v = max(baseCls(:, v));
                                H_pool{v} = sparse(1:N, baseCls(:, v), 1, N, max_c_v);
                            end
                            F_init = H_pool{1}; % 【修复点】直接传入稀疏指示矩阵，完美对齐原版 AHD_EC 接口！
                            
                            % 调用你自己的纯后端 ADCF
                            [pred_y, obj, t_run_adcf, alphaA] = ADCF(H_pool, F_init);  
                            t_run = t_run_adcf; % 直接使用你 ADCF 内部掐秒的精准时间
                            
                            % 转换矩阵为字符串用于 CSV 过程记录
                            obj_str = mat2str(round(obj, 6));
                            alpha_str = mat2str(round(alphaA, 4));
                    end
                    
                    % 评估性能
                    [acc, nmi, pur, fsc, ~, ~, ~, ari] = ClusteringMeasure4(Y, pred_y);
                    
                    % 填入过程记录表 (Detailed Row)
                    dataset_detail_cell(detail_row_ptr, :) = { ...
                        dataname, fe_name, be_name, seed, N, c, ...
                        acc, nmi, pur, fsc, ari, t_run, obj_str, alpha_str};
                    
                    % 存入统计矩阵
                    current_group_metrics(s_idx, :) = [acc, nmi, pur, fsc, ari, t_run];
                    detail_row_ptr = detail_row_ptr + 1;

                catch ME
                    is_oom = true;
                    fprintf('    [崩溃] %s-%s Seed %d: %s\n', fe_name, be_name, s_idx, ME.message);
                    break;
                end
            end % end seed
            
            % --- 写入 Master 汇总表 ---
            if is_oom
                fprintf(fid_master, '%s,%s,%s,%d,%d,%s\n', dataname, fe_name, be_name, N, c, repmat('OOM,', 1, 11));
            else
                avg_m = mean(current_group_metrics, 1, 'omitnan');
                std_m = std(current_group_metrics, 0, 1, 'omitnan');
                max_m = max(current_group_metrics, [], 1, 'omitnan');
                str_stats = cell(1, 5);
                for i = 1:5, str_stats{i} = sprintf('%.4f±%.4f', avg_m(i), std_m(i)); end
                fprintf(fid_master, '%s,%s,%s,%d,%d,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%.4f\n', ...
                    dataname, fe_name, be_name, N, c, ...
                    str_stats{1}, max_m(1), str_stats{2}, max_m(2), str_stats{3}, max_m(3), ...
                    str_stats{4}, max_m(4), str_stats{5}, max_m(5), avg_m(6));
                
                fprintf('  |-- [%-12s + %-8s] ACC = %s\n', fe_name, be_name, str_stats{1});
            end
        end % end backend
    end % end frontend
    
    % --- 保存当前数据集的详细过程表 (Detailed Process Table) ---
    detailCsvName = fullfile(mainResultsDir, sprintf('Detail_Process_%s.csv', dataname));
    detailVarNames = {'Dataset', 'Front_End', 'Back_End', 'Seed', 'N', 'K', ...
                      'ACC', 'NMI', 'PUR', 'Fscore', 'ARI', 'Runtime', 'Obj_History', 'Alpha_Weights'};
    valid_rows = ~cellfun(@isempty, dataset_detail_cell(:, 1));
    T_detail = cell2table(dataset_detail_cell(valid_rows, :), 'VariableNames', detailVarNames);
    writetable(T_detail, detailCsvName);
end

fclose(fid_master);
disp('====================================================================');
fprintf('[DONE] 消融实验细节记录完毕！所有文件已保存至文件夹：\n%s\n', mainResultsDir);
disp('====================================================================');