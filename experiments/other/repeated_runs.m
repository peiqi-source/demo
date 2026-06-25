function [ACC, MIhat, Purity, Fscore] = repeated_runs(X, c, anchors, order, num_sampling, k)

%% set seed
final_seeds = 1:20;

num_final_runs = length(final_seeds);
final_results = struct([]);

for r = 1:num_final_runs
    seed = final_seeds(r);
    fprintf('%d / %d | Seed=%d\n', r, num_final_runs, seed);

    rng(seed);

    [F, ~, ~, ~] = AHD_EC(k, order, X, anchors, c);
    [ACC, MIhat, Purity, Fscore, ~, ~, ~] = ClusteringMeasure4(Y, F);

    final_results(r).ACC = ACC;
    final_results(r).NMI = MIhat;
    final_results(r).Purity = Purity;
    final_results(r).Fscore = Fscore;

end

    %% 统计 mean ± std
    ACCs    = [final_results.ACC];
    NMIs    = [final_results.NMI];
    Puritys = [final_results.Purity];
    Fscores = [final_results.Fscore];
    Times   = [final_results.Runtime];

    summary_result.ACC_mean = mean(ACCs);
    summary_result.ACC_std  = std(ACCs);

    summary_result.NMI_mean = mean(NMIs);
    summary_result.NMI_std  = std(NMIs);

    summary_result.Purity_mean = mean(Puritys);
    summary_result.Purity_std  = std(Puritys);

    summary_result.Fscore_mean = mean(Fscores);
    summary_result.Fscore_std  = std(Fscores);

    summary_result.Runtime_mean = mean(Times);
    summary_result.Runtime_std  = std(Times);

    fprintf('\n>>>最终结果（mean ± std）如下：\n');
    fprintf('ACC    = %.4f ± %.4f\n', summary_result.ACC_mean, summary_result.ACC_std);
    fprintf('NMI    = %.4f ± %.4f\n', summary_result.NMI_mean, summary_result.NMI_std);
    fprintf('Purity = %.4f ± %.4f\n', summary_result.Purity_mean, summary_result.Purity_std);
    fprintf('Fscore = %.4f ± %.4f\n', summary_result.Fscore_mean, summary_result.Fscore_std);
    fprintf('Time   = %.4f ± %.4f\n', summary_result.Runtime_mean, summary_result.Runtime_std);

    %% 保存最终重复实验结果
    final_mat_name = fullfile(resultsDir, sprintf('Stage2_Final_Data%d_%s.mat', data_idx, timestamp));
    save(final_mat_name, 'final_results', 'summary_result', 'best_cfg', 'final_seeds', '-v7.3');

    % 导出逐次结果 CSV（不存大矩阵）
    final_cell = cell(num_final_runs, 14);
    for r = 1:num_final_runs
        final_cell(r, :) = { ...
            final_results(r).DatasetID, ...
            mat2str(final_results(r).Anchors), ...
            final_results(r).Order, ...
            final_results(r).NumSampling, ...
            final_results(r).K, ...
            final_results(r).Seed, ...
            final_results(r).ACC, ...
            final_results(r).NMI, ...
            final_results(r).Purity, ...
            final_results(r).Fscore, ...
            final_results(r).P, ...
            final_results(r).R, ...
            final_results(r).RI, ...
            final_results(r).Runtime};
    end

    final_var_names = {'DatasetID','Anchors','Order','NumSampling','K','Seed', ...
                       'ACC','NMI','Purity','Fscore','P','R','RI','Runtime'};
    final_table = cell2table(final_cell, 'VariableNames', final_var_names);
    final_csv_name = fullfile(resultsDir, sprintf('Stage2_Final_Data%d_%s.csv', data_idx, timestamp));
    writetable(final_table, final_csv_name);

    % 导出 summary CSV
    summary_table = struct2table(summary_result);
    summary_csv_name = fullfile(resultsDir, sprintf('Stage2_Summary_Data%d_%s.csv', data_idx, timestamp));
    writetable(summary_table, summary_csv_name);

    fprintf('\n>>> 数据集 %d 全部实验完成并已保存。\n', data_idx);

    clear X Y search_results final_results final_table summary_table;
    fclose('all');