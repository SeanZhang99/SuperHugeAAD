function plot_factorwise_comparison(log_path,dataset,target_stage,data_path)
    t = readcell(log_path);
    switch dataset
        case "DTU"
            dataset_id = 2;
            patterns = ["attend_mf","attend_lr","acoustic_condition","subject"];
        case "KUL"
            dataset_id = 4;
            patterns = ["attended_ear","condition","subject"];
    end
    %%
    trialwise_acc = find_trialwise_acc(t,dataset_id,target_stage);

    subjects = unique([trialwise_acc{:,2}]);

    expinfo = table;
    for subject = subjects
        switch dataset
            case "DTU"
                expinfo_tmp = load(fullfile(data_path,sprintf("S%d_data_preproc.mat",subject)),'expinfo').expinfo;
                expinfo_tmp = expinfo_tmp(expinfo_tmp.n_speakers==2,:);
                expinfo_tmp.subject(1:60) = subject;
                
            case "KUL"
                trials = [load('E:\RAW\KUL\exg\raw\S6.mat').trials{:}];
                trials = trials(1:8);
                expinfo_tmp = table(cellfun(@string,{trials.attended_ear})',cellfun(@string,{trials.condition})','VariableNames',["attended_ear","condition"]);
                expinfo_tmp.subject(1:8) = subject;
        end
        expinfo = [expinfo;expinfo_tmp];
    end

    acc = cell2mat(trialwise_acc(:,3)');
    [~,best_acc_idx] = max(mean(acc,2));
    acc = acc(best_acc_idx,:);
    %%
    figure;
    set(gcf,"Position",[0 0 1920 1080])
    i = 0;
    for pattern = patterns
        group = table2array(expinfo(:,pattern));
        i = i + 1;
        subplot(2,2,i);
        violinplot(acc,group,'ShowMean',true,'ShowNotch',true)
        title(sprintf("%s vs Accuracy",pattern),'Interpreter','none')
    end
end