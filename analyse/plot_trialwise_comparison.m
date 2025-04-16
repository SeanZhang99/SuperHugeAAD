function [acc,group] = plot_trialwise_comparison(versions,log_path,dataset,target_stage,data_path)
acc = [];
group = string.empty;
for version = versions
    t = readcell( fullfile(log_path,sprintf("version_%d",version),"metrics.csv"));
    switch dataset
        case "DTU"
            dataset_id = 2;
        case "KUL"
            dataset_id = 4;

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

    acc_tmp = cell2mat(trialwise_acc(:,3)');
    [~,best_acc_idx] = max(mean(acc_tmp,2));
    acc_tmp = acc_tmp(best_acc_idx,:);
    for i=1:length(acc_tmp)
        group_tmp = split(trialwise_acc{i,1},["_","/"]);
        group(end+1) = string(group_tmp{3});
        acc(end+1) = acc_tmp(i);
    end
end
%%
nsubject = fastif(dataset_id==2,18,16);
for subject = 1:nsubject
    pattern = sprintf("dataset-%03d-subject-%03d-.*?",dataset_id,subject);
    match = regexp(group,pattern);
    idx = ~cellfun(@isempty,match);
    figure;
    set(gcf,"Position",[0 0 1920 1080])
    violinplot(acc(idx),group(idx),'ShowMean',true,'ShowNotch',true)
    title("Trial vs Accuracy",'Interpreter','none')
end
end