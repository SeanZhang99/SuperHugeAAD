%%
clear
clc
close all
log_path = "C:\Users\sean\Downloads\logs\csv_logs\lightning_logs\";
data_path = "E:\RAW\DTU\sphg_preprocessed";
%%
for version = 41:52
    close all
    plot_factorwise_comparison(...
        fullfile(log_path,sprintf("version_%d",version),"metrics.csv"),...
        "DTU", ...
        "val", ...
        data_path);
end
%%
close all
plot_trialwise_comparison(21:40,log_path,"DTU","train",data_path);