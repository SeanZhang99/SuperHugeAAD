%%
clear
clc
close all
log_path = "C:\Users\sean\Downloads\logs\csv_logs\lightning_logs\";
%%
for version = [22,27,29,34,26,38,40]
    close all
    plot_comparison_violin_plot(...
        fullfile(log_path,sprintf("version_%d",version),"metrics.csv"),...
        "KUL", ...
        "test", ...
        "E:\RAW\KUL\exg\raw");
end