% 设置文件夹路径
folder1 = 'E:\EEG dataset\Data_for_CS 安徽大学\Data_for_CS\data_for_CS\audio-only'; % 请替换为第一个文件夹的路径
folder2 = 'E:\EEG dataset\Data_for_CS 安徽大学\Data_for_CS\data_for_CS\label'; % 请替换为第二个文件夹的路径

% 获取文件夹中所有 .mat 文件
files1 = dir(fullfile(folder1, '*.mat'));
files2 = dir(fullfile(folder2, '*.mat'));

% 获取文件数量（假设两个文件夹中的文件数量相同）
numFiles = min(length(files1), length(files2));

% 循环处理每一对文件
for i = 1:numFiles
    % 读取文件夹1中的第i个文件
    file1 = fullfile(folder1, files1(i).name);
    dataStruct1 = load(file1); % 假设文件中有一个名为 'data' 的变量

    % 读取文件夹2中的第i个文件
    file2 = fullfile(folder2, files2(i).name);
    dataStruct2 = load(file2); % 假设文件中有一个名为 'label' 的变量

    % 合并两个文件的数据
    mergedStruct.data = dataStruct1.data; % 合并数据
    mergedStruct.label = dataStruct2.data; % 合并标签

    % 保存合并后的结构体为新的 .mat 文件
    outputFile = fullfile('E:\EEG dataset\Data_for_CS 安徽大学\Data_for_CS\data_for_CS\audio_label', ['merged_' num2str(i) '.mat']); % 输出文件夹路径
    save(outputFile, 'mergedStruct'); % 保存文件
    disp(['Merged file ' num2str(i) ' and saved as ' outputFile]);
end

disp('All files have been processed.');