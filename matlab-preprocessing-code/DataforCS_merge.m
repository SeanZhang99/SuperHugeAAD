base_path = 'E:\EEG_dataset_Superhuge\Data_for_CS\Data_for_CS\data_for_CS';  % 基本路径
folder1 = sprintf('%s\\audio-only', base_path);  % 第一个文件夹
folder2 = sprintf('%s\\label', base_path);      % 第二个文件夹
outputFolder = fullfile(base_path, 'audio-label');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
% 获取文件夹中所有 .mat 文件
files1 = dir(fullfile(folder1, '*.mat'));
files2 = dir(fullfile(folder2, '*.mat'));
if length(files1) ~= 8 || length(files2) ~= 8
    error('Each folder must contain exactly 8 .mat files.');
end

numFiles = min(length(files1), length(files2));
% 循环处理每一对文件
for i = 1:numFiles
    file1 = fullfile(folder1, files1(i).name);
    file2 = fullfile(folder2, files2(i).name);
    [~, name1, ext1] = fileparts(file1);
    [~, name2, ext2] = fileparts(file2);
    assert(strcmp(name1, name2) && strcmp(ext1, ext2), ...
        '文件名或扩展名在 folder1 和 folder2 中不匹配。');
    dataStruct1 = load(file1); 
    dataStruct2 = load(file2); 
    mergedStruct.data = dataStruct1.data; % 合并数据
    mergedStruct.label = dataStruct2.data; % 合并标签

    outputFile = fullfile(sprintf('%s\\audio-label', base_path), ['merged_' num2str(i) '.mat']);
    save(outputFile, 'mergedStruct'); % 保存文件
end

disp('All files have been processed.');