function dataStruct = readH5Groups(fileName)
    % 读取HDF5文件中的所有组，并将它们保存在一个结构体中
    
    % 获取HDF5文件的所有信息
    fileInfo = h5info(fileName);
    
    % 初始化一个结构体来存储所有的组
    dataStruct = struct();
    
    % 遍历文件中的每个组
    for i = 1:length(fileInfo.Groups)
        groupName = fileInfo.Groups(i).Name;
        
        % 替换掉组名中的斜杠字符，确保它能作为字段名
        fieldName = strrep(groupName, '/', '_');
        
        % 如果字段名以数字或下划线开头，添加前缀
        if isstrprop(fieldName(1), 'digit') || fieldName(1) == '_'
            fieldName = ['group_' fieldName];
        end
        
        % 获取该组中的所有数据集
        datasets = fileInfo.Groups(i).Datasets;
        
        % 创建一个字段来存储该组的数据
        groupData = struct();
        
        for j = 1:length(datasets)
            datasetName = datasets(j).Name;
            
            % 读取数据集
            groupData.(datasetName) = h5read(fileName, [groupName '/' datasetName]);
        end
        
        % 将组数据添加到最终的结构体中
        dataStruct.(fieldName) = groupData;
    end
end


fileName = 'E:\EEG dataset\Estart_2019\eeg\P00.h5';
data = readH5Groups(fileName);
