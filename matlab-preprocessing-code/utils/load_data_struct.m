function s = load_data_struct(dataset_path,dataset_name)
%LOAD_DATA_STRUCT Summary of this function goes here
%   Detailed explanation goes here
switch dataset_name
    case {"sparKULee_raw","sparKULee_preproc"}
        s = [];
    case "Estart-2019_raw"
    h5Files = dir(fullfile(dataset_path, '*.h5'));
    dataStruct = struct();
    for k = 1:length(h5Files)
        fileName = fullfile(dataset_path, h5Files(k).name);
        fileInfo = h5info(fileName);
        fileData = struct();
        for i = 1:length(fileInfo.Groups)
            groupName = fileInfo.Groups(i).Name;
            fieldName = strrep(groupName, '/', '_');
            if isstrprop(fieldName(1), 'digit') || fieldName(1) == '_'
                fieldName = ['group_' fieldName];
            end
            datasets = fileInfo.Groups(i).Datasets;
            groupData = struct();
            for j = 1:length(datasets)
                datasetName = datasets(j).Name;
                groupData.(datasetName) = h5read(fileName, [groupName '/' datasetName]);
            end
            fileData.(fieldName) = groupData;
        end
        dataStruct.(h5Files(k).name) = fileData;
    end
    otherwise
        s = load(dataset_path);
end
end

