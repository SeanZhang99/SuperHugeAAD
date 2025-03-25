function s = load_data_struct(dataset_path,dataset_name)
%LOAD_DATA_STRUCT Summary of this function goes here
%   Detailed explanation goes here
switch dataset_name
    case {"sparKULee_raw","sparKULee_preprocessed"}
        s = [];
    otherwise
        s = load(dataset_path);
end
end

