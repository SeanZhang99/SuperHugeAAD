h5_filename = 'E:\EEG_dataset_Superhuge\Estart_2019\english_session_data_preprocessed.h5'; 
save_root = 'E:\EEG_dataset_Superhuge\Estart_2019\Processed_MAT'; 

group_prefixes = {'/fM/part', '/fW/part'}; 
part_nums = 1:4;
dataset_names = {'P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', ...
                 'P10', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19'};
att_unatt_names = {'attended', 'unattended'};

for gp = 1:length(group_prefixes)
    group_prefix = group_prefixes{gp};
    [group_root, ~] = strtok(group_prefix, '/');  
    group_name = erase(group_root, '/');         
    group_folder = fullfile(save_root, group_name); 

    if ~exist(group_folder, 'dir')
        mkdir(group_folder);
    end
    data_map = containers.Map();
    env = struct();
    env.attended = {};
    env.unattended = {};

    for part_num = part_nums
        part_path = sprintf('%s%d/', group_prefix, part_num);  
        for i = 1:length(dataset_names)
            key = dataset_names{i};
            dataset_path = strcat(part_path, key);
            
                temp_data = h5read(h5_filename, dataset_path);
                if isKey(data_map, key)
                    current_cell = data_map(key);
                else
                    current_cell = {};
                end
                current_cell{end+1} = temp_data;
                data_map(key) = current_cell;
            
        end

        for i = 1:length(att_unatt_names)
            key = att_unatt_names{i};
            dataset_path = strcat(part_path, key);
           
                temp_data = h5read(h5_filename, dataset_path);
                env.(key){end+1} = temp_data;
           
        end
    end

    all_keys = keys(data_map);
    for i = 1:length(all_keys)
        key = all_keys{i};
        data = data_map(key);
        save_path = fullfile(group_folder, [key, '.mat']);
        save(save_path, 'data');
        fprintf('Saved: %s\n', save_path);
    end

    save(fullfile(group_folder, 'env.mat'), 'env');
    fprintf('Saved: %s\n', fullfile(group_folder, 'env.mat'));
end
