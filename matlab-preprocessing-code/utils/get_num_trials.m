function num_trial = get_num_trials(dataset_name, subject_id, dataset_info)
    switch dataset_name
        case {"sparKULee_raw", "sparKULee_preprocessed"}
            subject_id_str = sprintf('%03d', subject_id);
            num_trial = length(dir(fullfile(dataset_info.base_path, ['sub-' subject_id_str], '*', '*.npy')));
        otherwise
            num_trial = dataset_info.num_trial;
    end
end
