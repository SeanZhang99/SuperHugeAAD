base_path = "E:\EEG_dataset_Superhuge\ASA";
save_path = "E:\EEG_dataset_Superhuge\ASA\rescaled_data";
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
for subject = 1:20
    eeg = load(fullfile(base_path, sprintf("preprocessed_data_subject_%d.mat", subject)));
    
    rescaled_data = eeg; 
    scaling_factor = 1e5;

    for trial = 1:numel(eeg.data)  
        rescaled_data.data{trial} = eeg.data{trial} .* scaling_factor;
    end
    
    save_file = fullfile(save_path, sprintf("rescaled_data_subject_%d.mat", subject));
    
    save(save_file, '-struct', 'rescaled_data'); 
end