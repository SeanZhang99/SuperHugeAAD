clear; clc; close all;
cd("D:\Matlab\eeg_pipeline_files");
addpath("import\");
addpath("align\");
addpath("preprocessing\");
addpath("utils\");
addpath("erp\");

file_path = pwd;

%% 定义文件夹路径
source_paths = {'E:\EEG_dataset_Superhuge\Estart_2019\Processed_MAT\fW', 'E:\EEG_dataset_Superhuge\Estart_2019\Processed_MAT\fM'};
chanlocs_path = 'E:\EEG_dataset_Superhuge\Estart_2019\Estart_chanlocs.ced';  
target_path = 'E:\EEG_dataset_Superhuge\EStart_2019\Processed_MAT\preproc_ica';   

if ~exist(target_path, 'dir')
    mkdir(target_path);
end

folders = {'P00', 'P01', 'P02','P03','P04','P05','P06','P07','P08','P09','P10','P13','P14','P15','P16','P17','P18','P19'};
fs = 125;
refs = 128;

fir_order = 1024;
fir_wn = 2 * 1 / refs;
b_highpass = fir1(fir_order,fir_wn,"high");
iir_order = 8; 
iir_passband = [1 62]; % 带通滤波 (0.5Hz ~ 62Hz)
notch_stopband = [48, 52]; % 带阻滤波 (50Hz 工频噪声)
[b_band, a_band] = butter(iir_order, 2 * iir_passband / refs, "bandpass"); % 带通滤波
[b_notch, a_notch] = butter(iir_order, 2 * notch_stopband / refs, "stop"); % 工频滤波

for source_idx = 1:length(source_paths)
    source_path = source_paths{source_idx};
    suffix = '';
    if source_idx == 1
        suffix = 'fW';
    elseif source_idx == 2
        suffix = 'fM';
    end
    for subject = 1:length(folders)
        folder_name = folders{subject};
        mat_file = fullfile(source_path, sprintf('%s.mat', folder_name)); 
    
        if isfile(mat_file)
            load(mat_file, 'data'); 
            fprintf("Loaded EEG data for %s. Proceeding to preprocessing...\n", folder_name);
            segs = {};
            fprintf("Applying bandpass (%.1f - %.1f Hz) and notch filter (%.1f Hz)...\n", ...
                    iir_passband(1), iir_passband(2), mean(notch_stopband));
            exg = [];
            for tr = 1:length(data)
                exg_seg = data{tr};
                if size(exg_seg, 2) > size(exg_seg, 1)
                    exg_seg = exg_seg.';  
                end
                exg_seg = resample(exg_seg,refs,fs);
                exg_seg = fftfilt(b_highpass,exg_seg);
                exg_seg = filtfilt(b_band, a_band, exg_seg); 
                exg_seg = filtfilt(b_notch, a_notch, exg_seg);
                exg = [exg; exg_seg]; 
                segs{tr,1} = exg_seg;
            end
    
            % %% **STEP 4: 调用 remove_bad_channel 进行坏通道去除**
            fprintf("Removing bad channels for %s...\n", folder_name);
            [exg, segs] = remove_bad_channel(source_path, segs,fs=refs,subject=subject,save_segs=false,update_cell=true,chanlocs_filepath=chanlocs_path);
            %% REJECT noise data
            [exg,segs] = reject_noisy_data(source_path, segs,fs=refs,subject=subject,save_segs=false,update_cell=true,chanlocs_filepath=chanlocs_path);
            %% **STEP 5: 运行 ICA**
            fprintf("Running ICA for %s...\n", folder_name);
            [exg, segs] = run_ica('', exg, segs, 'fs', refs, 'subject', subject, ...
                                  'do_aggresive_asr', true, 'remove_comp_trialwise', false, "chanlocs_filepath", chanlocs_path);
    
            ica_save_path = fullfile(target_path, sprintf('%s_ica_%s.mat', folder_name, suffix));
            
            save(ica_save_path, 'segs'); 
    
            fprintf("ICA completed and saved for %s at %s\n", folder_name, ica_save_path);
        else
            fprintf("File not found: %s. Skipping...\n", mat_file);
        end
    end
end

fprintf("All ICA processing complete.\n");

