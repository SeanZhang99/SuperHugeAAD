clear;clc;close all;
addpath(".\utils\");

config;

dataset_infos = get_dataset_info(dataset_names);
metadata = py.dict;

for dataset_id = 1:length(dataset_names)
    dataset_name = dataset_names(dataset_id);
    dataset_info = dataset_infos(dataset_id);
    fs = dataset_info.fs;
    disp(['Processing dataset: ', dataset_name]);  % 打印当前处理的 dataset_name
    for subject_id = 1:fastif(DEBUG_MODE,4,dataset_info.num_subject)
        data_struct = load_data_struct(fullfile(dataset_info.filelists(subject_id).folder,dataset_info.filelists(subject_id).name),dataset_name);
        num_trial = get_num_trials(dataset_name, subject_id, dataset_info);
        for trial_id = 1:fastif(DEBUG_MODE,4,num_trial)
            entry = sprintf("dataset-%03d-subject-%03d-trial-%03d",dataset_id,subject_id,trial_id);
            trial_info = extract_trials(data_struct,dataset_info.base_path,trial_id,subject_id,dataset_name,[]);
            trial_info = trial_info(1);

            exg = trial_info.exg;
            label = trial_info.label;
            stimuli_path = trial_info.stimuli_path;
            compet_stimuli_path = trial_info.compet_stimuli_path;
            env_path = trial_info.env_path;
            compet_env_path = trial_info.compet_env_path;
            mel_path = trial_info.mel_path;
            stimuli = trial_info.stimuli;
            compet_stimuli = trial_info.compet_stimuli;
            stimuli_fs = trial_info.stimuli_fs;
            env = trial_info.env;
            compet_env = trial_info.compet_env;
            mel = trial_info.mel;

            if isnan(exg)
                % trial index exceed
                break
            end
            if EXG_OVERRIDE || ~exist(fullfile(exg_path,sprintf("%s.npy",entry)),"file")
                py.numpy.save(fullfile(exg_path,sprintf("%s.npy",entry)),py.numpy.array(exg));
            end

            % 处理 stimuli_path 和 compet_stimuli_path
            stimuli_types = {"stimuli", "compet_stimuli"};
            paths = {stimuli_path, compet_stimuli_path};
            stimuli_data = {[], []};
            stimuli_fs = [NaN, NaN];
            
            % 处理 stimuli 和 compet_stimuli
            for i = 1:length(stimuli_types)
                if paths{i} ~= "" 
                    [stimuli_data{i}, stimuli_fs(i)] = load_stimuli(dataset_info.audio_path, paths{i});
                    assert(~isnan(stimuli_fs(i)));
                    paths{i} = split(paths{i}, "."); 
                    paths{i} = paths{i}(1);
                elseif ~isempty(stimuli_data{i}) 
                    paths{i} = entry + "_" + stimuli_types{i} + "_wav";
                    assert(~isnan(stimuli_fs(i)));
                end
            end
            
            stimuli_path = paths{1};
            compet_stimuli_path = paths{2};
            
            % 保存 stimuli 和 compet_stimuli
            if stimuli_path ~= "" && (STIMULI_OVERRIDE || ~exist(fullfile(wav_path, "wav", sprintf("%s.npy", stimuli_path)), "file"))
                py.numpy.save(fullfile(wav_path, "wav", sprintf("%s.npy", stimuli_path)), ...
                              py.numpy.array(stimuli_data{1}));
            end
            
            if compet_stimuli_path ~= "" && (STIMULI_OVERRIDE || ~exist(fullfile(compet_wav_path, "wav", sprintf("%s.npy", compet_stimuli_path)), "file"))
                py.numpy.save(fullfile(compet_wav_path, "wav", sprintf("%s.npy", compet_stimuli_path)), ...
                              py.numpy.array(stimuli_data{2}));
            end

            stimuli_is_available = stimuli_path~=""&&~isempty(stimuli_data);

            env_filenames = {"", ""};
            
            for i = 1:length(stimuli_types)
                if i == 1
                    base_dir = fullfile(wav_path, "env");          
                else
                    base_dir = fullfile(compet_wav_path, "env");   
                end
            
                if env_path ~= ""
                    tmp = split(env_path, ".");
                    env_filenames{i} = tmp{1};
                elseif stimuli_is_available
                    env_filenames{i} = sprintf("%s_%s_env", dataset_name, paths{i});
                elseif ~isempty(env)
                    env_filenames{i} = sprintf("%s_%s_env", entry, stimuli_types{i});
                end
            
                if ~strcmp(env_filenames{i}, "") ...
                   && (ENVELOPE_OVERRIDE || ~exist(fullfile(base_dir, sprintf("%s.npy", env_filenames{i})), "file"))
                    
                    if isempty(env) && stimuli_is_available
                    [~,env] = calculateEnvelopeERBGammatone(stimuli_data{i},stimuli_fs(i),15,.3);
                    env = resample(env,fs,stimuli_fs(i));
                    end
            
                    if isempty(env) && env_path ~="";[env,~] = load_stimuli(dataset_info.audio_path,env_path);end
            
                    py.numpy.save(fullfile(base_dir, sprintf("%s.npy", env_filenames{i})), env);
                end
            end



           mel_filenames = {[], []};
            for i = 1:length(stimuli_types)
                if i == 1
                    base_dir = fullfile(wav_path, "mel");          
                else
                    base_dir = fullfile(compet_wav_path, "mel");   
                end
                
                if mel_path ~= ""
                    tmp = split(mel_path, ".");
                    mel_filenames{i} = tmp{1};
                elseif paths{i} ~= ""
                    mel_filenames{i} = sprintf("%s_%s_mel", dataset_name, paths{i});
                elseif ~isempty(mel)
                    mel_filenames{i} = sprintf("%s_%s_mel", entry, stimuli_types{i});
                end
               
                if ~strcmp(mel_filenames{i}, "") ...
                   && (STIMULI_OVERRIDE || ~exist(fullfile(base_dir, sprintf("%s.npy", mel_filenames{i})), "file"))
                    if ~strcmp(mel_path, "")
                        [mel, ~] = load_stimuli(dataset_info.audio_path, mel_path);
                        mel = py.numpy.array(mel);
                    end
                    
                    if isempty(mel) && ~isempty(stimuli_data{i})
                        mel = py.mel.calculate_mel_spectrogram(...
                            audio=py.numpy.array(stimuli_data{i}), ...
                            fs=py.int(stimuli_fs(i)), ...
                            target_fs=py.int(fs));
                    end

                    py.numpy.save(fullfile(base_dir, sprintf("%s.npy", mel_filenames{i})), mel);
                end

            end
            %prepare metadata
            entry = py.str(entry);
            metadata{entry} = py.dict;
            metadata{entry}{"dataset_id"} = py.int(dataset_id);
            metadata{entry}{"subject_id"} = py.int(subject_id);
            metadata{entry}{"trial_id"} = py.int(trial_id);
            metadata{entry}{"signal_length"} = py.len(exg);
            metadata{entry}{"num_channel"} = py.int(dataset_info.nch);
            metadata{entry}{"fs"} = py.int(fs);
            metadata{entry}{"dataset_name"} = py.str(dataset_name);
            metadata{entry}{"channel_infos"} = dataset_info.channel_infos;

            if label~="";metadata{entry}{"label"}=py.str(label);end
            if stimuli_path~="";metadata{entry}{"stimuli_path"}=py.str(stimuli_path);metadata{entry}{"stimuli_fs"}=py.int(stimuli_fs(1));end
            if compet_stimuli_path~="";metadata{entry}{"compet_stimuli_path"}=py.str(compet_stimuli_path);metadata{entry}{"stimuli_fs"}=py.int(stimuli_fs(2));end
            if env_path~="";metadata{entry}{"env_path"}=py.str(env_path);metadata{entry}{"env_fs"}=py.int(fs);end
            if compet_env_path~="";metadata{entry}{"compet_env_path"}=py.str(compet_env_path);metadata{entry}{"env_fs"}=py.int(fs);end
            if mel_path~="";metadata{entry}{"mel_path"}=py.str(mel_path);metadata{entry}{"mel_fs"}=py.int(fs);end
            clearvars -except data_struct dataset_name metadata dataset_infos dataset_info fs dataset_id subject_id trial_id dataset_names save_path save_basepath meta_path exg_path wav_path compet_wav_path EXG_OVERRIDE STIMULI_OVERRIDE ENVELOPE_OVERRIDE MEL_SPECTRUM_OVERRIDE DEBUG_MODE
        end
    end
end

%% save metadata
pickle = py.importlib.import_module('pickle');
with_open = py.open(fullfile(save_path, "meta","metadata.pkl"),"wb");
pickle.dump(metadata,with_open);
with_open.close;