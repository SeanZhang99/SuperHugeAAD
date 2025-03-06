clear;clc;close all;
addpath(".\utils\");

config;

dataset_infos = get_dataset_info(dataset_names);

for dataset_id = 1:length(dataset_names)
    dataset_name = dataset_names(dataset_id);
    dataset_info = dataset_infos(dataset_id);
    fs = dataset_info.fs;

    save_path = fullfile(save_basepath, dataset_name);  % 修改为你实际的存储目录
    if ~exist(save_path, 'dir')
        mkdir(save_path);  % 如果目录不存在，创建目录
    end
    exg_path = fullfile(save_path,"exg");
    wav_path = fullfile(save_path,"stimuli");
    mkdir(exg_path);
    mkdir(wav_path);
    meta_path = fullfile(save_path, "meta");
    if ~exist(meta_path, 'dir')
        mkdir(meta_path);  % 如果目录不存在，创建目录
    end
    for feature_type = ["wav", "env", "mel"]
        mkdir(fullfile(wav_path, feature_type));
    end
    metadata = py.dict;
    for subject_id = 1:fastif(DEBUG_MODE,1,dataset_info.num_subject)
        data_struct = load_data_struct(fullfile(dataset_info.filelists(subject_id).folder,dataset_info.filelists(subject_id).name),dataset_name);
        for trial_id = 1:fastif(DEBUG_MODE,1,dataset_info.num_trial)
            entry = sprintf("dataset-%03d-subject-%03d-trial-%03d",dataset_id,subject_id,trial_id);

            trial_info = extract_trials(data_struct,dataset_info.base_path,trial_id,dataset_name,[]);
            trial_info = trial_info(1);

            exg = trial_info.exg;
            label = trial_info.label;
            stimuli_path = trial_info.stimuli_path;
            compet_stimuli_path = trial.info.compet_stimuli_path;
            env_path = trial_info.env_path;
            mel_path = trial_info.mel_path;
            stimuli = trial_info.stimuli;
            stimuli_fs = trial_info.stimuli_fs;
            env = trial_info.env;
            mel = trial_info.mel;

            if isnan(exg)
                % trial index exceed
                break
            end
            if EXG_OVERRIDE || ~exist(fullfile(exg_path,sprintf("%s.npy",entry)),"file")
                py.numpy.save(fullfile(exg_path,sprintf("%s.npy",entry)),py.numpy.array(exg));
            end

            if stimuli_path ~= ""
                % load original audio stimuli if provided.
                [stimuli,stimuli_fs] = load_stimuli(dataset_info.audio_path,stimuli_path);
                assert(~isnan(stimuli_fs))
                stimuli_path = split(stimuli_path,".");
                stimuli_path = stimuli_path(1);
            elseif ~isempty(stimuli)
                stimuli_path = entry+"_wav";
                assert(~isempty(stimuli_fs));
            end
            if (stimuli_path~="") && (STIMULI_OVERRIDE || ~exist(fullfile(stimuli_path,sprintf("%s.npy",stimuli_path)),"file"))
                py.numpy.save(fullfile(wav_path,"wav",sprintf("%s.npy",stimuli_path)),py.numpy.array(stimuli));
            end
            stimuli_is_available = stimuli_path~=""&&~isempty(stimuli);

           
            env_filename = sprintf("%s_env",entry);
            py.numpy.save(fullfile(wav_path,"env",sprintf("%s.npy",env_filename)),py.numpy.array(env));
          

            if mel_path ~= ""
                % load attended mel spectrum if provided
                mel_filename = split(mel_path,".");
                mel_filename = mel_filename(1);
            elseif stimuli_path ~= ""
                % infer mel spectrum from stimuli signal
                mel_filename = sprintf("%s_%s_mel",dataset_name,stimuli_path);
            elseif ~isempty(mel)
                % dataset provide mel spectrum, unknown file name
                mel_filename = sprintf("%s_mel",entry);
            end
            if (mel_path~="" && mel_filename~="" ) && (STIMULI_OVERRIDE || ~exist(fullfile(stimuli_path,sprintf("%s.npy",mel_path)),"file"))
                if isempty(mel) && mel_path ~="";[mel,~] = load_stimuli(dataset_info.audio_path,mel_path);mel = py.numpy.array(mel);end
                if isempty(mel) && stimuli_is_available;mel = py.mel.calculate_mel_spectrogram(audio=py.numpy.array(stimuli),fs=py.int(stimuli_fs),target_fs=py.int(fs));end
                py.numpy.save(fullfile(wav_path,"mel",sprintf("%s.npy",mel_filename)),mel);
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
            if isfield(dataset_info(end), 'channel') && ~isempty(dataset_info(end).channel)
                metadata{entry}{"channel"} = py.str(strjoin(dataset_info(end).channel, ','));
            else
                metadata{entry}{"channel"} = py.str("");
            end

            if label~="";metadata{entry}{"label"}=py.str(label);end
            if stimuli_path~="";metadata{entry}{"stimuli_path"}=py.str(stimuli_path);metadata{entry}{"stimuli_fs"}=py.int(stimuli_fs);end
            if compet_stimuli_path~="";metadata{entry}{"compet_stimuli_path"}=py.str(compet_stimuli_path);metadata{entry}{"stimuli_fs"}=py.int(stimuli_fs);end
            if env_path~="";metadata{entry}{"env_path"}=py.str(env_path);metadata{entry}{"env_fs"}=py.int(fs);end
            if mel_path~="";metadata{entry}{"mel_path"}=py.str(mel);metadata{entry}{"mel_fs"}=py.int(fs);end
            clearvars -except data_struct dataset_name metadata dataset_infos dataset_info fs dataset_id subject_id trial_id dataset_names save_path exg_path wav_path EXG_OVERRIDE STIMULI_OVERRIDE ENVELOPE_OVERRIDE MEL_SPECTRUM_OVERRIDE DEBUG_MODE
        end
    end
end

%% save metadata
pickle = py.importlib.import_module('pickle');
with_open = py.open(fullfile(save_path,"meta","metadata.pkl"),"wb");
pickle.dump(metadata,with_open);
with_open.close;
