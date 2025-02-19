clear;clc;close all;
addpath(".\utils\");
profile on

config;

metadata = py.dict;
dataset_infos = get_dataset_info(dataset_names);

for dataset_id = 1:length(dataset_names)
    dataset_name = dataset_names(dataset_id);
    dataset_info = dataset_infos(dataset_id);
    fs = dataset_info.fs;
    
    for subject_id = 1:fastif(DEBUG_MODE,1,dataset_info.num_subject)

        data_struct = load_data_struct(fullfile(dataset_info.filelists(subject_id).folder,dataset_info.filelists(subject_id).name),dataset_name);

        for trial_id = 1:fastif(DEBUG_MODE,1,dataset_info.num_trial)
            entry = sprintf("dataset-%03d-subject-%03d-trial-%03d",dataset_id,subject_id,trial_id);

            trial_info = extract_trials(data_struct,dataset_info.base_path,trial_id,dataset_name,[]);
            trial_info = trial_info(1);

            exg = double(trial_info.exg);
            label = trial_info.label;
            stimuli_path = trial_info.stimuli_path;
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
            if size(exg,2) > size(exg,1)
                % always time-chan.
                exg = exg';
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

            env_filename = "";
            if env_path ~= ""
                % load attended envelope if provided
                env_filename = split(env_path,".");
                env_filename = env_filename(1);
            elseif stimuli_is_available
                % infer envelope from stimuli signal
                env_filename = sprintf("%s_%s_env",dataset_name,stimuli_path);
            elseif ~isempty(env)
                % dataset provide envelope, unknown name
                env_filename = sprintf("%s_env",entry);
            end
            if env_filename~=""  && (ENVELOPE_OVERRIDE || ~exist(fullfile(stimuli_path,sprintf("%s.npy",env_path)),"file"))
                if isempty(env) && stimuli_is_available
                    [~,env] = calculateEnvelopeERBGammatone(stimuli,stimuli_fs,15,.3);
                    env = resample(env,fs,stimuli_fs);
                end
                if isempty(env) && env_path ~="";[env,~] = load_stimuli(dataset_info.audio_path,env_path);end
                env_path = fullfile(wav_path,"env",sprintf("%s.npy",env_filename));
                py.numpy.save(env_path,py.numpy.array(env));
            end

            mel_filename = "";
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
            if mel_filename~=""  && (STIMULI_OVERRIDE || ~exist(fullfile(stimuli_path,sprintf("%s.npy",mel_path)),"file"))
                if isempty(mel) && mel_path ~="";[mel,~] = load_stimuli(dataset_info.audio_path,mel_path);mel = py.numpy.array(mel);end
                if isempty(mel) && stimuli_is_available
                    mel = py.mel.calculate_mel_spectrogram(audio=py.numpy.array(stimuli),fs=py.int(stimuli_fs),target_fs=py.int(fs));
                end
                mel_path = fullfile(wav_path,"mel",sprintf("%s.npy",mel_filename));
                py.numpy.save(mel_path,mel);
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
            if string(label)~="";metadata{entry}{"label"}=label;end
            if stimuli_path~="";metadata{entry}{"stimuli_path"}=py.str(stimuli_path);metadata{entry}{"stimuli_fs"}=py.int(stimuli_fs);end
            if env_path~="";metadata{entry}{"env_path"}=py.str(env_path);metadata{entry}{"env_fs"}=py.int(fs);end
            if mel_path~="";metadata{entry}{"mel_path"}=py.str(mel_path);metadata{entry}{"mel_fs"}=py.int(fs);end
        end
    end
end

%% save metadata
pickle = py.importlib.import_module('pickle');
fid = py.open(fullfile(save_path,"meta","metadata.pkl"),"wb");
pickle.dump(metadata,fid);
fid.close;
