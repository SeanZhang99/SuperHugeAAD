function trial_infos = extract_trials(data_struct,dataset_path,trial_idxs,dataset_name,desired_length)
%EXTRACT_TRIALS Summary of this function goes here
%   Detailed explanation goes here
trial_infos = struct("exg",[],"stimuli_path",[],"label",[],"env_path",[],"mel_path",[],"stimuli",[],"env",[],"mel",[],"stimuli_fs",[]);
for trial_idx = trial_idxs
    exg = NaN;
    stimuli_path = "";
    label = "";
    env_path = "";
    mel_path = "";
    stimuli = [];
    env = [];
    mel = [];
    stimuli_fs = [];
    switch dataset_name
        case {"NJU_raw","NJU_preprocessed"}
            if trial_idx <= length(data_struct.data.eeg)
                exg = data_struct.data.eeg{trial_idx};
                [~,label] = get_attention_directions(data_struct.expinfo,trial_idx);
                % original label: string of number, convert to pure numeric
                stimuli_path = table2array(data_struct.expinfo(trial_idx,2+(data_struct.expinfo.attended_lr(trial_idx)=="right")));
            end
        case "Alices_raw"
            if trial_idx <= length(data_struct.data.exg)
                exg = data_struct.data.exg{trial_idx};
                stimuli_path = sprintf("DownTheRabbitHoleFinal_SoundFile%d.wav",trial_idx);
            end
        case {"KUL_raw","KUL_preprocessed"}
            if dataset_name == "KUL_raw"
                field = "trials";
            elseif dataset_name == "KUL_preprocessed"
                field = "preproc_trials";
            end
            if trial_idx <= length(data_struct.(field))
                exg = data_struct.(field){trial_idx}.RawData.EegData;
                % original label: "L" and "R", convert to "left" and
                % "right"
                label = string(data_struct.(field){trial_idx}.attended_ear);
                stimuli_path = string(data_struct.(field){trial_idx}.stimuli{1+(label=="R")});
                split_stimuli = split(stimuli_path,"_");
                env_path = sprintf("powerlaw subbands %s_dry.mat",join(split_stimuli(1:end-1),"_"));
                label = fastif(label=="L","left","right");
            end
        case {"sparKULee_raw","sparKULee_preprocessed"}
            filelists = dir(fullfile(dataset_path,"sub-*","*","*.npy"));
            if trial_idx <= length(filelists)
                exg = py.numpy.load(fullfile(filelists(trial_idx).folder,filelists(trial_idx).name))';
            end
            target_audio = regexp(filelists(trial_idx).name,'.*?desc-preproc-audio-(.*?_\d_?\d*)_eeg.npy',"tokens");
            env_path = target_audio{1}+"_-_envelope.npy";
            mel_path = target_audio{1}+"_-_mel.npy";
        case "DTU_preprocessed"
            if trial_idx <= length(data_struct.data.eeg)
                exg = data_struct.data.eeg{trial_idx};
                % original label: "1" and "2", convert to "left" and
                % "right"
                label = int32(data_struct.data.event.eeg(trial_idx).value{:});
                env = fastif(label==1,data_struct.data.wavA{trial_idx},data_struct.data.wavB{trial_idx});
                label = fastif(label==1,"left","right");
            end
        otherwise
            error("Unimplemented dataset %s",dataset_name)
    end
    trial_infos(end).exg = exg;
    trial_infos(end).stimuli_path = stimuli_path;
    trial_infos(end).label = label;
    trial_infos(end).env_path = env_path;
    trial_infos(end).mel_path = mel_path;
    trial_infos(end).stimuli = stimuli;
    trial_infos(end).env = env;
    trial_infos(end).mel = mel;
    trial_infos(end).stimuli_fs = stimuli_fs;
    trial_infos(end+1) = struct("exg",[],"stimuli_path",[],"label",[],"env_path",[],"mel_path",[],"stimuli",[],"env",[],"mel",[],"stimuli_fs",[]);
end
end


