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
        case {"NJU_preprocessed"}
            if trial_idx <= length(data_struct.data.eeg)
                exg = data_struct.data.eeg{trial_idx};
                [~,label] = get_attention_directions(data_struct.expinfo,trial_idx);
                stimuli_path = table2array(data_struct.expinfo(trial_idx,2+(data_struct.expinfo.attended_lr(trial_idx)=="right")));
            end
        case "Alices"
            if trial_idx <= length(data_struct.data.eeg)
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
                label = string(data_struct.(field){trial_idx}.attended_ear);
                stimuli_path = string(data_struct.(field){trial_idx}.stimuli{1+(label=="R")});
                split_stimuli = split(stimuli_path,"_");
                env_path = sprintf("powerlaw subbands %s_dry.mat",join(split_stimuli(1:end-1),"_"));
            end
        case {"sparKULee_raw","sparKULee_preprocessed"}
            filelists = dir(fullfile(dataset_path,"sub-*","*","*.npy"));
            if trial_idx <= length(filelists)
                exg = py.numpy.load(fullfile(filelists(trial_idx).folder,filelists(trial_idx).name))';
            end
            target_audio = regexp(filelists(trial_idx).name,'.*?desc-preproc-audio-(.*?_\d_\d)_eeg.npy',"tokens");
            env_path = target_audio{1}+"_-_envelope.npy";
            mel_path = target_audio{1}+"_-_mel.npy";
        case "DTU_preprocessed"
            if trial_idx <= length(data_struct.data.eeg)
                exg = data_struct.data.eeg{trial_idx};
                if int32(data_struct.data.event.eeg(trial_idx).value{:})==1
                    label = 'left';
                else
                    label = 'right';
                end
                env = fastif(label==1,data_struct.data.wavA{trial_idx},data_struct.data.wavB{trial_idx});
            end
        case "MAD_EEG_preprocessed"
            if trial_idx <= length(data_struct.data.eeg)
                exg = data_struct.data.eeg{trial_idx};
                label = int32(data_struct.data.event.eeg(trial_idx).value{:});
                env = fastif(label==1,data_struct.data.wavA{trial_idx},data_struct.data.wavB{trial_idx});
            end
        case "PKU-4talker-EEG_preprocessed"
            data_struct.EEG_space.data=data_struct.EEG_space.data';
                exg = data_struct.EEG_space.data;
                label = int32(data_struct.EEG_space.event.latency);
                %env = fastif(label==1,data_struct.data.wavA{trial_idx},data_struct.data.wavB{trial_idx});
        case "Estart-2019_raw"
                exg = data_struct.group__fM.("part_"+num2str(trial_idx))(:);
                stimuli_path = "part_" + string(trial_idx) + "_story.wav";
        case "Data-for-CS_preprocessed"
                exg = data_struct.mergedStruct.data(trial_idx,:,:);
                label =  string(data_struct.mergedStruct.label(trial_idx,:,:));
        case "KUL-AV-GC_preprocessed"
                exg = data_struct.data{1,trial_idx};
                label =  string(data_struct.initAttention(1,trial_idx));
                env = fastif(label=="left",data_struct.stimulus.leftEnvelopes{1,trial_idx},data_struct.stimulus.rightEnvelopes{1,trial_idx});
        case "ASA_preprocessed"
                exg = data_struct.data{1,trial_idx};
                if data_struct.labels(trial_idx, 1) == 0
                    label = 'left';
                else
                    label = 'right';
                end
        otherwise
            error("Unimplemented dataset %s",dataset_name)
    end
    trial_infos(end).exg = exg;
    trial_infos(end).stimuli_path = stimuli_path;
    trial_infos(end).label = string(label);
    trial_infos(end).env_path = env_path;
    trial_infos(end).mel_path = mel_path;
    trial_infos(end).stimuli = stimuli;
    trial_infos(end).env = env;
    trial_infos(end).mel = mel;
    trial_infos(end).stimuli_fs = stimuli_fs;
    trial_infos(end+1) = struct("exg",[],"stimuli_path",[],"label",[],"env_path",[],"mel_path",[],"stimuli",[],"env",[],"mel",[],"stimuli_fs",[]);
end
end


