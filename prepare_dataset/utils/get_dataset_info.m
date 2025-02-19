function dataset_infos = get_dataset_info(dataset_names)
arguments (Input)
    dataset_names (1,:) string
end
arguments (Output)
    dataset_infos (:,1) struct
end
dataset_infos = struct("filelists",[], ...
    "num_subject",[],"nch",[],"fs",[],"f_upper",[], ...
    "num_trial",[],"desired_length",[],"audio_path",[],"base_path",[]);
for dataset_name = dataset_names
    type = split(dataset_name,"_");
    type = type(2);
    n_subject = [];
    switch dataset_name
        case {"NJU_raw","NJU_preprocessed"}
            base_path = sprintf("E:\\EEG_RAW_DATA\\NJU-15class-Emotiv-AAD\\exg\\%s",type);
            audio_path = "E:\EEG_RAW_DATA\NJU-15class-Emotiv-AAD\stimuli";
            fs = 128;
            filelists = dir(fullfile(base_path,"S*.mat"));
            nch = 32;
            f_upper = 64;
            num_trial = 32;
            desired_length = 115 * fs;
        case {"KUL_raw","KUL_preprocessed"}
            base_path = sprintf("E:\\EEG_RAW_DATA\\KUL\\exg\\%s",type);
            audio_path = "E:\EEG_RAW_DATA\KUL\stimuli";
            fs = 128;
            filelists = dir(fullfile(base_path,"S*.mat"));
            nch = 64;
            f_upper = 64;
            num_trial = 20;
            desired_length = 1.2e4;
        case "Alices_raw"
            base_path = "E:\EEG_RAW_DATA\BrennansAliceStory\aligned_eeg";
            audio_path = "E:\EEG_RAW_DATA\BrennansAliceStory\audio\audio";
            fs = 500;
            filelists = dir(fullfile(base_path,"S*.mat"));
            nch = 60;
            f_upper = 200;
            num_trial = 12;
            desired_length = 28e3;
        case {"sparKULee_raw","sparKULee_preprocessed"}
            base_path = sprintf("E:\\EEG_RAW_DATA\\sparrKULee\\derivatives\\%s_eeg",type);
            audio_path = "E:\EEG_RAW_DATA\sparrKULee\derivatives\preprocessed_stimuli";
            fs = 128;
            tmp_list = dir(fullfile(base_path,"sub-*","ses*"));
            f = fieldnames(tmp_list)';
            f{2,1} = {};
            filelists = struct(f{:});
            for ff = 1:length(tmp_list)
                if (tmp_list(ff).name ~= ".") && (tmp_list(ff).name ~= "..")
                    filelists(end+1) = tmp_list(ff);
                end
            end
            nch = 64;
            f_upper = 64;
            num_trial = 10;
            desired_length = 10e4;
            n_subject = length(dir(fullfile(base_path,"sub-*")));
        case "sparKULee_preproc"
            base_path = "E:\EEG_RAW_DATA\sparrKULee\derivatives\preprocessed_eeg";
            audio_path = "E:\EEG_RAW_DATA\sparrKULee\derivatives\preprocessed_stimuli";
            fs = 128;
            tmp_list = dir(fullfile(base_path,"sub-*","ses*"));
            f = fieldnames(tmp_list)';
            f{2,1} = {};
            filelists = struct(f{:});
            for ff = 1:length(tmp_list)
                if (tmp_list(ff).name ~= ".") && (tmp_list(ff).name ~= "..")
                    filelists(end+1) = tmp_list(ff);
                end
            end
            nch = 64;
            f_upper = 64;
            num_trial = 10;
            desired_length = 10e4;
        case "DTU_preprocessed"
            base_path = sprintf("E:\\EEG_RAW_DATA\\DTU\\exg\\%s",type);
            audio_path = "";
            fs = 64;
            filelists = dir(fullfile(base_path,"S*.mat"));
            nch = 64;
            f_upper = 32;
            desired_length = 3200;
            num_trial = 60;
        otherwise
            error("Unimplemented dataset %s",dataset_name)
    end
    nfile = length(filelists);
    dataset_infos(end).desired_length = desired_length;
    dataset_infos(end).f_upper = f_upper;
    dataset_infos(end).filelists = filelists;
    dataset_infos(end).fs = fs;
    dataset_infos(end).nch = nch;
    dataset_infos(end).num_trial = num_trial;
    dataset_infos(end).num_subject = fastif(isempty(n_subject),nfile,n_subject);
    dataset_infos(end).audio_path = audio_path;
    dataset_infos(end).base_path = base_path;
    dataset_infos(end+1) = struct("filelists",[], ...
    "num_subject",[],"nch",[],"fs",[],"f_upper",[], ...
    "num_trial",[],"desired_length",[],"audio_path",[],"base_path",[]);
end