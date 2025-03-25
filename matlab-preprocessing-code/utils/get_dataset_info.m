function dataset_infos = get_dataset_info(dataset_names,raw_path)
arguments (Input)
    dataset_names (1,:) string
    raw_path (1,1) string
end
arguments (Output)
    dataset_infos (:,1) struct
end
empty_struct = struct("filelists",[], ...
    "num_subject",[],"nch",[],"channel_indices",[],"fs",[],"f_upper",[], ...
    "num_trial",[],"desired_length",[],"audio_path",[],"base_path",[],"channel_infos",[]);
dataset_infos = empty_struct;
for dataset_name = dataset_names
    type = split(dataset_name,"_");
    type = type(2);
    num_subject = [];
    switch dataset_name
        case {"NJU_raw","NJU_preprocessed"}
            base_path = fullfile(raw_path,"NJU-15class-Emotiv-AAD","exg",type);
            audio_path = fullfile(raw_path,"NJU-15class-Emotiv-AAD","stimuli");
            fs = 128;
            filelists = dir(fullfile(base_path,"S*.mat"));
            channel_indices = 1:32;
            f_upper = 64;
            num_trial = 32;
            num_subject = 21;
            desired_length = 115 * fs;
            channel = ["Cz"; "Fz"; "Fp1"; "F7"; "F3"; "FC1"; "C3"; "FC5"; "FT9"; "T7"; "CP5";
                "CP1"; "P3"; "P7"; "PO9"; "O1"; "Pz"; "Oz"; "O2"; "PO10"; "P8"; "P4"; "CP2";
                "CP6"; "T8"; "FT10"; "FC6"; "C4"; "FC2"; "F4"; "F8"; "Fp2"];
        case {"KUL_raw","KUL_preprocessed"}
            base_path = fullfile(raw_path,"KUL","exg",type);
            audio_path = fullfile(raw_path,"KUL","stimuli");
            fs = 128;
            filelists = dir(fullfile(base_path,"S*.mat"));
            channel_indices = 1:64;
            f_upper = 64;
            num_trial = 20;
            desired_length = 1.2e4;
            channel = ["Fp1";"AF7";"AF3";"F1";"F3";"F5";"F7";"FT7";"FC5";
                "FC3";"FC1";"C1";"C3";"C5";"T7";"TP7";"CP5";"CP3";"CP1";
                "P1";"P3";"P5";"P7";"P9";"PO7";"PO3";"O1";"Iz";"Oz";"POz";
                "Pz";"CPz";"Fpz";"Fp2";"AF8";"AF4";"AFz";"Fz";"F2";"F4";
                "F6";"F8";"FT8";"FC6";"FC4";"FC2";"FCz";"Cz";"C2";"C4";
                "C6";"T8";"TP8";"CP6";"CP4";"CP2";"P2";"P4";"P6";"P8";
                "P10";"PO8";"PO4";"O2";];
        case "Alices_raw"
            base_path = fullfile(raw_path,"BrennansAliceStory","aligned_eeg");
            audio_path = fullfile(raw_path,"BrennansAliceStory","audio","audio");
            fs = 500;
            filelists = dir(fullfile(base_path,"S*.mat"));
            channel_indices = 1:61;
            f_upper = 200;
            num_trial = 12;
            desired_length = 26e3;
            channel = [];
        case {"sparKULee_raw","sparKULee_preprocessed"}
            base_path = sprintf("%s\\sparrKULee\\derivatives\\%s_eeg",raw_path,type);
            audio_path = sprintf("%s\\sparrKULee\\derivatives\\preprocessed_stimuli",raw_path);
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
            channel_indices = 1:64;
            f_upper = 64;
            num_trial = 0;
            desired_length = 10e4;
            num_subject = length(dir(fullfile(base_path,"sub-*")));
            channel = ["Fp1";"AF7";"AF3";"F1";"F3";"F5";"F7";"FT7";"FC5";
                "FC3";"FC1";"C1";"C3";"C5";"T7";"TP7";"CP5";"CP3";"CP1";
                "P1";"P3";"P5";"P7";"P9";"PO7";"PO3";"O1";"Iz";"Oz";"POz";
                "Pz";"CPz";"Fpz";"Fp2";"AF8";"AF4";"AFz";"Fz";"F2";"F4";
                "F6";"F8";"FT8";"FC6";"FC4";"FC2";"FCz";"Cz";"C2";"C4";
                "C6";"T8";"TP8";"CP6";"CP4";"CP2";"P2";"P4";"P6";"P8";
                "P10";"PO8";"PO4";"O2"];
        case "DTU_preprocessed"
            base_path = fullfile(raw_path,"DTU","preprocessed");
            audio_path = "";
            fs = 64;
            filelists = dir(fullfile(base_path,"S*.mat"));
            channel_indices = 1:64;
            f_upper = 32;
            desired_length = 3200;
            num_trial = 60;
            channel = ["Fp1"; "AF7"; "AF3"; "F1"; "F3"; "F5"; "F7"; "FT7"; "FC5"; "FC3"; 
                "FC1"; "C1"; "C3"; "C5"; "T7"; "TP7"; "CP5"; "CP3"; "CP1"; "P1"; "P3"; "P5";
                "P7"; "P9"; "PO7"; "PO3"; "O1"; "Iz"; "Oz"; "POz"; "Pz"; "CPz"; "Fpz"; "Fp2";
                "AF8"; "AF4"; "AFz"; "Fz"; "F2"; "F4"; "F6"; "F8"; "FT8"; "FC6"; "FC4"; "FC2";
                "FCz"; "Cz"; "C2"; "C4"; "C6"; "T8"; "TP8"; "CP6"; "CP4"; "CP2"; "P2"; "P4"; 
                "P6"; "P8"; "P10"; "PO8"; "PO4"; "O2";
            ];
        case "PKU_preprocessed"
            base_path = fullfile(raw_path,"PKU-4talker-EEG","preprocess_data","data_space");
            audio_path = "";
            fs = 128;
            filelists = dir(fullfile(base_path, "sub*",'*cap.mat'));
            channel_indices = 1:59;
            f_upper = 64;
            num_trial = 40;
            num_subject = 16;
            desired_length = 7680;
            channel = [
                "Fpz"; "Fp1"; "Fp2"; "AF3"; "AF4"; "AF7"; "AF8"; "Fz"; "F1"; "F2"; 
                "F3"; "F4"; "F5"; "F6"; "F7"; "F8"; "FCz"; "FC1"; "FC2"; "FC3"; 
                "FC4"; "FC5"; "FC6"; "FT7"; "FT8"; "Cz"; "C1"; "C2"; "C3"; "C4"; 
                "C5"; "C6"; "T7"; "T8"; "CP1"; "CP2"; "CP3"; "CP4"; "CP5"; "CP6"; 
                "TP7"; "TP8"; "Pz"; "P3"; "P4"; "P5"; "P6"; "P7"; "P8"; "POz"; 
                "PO3"; "PO4"; "PO5"; "PO6"; "PO7"; "PO8"; "Oz"; "O1"; "O2"; 
            ];
        case "Estart_preprocessed"
            base_path = fullfile(raw_path,"Estart","preproc_ica");
            audio_path = "";
            fs = 128;
            filelists = dir(fullfile(base_path,"*.mat"));
            channel_indices = 1:63;
            f_upper = 64;
            desired_length = 17821;
            num_trial = 4;
            num_subject = 36;
            channel = ["AF3"; "AF4"; "AF7"; "AF8"; "C1"; "C2"; "C3"; "C4"; "C5"; "C6";
                "CP1"; "CP2"; "CP3"; "CP4"; "CP5"; "CP6"; "CPz"; "Cz"; "F1"; "F2"; "F3";
                "F4"; "F5"; "F6"; "F7"; "F8"; "FC1"; "FC2"; "FC3"; "FC4"; "FC5"; "FC6";
                "Fp1"; "Fp2"; "FT10"; "FT7"; "FT8"; "FT9"; "Fz"; "O1"; "O2"; "Oz"; "P1"; 
                "P2"; "P3"; "P4"; "P5"; "P6"; "P7"; "P8"; "PO3"; "PO7"; "PO8"; "POz"; "Pz"; 
                "FCz"; "T7"; "T8"; "TP10"; "TP7"; "TP8"; "TP9"; "AFz"];
        % case "AHU_preprocessed"
        %     base_path = fullfile(raw_path,"AHU","preprocessed");
        %     audio_path = "";
        %     fs = 128;
        %     filelists = dir(fullfile(base_path,"*.mat"));
        %     channel_indices = 1:32;
        %     f_upper = 64;
        %     num_trial = 4352;
        %     num_subject = 8;
        %     desired_length = 128;
        %     channel = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", ...
        %           "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", ...
        %           "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "A1", "T5", ...
        %           "P3", "PZ", "P4", "T6", "A2", "O1", "OZ", "O2"];
        case "KUL-AV-GC_preprocessed"
            base_path = fullfile(raw_path,"KUL_AV_GC");
            audio_path = base_path;
            fs = 128;
            filelists = dir(fullfile(base_path,"*.mat"));
            channel_indices = 1:64;
            f_upper = 64;
            num_trial = 6;
            desired_length = 76800;
            channel = ["Fp1"; "AF7"; "AF3"; "F1"; "F3"; "F5"; "F7"; "FT7"; "FC5"; "FC3";
                "FC1"; "C1"; "C3"; "C5"; "T7"; "TP7"; "CP5"; "CP3"; "CP1"; "P1"; "P3"; 
                "P5"; "P7"; "P9"; "PO7"; "PO3"; "O1"; "Iz"; "Oz"; "POz"; "Pz"; "CPz"; "Fpz"; 
                "Fp2"; "AF8"; "AF4"; "AFz"; "Fz"; "F2"; "F4"; "F6"; "F8"; "FT8"; "FC6"; "FC4";
                "FC2"; "FCz"; "Cz"; "C2"; "C4"; "C6"; "T8"; "TP8"; "CP6"; "CP4"; "CP2"; "P2"; 
                "P4"; "P6"; "P8"; "P10"; "PO8"; "PO4"; "O2"];
        case "NUS_preprocessed"
            base_path = fullfile(raw_path,"NUS_ASA");
            audio_path = "";
            fs = 128;
            filelists = dir(fullfile(base_path,"derivatives","*.mat"));
            channel_indices = 1:64;
            f_upper = 64;
            num_trial = 20;
            desired_length = 7299;
            channel = [
                "Fp1"; "Fp2"; "F3"; "F4"; "C3"; "C4"; "P3"; "P4"; "O1"; "O2"; 
                "F7"; "F8"; "T7"; "T8"; "P7"; "P8"; "Fz"; "Cz"; "Pz"; "Oz"; 
                "FC1"; "FC2"; "CP1"; "CP2"; "FC5"; "FC6"; "CP5"; "CP6"; "TP9"; "TP10"; 
                "POz"; "F1"; "F2"; "C1"; "C2"; "P1"; "P2"; "AF3"; "AF4"; "FC3"; "FC4"; 
                "CP3"; "CP4"; "PO3"; "PO4"; "F5"; "F6"; "C5"; "C6"; "P5"; "P6"; 
                "AF7"; "AF8"; "FT7"; "FT8"; "TP7"; "TP8"; "PO7"; "PO8"; "FT9"; "FT10"; 
                "Fpz"; "CPz"; "FCz"
            ];

        otherwise
            error("Unimplemented dataset %s",dataset_name)
    end
    py_dict = py.dict();
    for i = 1:length(channel)
        channel_info = py.dict();
        channel_info{'name'} = channel(i);
        py_dict{int32(i)} = channel_info;
    end
    nfile = length(filelists);
    dataset_infos(end).desired_length = desired_length;
    dataset_infos(end).f_upper = f_upper;
    dataset_infos(end).filelists = filelists;
    dataset_infos(end).fs = fs;
    dataset_infos(end).channel_indices = channel_indices;
    dataset_infos(end).nch = numel(channel_indices);
    dataset_infos(end).num_trial = num_trial;
    dataset_infos(end).num_subject = fastif(isempty(num_subject),nfile,num_subject);
    dataset_infos(end).audio_path = audio_path;
    dataset_infos(end).base_path = base_path;
    dataset_infos(end).channel_infos = py_dict;
    dataset_infos(end+1) = empty_struct;
end