function dataset_infos = get_dataset_info(dataset_names)
arguments (Input)
    dataset_names (1,:) string
end
arguments (Output)
    dataset_infos (:,1) struct
end
dataset_infos = struct("filelists",[], ...
    "num_subject",[],"nch",[],"fs",[],"f_upper",[], ...
    "num_trial",[],"desired_length",[],"audio_path",[],"base_path",[],"channel_infos",[]);
for dataset_name = dataset_names
    type = split(dataset_name,"_");
    type = type(2);
    num_subject = [];
    switch dataset_name
        case {"NJU_preprocessed"}
            base_path = "D:\EEG dataset\NJU\filter_zscore_asr_ica_no_eye_inspect";
            audio_path = "D:\EEG dataset\NJU\stimuli";
            fs = 128;
            filelists = dir(fullfile(base_path,"S*.mat"));
            nch = 32;
            f_upper = 64;
            num_trial = 32;
            num_subject = 17;
            desired_length = 115 * fs;
            channel = ["Cz"; "Fz"; "Fp1"; "F7"; "F3"; "FC1"; "C3"; "FC5"; "FT9"; "T7"; "CP5";
                "CP1"; "P3"; "P7"; "PO9"; "O1"; "Pz"; "Oz"; "O2"; "PO10"; "P8"; "P4"; "CP2";
                "CP6"; "T8"; "FT10"; "FC6"; "C4"; "FC2"; "F4"; "F8"; "Fp2"];
        case {"KUL_raw","KUL_preprocessed"}
            base_path = "E:\EEG dataset\KUL";
            audio_path = "E:\EEG dataset\KUL\stimuli\stimuli";
            fs = 128;
            filelists = dir(fullfile(base_path,"S*.mat"));
            nch = 64;
            f_upper = 64;
            num_trial = 20;
            desired_length = 1.2e4;
            channel = [];
        case "Alices_raw"
            base_path = "E:\EEG_dataset_Superhuge\BrennansAliceStory\aligned_eeg";
            audio_path = "E:\EEG_dataset_Superhuge\BrennansAliceStory\audio\audio";
            fs = 500;
            filelists = dir(fullfile(base_path,"S*.mat"));
            nch = 62;
            f_upper = 200;
            num_trial = 12;
            desired_length = 26e3;
            channel = [];
        case {"sparKULee_raw","sparKULee_preprocessed"}
            base_path = sprintf("E:\\EEG_dataset_Superhuge\\sparrKULee\\sparrKULee\\derivatives\\%s_eeg",type);
            audio_path = "E:\EEG_dataset_Superhuge\sparrKULee\sparrKULee\derivatives\preprocessed_stimuli";
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
            num_trial = 0;
            desired_length = 10e4;
            num_subject = length(dir(fullfile(base_path,"sub-*")));
            channel = [];
        case "DTU_preprocessed"
            base_path = "E:\EEG dataset\DTU\DATA_preproc";
            audio_path = "";
            fs = 64;
            filelists = dir(fullfile(base_path,"S*.mat"));
            nch = 64;
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
        case "MAD-EEG_preprocessed"
            base_path = "E:\\EEG dataset\\MAD-EEG\\madeeg_preprocessed.hdf5"
            audio_path = "E:\\EEG dataset\\MAD-EEG\\stimuli\\stimuli";
            fs = 256;
            filelists = [1];
            nch = 20;
            f_upper = 128;
            desired_length = 5052;
            num_trial = 22;
            num_subject = 16;
            dataset_infos.filelists = filelists;
        case "PKU-4talker-EEG_preprocessed"
            base_path = "E:\EEG dataset\PKU-4talker-EEG\preprocess_data\data_space";
            audio_path = "";
            fs = 128;
            filelists = dir(fullfile(base_path, "sub*",'*cap.mat'));
            nch = 64;
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
                "PO3"; "PO4"; "PO5"; "PO6"; "PO7"; "PO8"; "Oz"; "O1"; "O2"; "ECG"; 
                "HEOR"; "HEOL"; "VEOU"; "VEOL"
            ];
        case "Estart-2019_raw"
            base_path = "E:\EEG_dataset_Superhuge\Estart_2019\eeg";
            audio_path = "E:\EEG_dataset_Superhuge\Estart_2019\audiobooks\audiobooks\fM";
            fs = 1000;
            filelists = dir(fullfile(base_path,"*.h5"));
            nch = 63;
            f_upper = 128;
            desired_length = 140000;
            num_trial = 4;
            num_subject = 18;
            channel = [];
        case "Data-for-CS_preprocessed"
            base_path = "E:\EEG dataset\Data_for_CS 安徽大学\Data_for_CS\data_for_CS\audio_label";
            audio_path = "";
            fs = 128;
            filelists = dir(fullfile(base_path,"*.mat"));
            nch = 32;
            f_upper = 64;
            num_trial = 4352;
            num_subject = 8;
            desired_length = 128;
        case "KUL-AV-GC_preprocessed"
            base_path = "E:\EEG dataset\KUL_AV_GC";
            audio_path = "E:\EEG dataset\KUL_AV_GC";
            fs = 128;
            filelists = dir(fullfile(base_path,"*.mat"));
            nch = 68;
            f_upper = 64;
            num_trial = 6;
            desired_length = 76800;
            channel = ["Fp1"; "AF7"; "AF3"; "F1"; "F3"; "F5"; "F7"; "FT7"; "FC5"; "FC3";
                "FC1"; "C1"; "C3"; "C5"; "T7"; "TP7"; "CP5"; "CP3"; "CP1"; "P1"; "P3"; 
                "P5"; "P7"; "P9"; "PO7"; "PO3"; "O1"; "Iz"; "Oz"; "POz"; "Pz"; "CPz"; "Fpz"; 
                "Fp2"; "AF8"; "AF4"; "AFz"; "Fz"; "F2"; "F4"; "F6"; "F8"; "FT8"; "FC6"; "FC4";
                "FC2"; "FCz"; "Cz"; "C2"; "C4"; "C6"; "T8"; "TP8"; "CP6"; "CP4"; "CP2"; "P2"; 
                "P4"; "P6"; "P8"; "P10"; "PO8"; "PO4"; "O2"; "EXG3"; "EXG4"; "EXG5"; "EXG6"];
        case "ASA_preprocessed"
            base_path = "E:\EEG dataset\ASA";
            audio_path = "";
            fs = 128;
            filelists = dir(fullfile(base_path,"*.mat"));
            nch = 64;
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
    dataset_infos(end).nch = nch;
    dataset_infos(end).num_trial = num_trial;
    dataset_infos(end).num_subject = fastif(isempty(num_subject),nfile,num_subject);
    dataset_infos(end).audio_path = audio_path;
    dataset_infos(end).base_path = base_path;
    dataset_infos(end).channel_infos = py_dict;
    dataset_infos(end+1) = struct("filelists",[], ...
    "num_subject",[],"nch",[],"fs",[],"f_upper",[], ...
    "num_trial",[],"desired_length",[],"audio_path",[],"base_path",[],"channel_infos",[]);
end