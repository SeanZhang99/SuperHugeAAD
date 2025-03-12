EXG_OVERRIDE = 0;
STIMULI_OVERRIDE = 0;
ENVELOPE_OVERRIDE = 0;
MEL_SPECTRUM_OVERRIDE = 0;
DEBUG_MODE = 0;


save_path = "E:\SuperHuge\derivatives\";
exg_path = fullfile(save_path, "exg");
wav_path = fullfile(save_path, "stimuli");
compet_wav_path = fullfile(save_path, "compet_stimuli");
mkdir(fullfile(save_path, "meta"));
mkdir(exg_path);
mkdir(wav_path);
mkdir(compet_wav_path);
for feature_type = ["wav", "env", "mel"]
    mkdir(fullfile(wav_path, feature_type));
    mkdir(fullfile(compet_wav_path, feature_type)); 
end

%dataset_names = ["KUL-AV-GC_preprocessed"];
dataset_names = ["NJU_preprocessed","DTU_preprocessed","KUL_raw","sparKULee_raw","sparKULee_preprocessed","Alices_raw","Estart-2019_raw","Data-for-CS_preprocessed","KUL-AV-GC_preprocessed","ASA_preprocessed"];

py.sys.path().append(".\utils\")