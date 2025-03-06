EXG_OVERRIDE = 0;
STIMULI_OVERRIDE = 0;
ENVELOPE_OVERRIDE = 0;
MEL_SPECTRUM_OVERRIDE = 0;
DEBUG_MODE = 0;

save_basepath = 'E:\split_datasets';
%dataset_names = ["NJU_preprocessed","DTU_preprocessed","KUL_raw"];
dataset_names = ["NJU_preprocessed","DTU_preprocessed","KUL_raw","sparKULee_raw","sparKULee_preprocessed","Alices_raw","PKU-4talker-EEG_preprocessed","Estart-2019_raw","Data-for-CS_preprocessed","KUL-AV-GC_preprocessed","ASA_preprocessed"];

py.sys.path().append(".\utils\")