function [exg,segs,ICA_weight,ICA_sphere] = run_ica(import_path,exg,segs,opts,ica_opts)
%RUN_ICA Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    import_path (1,1) string
    exg (:,:) double
    segs (:,:) cell
    opts.fs (1,1) double = 128
    opts.subject (1,1) double = 1
    opts.update_cell (1,1) logical = 1
    opts.save_segs (1,1) logical = 1
    opts.chanlocs_filepath (1,1) string = "scripts\\eeg_pipeline\\cEEGridChanLocsFull.ced"
    ica_opts.remove_comp_trialwise (1,1) logical = 0
    ica_opts.prefilter (1,1) logical = 1
    ica_opts.filter_order (1,1) double = 8
    ica_opts.locutoff (1,1) double = 2
    ica_opts.do_aggresive_asr (1,1) logical = 1
end
arguments (Output)
    exg (:,:) double
    segs (:,:) cell
    ICA_weight double
    ICA_sphere double
end

%% put all exg segments into a long array
clc;
nsegs = size(segs,1);
npts = size(segs{1},1);
nch = size(segs{1},2);
chanlocs_filepath = char(opts.chanlocs_filepath);
opts.source_priority = get_source_priority(opts.fs);

%% prefilter the EXG data to obtain better ICA
if ica_opts.prefilter
    [b,a] = butter(ica_opts.filter_order,ica_opts.locutoff*2/opts.fs,"high");
    ica_exg = zeros(size(exg));
    for seg_idx = 1:nsegs
        ica_exg((1:npts)+(seg_idx-1)*npts,:) = filtfilt(b,a,exg((1:npts)+(seg_idx-1)*npts,:));
    end
else
    ica_exg = exg;
end

%% performe an aggresive ASR to clean the data
if ica_opts.do_aggresive_asr
    exg_tmp = ica_exg;
    eeglab('nogui')
    EEG = pop_importdata(...
        'dataformat','array',...
        'nbchan',nch,...
        'data',exg_tmp',... % EEG prefer channel first
        'srate',opts.fs,...
        'pnts',0,...
        'xmin',0,...
        'chanlocs',chanlocs_filepath,...
        'ref','average');
    ALLEEG = EEG;
    assert(all(EEG.data==single(exg_tmp'),"all"),"BAD IMPORT BEHAVIOUR");
    EEG = pop_clean_rawdata(EEG, ...
        'FlatlineCriterion',5,...
        'ChannelCriterion',0.5,...
        'LineNoiseCriterion',4,...
        'BurstCriterion',10,...
        'BurstRejection','on',...
        'WindowCriterion', 0.2,...
        'MaxMen',4096);
    EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical'); % still, I want to keep the channel dimension consistent for ICA.
    ica_exg = EEG.data'; %EEGLAB prefer channel first. Permute to time first (conventional order).
end

%% run ICA
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
EEG = pop_importdata(...
    'dataformat','array',...
    'nbchan',nch,...
    'data',ica_exg',... % EEG prefer channel first
    'srate',opts.fs,...
    'pnts',0,...
    'xmin',0,...,
    'chanlocs',chanlocs_filepath,...
    'ref','average');
assert(all(EEG.data==single(ica_exg'),"all"),"BAD IMPORT BEHAVIOUR");
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
ICA_weight = EEG.icaweights;
ICA_sphere = EEG.icasphere;
clear("ALLEEG","EEG","CURRENTSET","ALLCOM");
% the ICA_weight and ICA_sphere are two key parameters to decompose EXG
% signals.

%% inspect and remove ICs
if ~ica_opts.remove_comp_trialwise
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
    EEG = pop_importdata(...
        'dataformat','array',...
        'nbchan',nch,...
        'data',exg',...
        'srate',opts.fs,...
        'pnts',0,...
        'xmin',0,...,
        'chanlocs',chanlocs_filepath,...
        'ref','average',...
        'icaweights',ICA_weight,...
        'icasphere',ICA_sphere);
    assert(all(EEG.data==single(exg'),"all"),"BAD IMPORT BEHAVIOUR");
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'gui','off');
    EEG.icaweights = ICA_weight;
    EEG.icasphere = ICA_sphere;
    EEG.chaninfo.nosedir = '+Y';
    assignin("base","EEG",EEG);
    EEG = pop_iclabel(EEG, 'default');
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG = pop_icflag(EEG, [NaN NaN;0.8 1;0.8 1;0.8 1;NaN NaN;NaN NaN;NaN NaN]);
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG = pop_subcomp(EEG, []);
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'gui','off');
    exg = EEG.data'; %EEGLAB prefer channel first. Permute to time first (conventional order).
else
    for i = 1:nsegs
        [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
        EEG = pop_importdata(...
            'dataformat','array',...
            'nbchan',nch,...
            'data',exg((1:npts)+(i-1)*npts,:)',...
            'srate',opts.fs,...
            'pnts',0,...
            'xmin',0,...,
            'chanlocs',chanlocs_filepath,...
            'ref','average',...
            'icaweights',ICA_weight,...
            'icasphere',ICA_sphere);
        assert(all(EEG.data==single(exg((1:npts)+(i-1)*npts,:)'),"all"),"BAD IMPORT BEHAVIOUR");
        [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'gui','off');
        EEG.icaweights = ICA_weight;
        EEG.icasphere = ICA_sphere;
        EEG.chaninfo.nosedir = '+Y';
        assignin("base","EEG",EEG);
        EEG = pop_iclabel(EEG, 'default');
        [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
        EEG = pop_icflag(EEG, [NaN NaN;0.8 1;0.8 1;0.8 1;NaN NaN;NaN NaN;NaN NaN]);
        [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
        EEG = pop_subcomp(EEG, []);
        [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'gui','off');
        exg((1:npts)+(i-1)*npts,:) = EEG.data'; %EEGLAB prefer channel first. Permute to time first (conventional order).
    end
end
%% update the segs cell if required
if opts.update_cell
    for i = 1:nsegs
        segs{i,1} = exg((1:npts)+(i-1)*npts,:);
    end
end

%% save the preprocessed segments
if opts.save_segs
    save_path = fullfile(import_path,"preproc3_exg",opts.source_priority);
    if ~isfolder(save_path)
        mkdir(save_path)
    end
    for i = 1:nsegs
        seg = exg((1:npts)+(i-1)*npts,:);
        parsave(fullfile(save_path,sprintf("S%02dT%02d_seg.mat",opts.subject,i)),seg);
    end
    parsave(fullfile(save_path,sprintf("S%02d_all.mat",opts.subject)),segs);
end
end

