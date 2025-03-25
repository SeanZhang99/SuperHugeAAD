function [exg,segs] = remove_bad_channel(import_path,segs,opts)
arguments (Input)
    import_path (1,1) string
    segs (:,:) cell
    opts.update_cell (1,1) logical = 1
    opts.save_segs (1,1) logical = 1
    opts.subject (1,1) double = 1
    opts.chanlocs_filepath (1,1) string = "AHU_chanlocs.ced"
    opts.fs (1,1) double = 250
end
arguments (Output)
    exg (:,:) double
    segs (:,:) cell
end

clc;

%% puts all exg segments into a long array.
nsegs = size(segs,1);
npts_min = min(cellfun(@(x) size(x,1), segs));
nch = size(segs{1},2);
exg = zeros(nsegs * npts_min, nch);
chanlocs_filepath = char(opts.chanlocs_filepath);
opts.source_priority = get_source_priority(opts.fs);

%% normalize exg
for i = 1:nsegs
    exg_seg = segs{i,1}';   
    if size(exg_seg,2) > size(exg_seg,1)
        exg_seg = exg_seg.';
    end
    exg_seg_truncated = exg_seg(1:npts_min, :);
    exg((1:npts_min) + (i - 1) * npts_min, :) = exg_seg_truncated;
end
trimmed_mean = trimmean(exg.^2,20,1);
exg_scaling_factor = sqrt(median(trimmed_mean));
exg = exg ./ exg_scaling_factor;

%% remove bad channels with ASR
% for i = 1:nsegs
% exg_seg = exg((1:npts)+(i-1)*npts,:)';
exg_seg = exg';
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
EEG = pop_importdata(...
    'dataformat','array',...
    'nbchan',nch,...
    'data',exg_seg,...
    'srate',opts.fs,...
    'pnts',0,...
    'xmin',0,...,
    'chanlocs',chanlocs_filepath,...
    'ref','average');
assert(all(EEG.data==single(exg_seg),"all"),"BAD IMPORT BEHAVIOUR");
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'gui','off');

EEG = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion',2,...
    'ChannelCriterion',0.4,...
    'LineNoiseCriterion','off',...
    'BurstCriterion','off',...
    'WindowCriterion','off',...
    'BurstRejection','off',...
    'MaxMen',4000);
EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'gui','off');

assert(size(EEG.data,1)==size(exg_seg,1),"Channel number does not match")

exg_seg = EEG.data;
% exg((1:npts)+(i-1)*npts,:) = exg_seg';
exg = exg_seg';
% end
clc;
% EEGLAB prefer channel first
% also re-reference to grand average
exg = exg - mean(exg,2);

%% update the segs cell if required
if opts.update_cell
    for i = 1:nsegs
        segs{i,1} = exg((1:npts_min)+(i-1)*npts_min,:);
    end
end

%% save the preprocessed segments
if opts.save_segs
    save_path = fullfile(import_path,"preproc2_exg",opts.source_priority);
    if ~isfolder(save_path)
        mkdir(save_path)
    end
    for i = 1:nsegs
        if ~isempty(segs{i,2})
            npts1 = size(segs{i,2},1);
            seg1 = exg((1:npts1)+(i-1)*npts,:);
            seg2 = exg((npts1+1:npts)+(i-1)*npts,:);
            parsave(fullfile(save_path,sprintf("S%02dT%02d_seg1.mat",opts.subject,i)),seg1);
            parsave(fullfile(save_path,sprintf("S%02dT%02d_seg2.mat",opts.subject,i)),seg2);
        end
        seg = exg((1:npts)+(i-1)*npts,:);
        parsave(fullfile(save_path,sprintf("S%02dT%02d_seg.mat",opts.subject,i)),seg);
    end
    parsave(fullfile(save_path,sprintf("S%02d_all.mat",opts.subject)),segs);
end



