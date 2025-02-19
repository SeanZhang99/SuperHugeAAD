function [stimuli,fs] = load_stimuli(path,stimuli_path)
%LOAD_STIMULI Summary of this function goes here
%   Detailed explanation goes here
[~,~,extension] = fileparts(stimuli_path);
switch extension
    case {".mat"}
        s = load(fullfile(path,stimuli_path));
        stimuli = s.envelope;
        fs = s.Fs;
    case {".npy"}
        stimuli = npyread(fullfile(path,stimuli_path));
        fs = NaN;
    otherwise
        [stimuli,fs] = audioread(fullfile(path,stimuli_path));
end

