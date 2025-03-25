function silent_idx = detect_silent_tailing(signal, threshold, fs)
% DETECT_SILENT_TAIL Detects trailing silent segments in a multi-feature signal
% signal: (T by ...) multidimensional matrix, where T is the number of time steps
% threshold: scalar value indicating the silence threshold
% fs: sampling rate in Hz
% silent_idx: (T by 1) binary vector, where 1 indicates silent time steps

T = size(signal, 1); % Extract time dimension size

% Create a mask where any feature is below the threshold
silent_mask = abs(signal) < threshold;

% Collapse all dimensions except T to check if any feature is silent
silent_at_timestep = squeeze(any(silent_mask, 2:ndims(signal)));

% Find the start of the trailing silent segment
silent_idx = zeros(T, 1);
if any(silent_at_timestep)
    % Find the first fully silent index from the end
    first_silent_idx = find(~silent_at_timestep, 1, 'last') + 1;
    
    if ~isempty(first_silent_idx) && first_silent_idx <= T
        % Compute duration of silent segment
        silent_duration = T - first_silent_idx + 1;
        silent_time = silent_duration / fs;
        
        % Only mark as silent if duration is at least 1 second
        if silent_time >= 1
            silent_idx(first_silent_idx:end) = 1;
        end
    end
end

end