function [competing_directions,attended_direction] = get_attention_directions(expinfo,trial)
%GET_AZIMUTH Summary of this function goes here
%   Detailed explanation goes here
competing_directions = expinfo.azimuth{trial};
while iscell(competing_directions)
    competing_directions = competing_directions{:};
end
competing_directions = sort(competing_directions, 'ascend');
attended_direction = competing_directions((expinfo.attended_lr(trial) == "right") + 1);
end

