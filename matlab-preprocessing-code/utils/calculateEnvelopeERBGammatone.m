function [combinedEnvelope, subbandEnvelopes] = calculateEnvelopeERBGammatone(signal, fs, freq_range,numBands, p)
% calculateEnvelopeERBGammatone
% Computes subband envelopes and combined envelope using ERB-scaled Gammatone filters and power-law processing.
%
% INPUTS:
% - signal: Input signal (1D array)
% - fs: Sampling frequency (Hz)
% - numBands: Number of subbands
% - p: Power-law exponent
%
% OUTPUTS:
% - subbandEnvelopes: Matrix of subband envelopes (numBands x length(signal))
% - combinedEnvelope: Combined envelope (1D array)

    % Initialize variables
    subbandEnvelopes = zeros(length(signal),numBands);

    % Step 2: Filter the signal into subbands using Gammatone filter bank
        % Create a Gammatone filter for the current center frequency
    gFilter = gammatoneFilterBank(freq_range,numBands,fs);
    filteredSignal = gFilter(signal);

    % Step 3: Apply power-law transformation
    magnitude = abs(filteredSignal);
    % Apply power-law
    subbandEnvelopes = magnitude.^p;

    % Step 4: Combine envelopes
    combinedEnvelope = sum(subbandEnvelopes, 2);
end

function fc = erbScaleFrequencies(fs, numBands)
% erbScaleFrequencies
% Computes ERB-scaled center frequencies for a Gammatone filter bank.
%
% INPUTS:
% - fs: Sampling frequency (Hz)
% - numBands: Number of subbands
%
% OUTPUTS:
% - fc: ERB-scaled center frequencies (1D array)

    % Minimum and maximum frequencies based on the Nyquist limit
    fmin = 100;                      % Minimum frequency (Hz)
    fmax = fs / 2;                   % Maximum frequency (Nyquist limit)

    % ERB scale formula parameters
    EarQ = 9.26449;                  % Glasberg and Moore (1990)
    minBW = 24.7;                    % Minimum bandwidth
    erb = @(f) f / EarQ + minBW;     % ERB formula

    % Generate ERB-spaced frequencies
    erbMin = erb(fmin);
    erbMax = erb(fmax);
    erbPoints = linspace(erbMin, erbMax, numBands);
    fc = EarQ * (erbPoints - minBW); % Inverse ERB formula to get center frequencies
end
