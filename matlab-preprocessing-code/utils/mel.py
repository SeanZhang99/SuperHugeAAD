import math
import librosa
import numpy as np

def calculate_mel_spectrogram(
    audio,
    fs,
    target_fs=64,
    fmin=0,
    fmax=5000,
    nb_filters=10,
    hop_length=None,
    win_length=None,
):
    """Calculates mel spectrogram with consistent output length."""
    
    # Compute hop_length correctly
    if not hop_length:
        hop_length = round(fs / target_fs)  # Ensures integer division
    if not win_length:
        win_length = int(0.025 * fs)  # 25 milliseconds

    # Finds the closest power of 2 that is >= win_length
    n_fft = int(math.pow(2, math.ceil(math.log2(win_length))))

    # DC removal
    audio = audio - np.mean(audio)

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        n_mels=nb_filters,
        center=False,
        htk=False,
        norm="slaney",
    )

    # Expected number of frames
    expected_length = int(len(audio) / hop_length)

    # Fix output length
    if mel_spectrogram.shape[1] < expected_length:
        # Pad with zeros
        mel_spectrogram = np.pad(
            mel_spectrogram, ((0, 0), (0, expected_length - mel_spectrogram.shape[1])),
            mode="constant"
        )
    elif mel_spectrogram.shape[1] > expected_length:
        # Trim to match
        mel_spectrogram = mel_spectrogram[:, :expected_length]

    return mel_spectrogram
