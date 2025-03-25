import os
import json
import h5py
import numpy as np
import mne

from scipy.io import wavfile
from scipy.signal import hilbert
from scipy.stats import zscore
from mne.filter import resample, filter_data

# change these to where you downloaded and uncompressed the data, and where you want to save the preprocessed data.
data_download_dir = "E:\EEG_dataset_Superhuge\Estart_2019"
preprocessed_data_dir = data_download_dir

mne.set_log_level('CRITICAL')

# find out which sessions the participants took part in
with open(os.path.join(data_download_dir, 'session_info.json'), 'r') as f:
    session_info = json.load(f)

output_fname = 'english_session_data_preprocessed.h5'


for condition in ['clean', 'lb', 'mb', 'hb', 'fM', 'fW']:

    print(condition)

    for part in [1,2,3,4]:

        fs, audio = wavfile.read(os.path.join(data_download_dir, 'audiobooks', 'audiobooks',condition, f'part_{part}_story.wav'))
        assert fs == 44100

        # process the attended speech stream
        attended_envelope = np.abs(hilbert(audio))
        attended_envelope = resample(attended_envelope, 5, 1764) # resample envelope to 125 Hz
        attended_envelope = filter_data(attended_envelope, sfreq=125, l_freq=None, h_freq=50)


        # account for one second of babble (only) at start and end of SiN trials
        if condition in ['lb', 'mb', 'hb']:
            attended_envelope = attended_envelope[125:-125]

        # also pre-process the unattended speech stream in the case of competing-speakers
        if condition in ['fM', 'fW']:

            fs, audio_unattended = wavfile.read(os.path.join(data_download_dir, 'audiobooks', 'audiobooks', condition, f'part_{part}_distractor.wav'))
            assert fs == 44100
            unattended_envelope = np.abs(hilbert(audio_unattended))
            unattended_envelope = resample(unattended_envelope, 5, 1764) # resample envelope to 125 Hz
            unattended_envelope = filter_data(unattended_envelope, sfreq=125, l_freq=None, h_freq=50)

            # sometimes one audiobook plays for longer than the other. Truncate & save.
            max_length = min(len(attended_envelope), len(unattended_envelope))
            with h5py.File(os.path.join(preprocessed_data_dir, output_fname), 'a') as f:
                f.create_dataset(f"{condition}/part{part}/unattended", data=unattended_envelope[:max_length])

        else:
            max_length = len(attended_envelope)

        # save the pre-processed attendeds peech envelope
        with h5py.File(os.path.join(preprocessed_data_dir, output_fname), 'a') as f:
            f.create_dataset(f"{condition}/part{part}/attended", data=zscore(attended_envelope[:max_length]))

        # now pre-process the eeg
        for participant in session_info['english_session_participants']:

            with h5py.File(os.path.join(data_download_dir, 'eeg', f'{participant}.h5'), 'r') as f:
                eeg = f[f'{condition}/part_{part}'][:]

            eeg = resample(eeg, 1, 8) # resample 1kHz to 125Hz.
            eeg = filter_data(eeg, sfreq=125, l_freq=0.5, h_freq=8)

            # account for one second of babble (only) at start and end of SiN trials
            if condition in ['lb', 'mb', 'hb']:
                eeg = zscore(eeg[:, 125:125+len(attended_envelope)], axis=1)
            else:
                eeg = zscore(eeg[:, :max_length], axis=1)

            print(attended_envelope.shape, eeg.shape)

            with h5py.File(os.path.join(preprocessed_data_dir, output_fname), 'a') as f:
                f.create_dataset(f"{condition}/part{part}/{participant}", data=eeg)