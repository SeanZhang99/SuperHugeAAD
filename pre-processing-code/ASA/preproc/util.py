#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   util.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/28 20:37   lintean      1.0         None
'''
import numpy as np


def voice_dirct2attd(voice, label):
    for i in range(len(voice)):
        if label[i][0] == 1:
            voice[i][0], voice[i][1] = voice[i][1], voice[i][0]
    return voice


def voice_attd2speaker(voice, label):
    for i in range(len(voice)):
        if label[i][1] == 1:
            voice[i][0], voice[i][1] = voice[i][1], voice[i][0]
    return voice


def select_label(label, label_type):
    nlabel = []
    label_index = None
    if label_type == "direction":
        label_index = 0
    elif label_type == "speaker":
        label_index = 1
    else:
        raise ValueError('"label_type" does not belong to known (direction, speaker)')

    for i in range(len(label)):
        nlabel.append(label[i][label_index])
    return nlabel


def _adjust_order(x, sub_id):
    from db.SCUT import scut_suf_order
    if int(sub_id) > 8:
        nx = []
        for k_tra in scut_suf_order:
            nx.append(x[k_tra])
        x = nx
    return x


def _reverse_label(label, sub_id):
    if int(sub_id) > 8:
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = 1 - label[i][j]
    return label


def scut_order(voice, label, sub_id):
    """
    The audio sequences of the first 8 and the last 9 are different, and the corresponding EEG signals of the latter 9 are recovered through this
    Subject numbering starts at 1, so S8 doesn't need to be recoded either
    The attention of the last 9 is also different
    Args:
        voice: voice
        label: label
        sub_id: subject number

    Returns: [voice, label] or label

    """
    label = _adjust_order(label, sub_id)
    label = _reverse_label(label, sub_id)
    if voice == None:
        return label
    else:
        voice = _adjust_order(voice, sub_id)
        return voice, label


def scut_remove(eeg, voice, label, sub_id):
    """
    Handling abnormal Trails, such as incomplete Trails, etc.
    Args:
        eeg: eeg
        voice: voice
        label: label

    Returns: [eeg, voice, label] or [eeg, label]

    """
    from db.SCUT import scut_remove_trail
    if isinstance(eeg, list):
        if sub_id in scut_remove_trail:
            remove_index = scut_remove_trail[sub_id]
            remove_index = sorted(remove_index, reverse=True)

            print('Attention: delete a specific index')
            print(remove_index)

            # remove abnormal Trail
            for k_pop in remove_index:
                eeg.pop(k_pop - 1)
                if voice is not None:
                    voice.pop(k_pop - 1)
                label.pop(k_pop - 1)
    elif isinstance(eeg, np.ndarray):
        if sub_id in scut_remove_trail:
            keep_index = list(range(32))
            remove_index = scut_remove_trail[sub_id]

            print('Attention: delete a specific index')
            print(remove_index)

            # remove abnormal Trail
            for k_pop in reversed(remove_index):
                keep_index.pop(k_pop - 1)
                label.pop(k_pop - 1)
            eeg = eeg[keep_index]
    if voice is not None:
        return eeg, voice, label
    else:
        return eeg, label


def trails_split(x, time_len, window_lap, sampling_rate=128):
    """
    divide the signal
    Args:
        x: list[np.ndarray], len(x) is trails_num, each element shape as [time, channel]
        time_len:
        window_lap:

    Returns: divided data, shape as [sample, time, channel]

    """
    sample_len = int(sampling_rate * time_len)

    split_index = []
    split_x = []
    # divide for each trail
    for i_tra in range(len(x)):
        trail_len = x[i_tra].shape[0]
        left = np.arange(start=0, stop=trail_len, step=window_lap)
        while left[-1] + sample_len - 1 > trail_len - 1:
            left = np.delete(left, -1, axis=0)
        right = left + sample_len

        split_index.append(np.stack([left, right], axis=-1))

        temp = [x[i_tra][left[i]: right[i]] for i in range(left.shape[0])]
        split_x.append(np.stack(temp, axis=0))

    return split_x, split_index


def list_data_split_regression(eeg, voice, label, time_len, window_lap):
    """
    divide the data, windowing the data at fixed intervals
    Args:
        voice: voice
        eeg: eeg
        label: label
        time_len: sample duration
        window_lap: fixed interval
        num_classes: number of classes for the classification task, default is 2
        
    Returns:
        eeg：divided eeg
        label：divided label
        split_index: record the left and right positions of each window
    """
    # eeg and voice
    if voice is not None:
        for i in range(len(voice)):
            voice[i] = np.stack(voice[i], axis=-1)
            if voice[i].shape[0] != eeg[i].shape[0]:
                raise ValueError("the length of voice and eeg must be the same")
        voice, split_index = trails_split(voice, time_len, window_lap)
    eeg, split_index = trails_split(eeg, time_len, window_lap)

    # label uses one-hot encoding
    total_label = []
    for i_tra in range(len(eeg)):
        samples_num = eeg[i_tra].shape[0]
        sub_label = label[i_tra] * np.ones(samples_num)

        total_label.append(sub_label)

    label = total_label

    if voice is not None:
        return eeg, voice, label, split_index
    else:
        return eeg, label, split_index
    
    
def list_data_split(eeg, voice, label, time_len, window_lap, sampling_rate=128, num_classes=2):
    """
    divide the data, windowing the data at fixed intervals
    Args:
        voice: voice
        eeg: eeg
        label: label
        time_len: sample duration
        window_lap: fixed interval
        sampling_rate: sampling rate, default is 128
        num_classes: number of classes for the classification task, default is 2
        
    Returns:
        eeg：divided eeg
        label：divided label
        split_index: record the left and right positions of each window
    """
    # eeg and voice
    if voice is not None:
        for i in range(len(voice)):
            voice[i] = np.stack(voice[i], axis=-1)
            if voice[i].shape[0] != eeg[i].shape[0]:
                raise ValueError("the length of voice and eeg must be the same")
        voice, split_index = trails_split(voice, time_len, window_lap, sampling_rate=sampling_rate)
    eeg, split_index = trails_split(eeg, time_len, window_lap, sampling_rate=sampling_rate)

    # label uses one-hot encoding
    total_label = []
    for i_tra in range(len(eeg)):
        samples_num = eeg[i_tra].shape[0]
        sub_label = label[i_tra] * np.ones(samples_num)
        # Modify here to use num_classes for one-hot encoding
        sub_label = np.eye(num_classes)[sub_label.astype(int)]
        total_label.append(sub_label)

    label = total_label

    if voice is not None:
        return eeg, voice, label, split_index
    else:
        return eeg, label, split_index



def normalize_data(data):
    """
    Normalize the data to have a mean of 0 and standard deviation of 1.
    Args:
        data: The data to be normalized.

    Returns:
        The normalized data.
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
