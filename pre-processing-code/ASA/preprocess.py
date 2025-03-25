# python3
# encoding: utf-8
#
# @Time    : 2021/12/22 14:51
# @Author  : enze
# @Email   : enzesu@hotmail.com
# @File    : preprocess.py
# @Software: Pycharm

import copy
import os
import mne
import scipy.io
import sys
db_folder_path = r"E:/EEG dataset/ASA/db/"
sys.path.append(db_folder_path)
from mne.preprocessing import ICA, corrmap
from db.SCUT import scut_eeg_fs, scut_label, trail_number, channel_names_scut
from preproc.util import *
from db.database import db_path as folder_path

# basic settings
montage_dict = {'SCUT': 'standard_1020'}
fs_dict = {'SCUT': scut_eeg_fs}


def preprocess(dataset_name, sub_id, l_freq=None, h_freq=None, is_ica=True, label_type='direction'):
    """
    Read data, ICA, bandpass filtering
    Args:
        dataset_name: dataset name
        sub_id: subject number
        l_freq: bandpass filter parameters
        h_freq: bandpass filter parameters
        is_ica: whether to perform ICA processing
        time_len: the duration of the sample, in seconds
        label_type: category labels，direction or speaker
        window_lap: the interval between two adjacent windows when data is divided

    Returns:
        data：raw data，shape: N*P*C，N: number of samples，P: sample length，C: number of channels
        label：shape：N*2
        split_index:

    """
    if h_freq is None:
        h_freq = [50]
    if l_freq is None:
        l_freq = [1]
    if isinstance(l_freq, int):
        l_freq = [l_freq]
        h_freq = [h_freq]

    # load data，data：Trail*Time*Channels，label：Trail*1
    data, label = data_loader(dataset_name, sub_id, label_type)

    # ICA
    data = data_ica(data, dataset_name, label_type) if is_ica else data

    # filter
    data = data_filter(data, dataset_name, l_freq, h_freq)

    return data, label

def data_split(eeg, label, target, time_len, overlap):
    """
    将序列数据转化为样本，并按照5折的方式的排列整齐
    不兼容128Hz以外的数据（含有语音和脑电信号）
    :param eeg: 脑电数据，列表形式的Trail，每个Trail的形状是Time*Channels
    :param voice: 语音数据，列表形式的Trail，每个Trail的形状是Time*2
    :param label: 标签数据，根据targe决定输出标签
    :param target: 确定左右/讲话者的分类模型
    :param time_len: 样本的长度
    :param overlap: 样本的重叠率
    :return:
    """

    sample_len = int(128 * time_len)

    my_trail_samples = np.empty((5, 0, sample_len, eeg[0].shape[-1]))

    my_labels = np.empty((5, 0))

    for k_tra in range(len(eeg)):
        trail_eeg = eeg[k_tra]

        trail_label = label[k_tra][target]

        # 确定重叠率的数据
        over_samples = int(sample_len * (1 - overlap))
        over_start = list(range(0, sample_len, over_samples))
        # 根据起点划分数据
        for k_sta in over_start:
            tmp_eeg = set_samples(trail_eeg, k_sta, sample_len, overlap)


            my_trail_samples = np.concatenate((my_trail_samples, tmp_eeg), axis=1)
            # my_voice_samples = np.concatenate((my_voice_samples, tmp_voice), axis=1)
            my_labels = np.concatenate((my_labels, trail_label * np.ones((5, tmp_eeg.shape[1]))), axis=1)

    # 转化为单一维度的数据
    my_trail_samples = np.reshape(my_trail_samples, [-1, my_trail_samples.shape[2], my_trail_samples.shape[3]])
    # my_voice_samples = np.reshape(my_voice_samples, [-1, my_voice_samples.shape[2], my_voice_samples.shape[3]])
    my_labels = np.reshape(my_labels, -1)

    return my_trail_samples, my_labels

def set_samples(trail_data, k_sta, sample_len, overlap):
    # 切分整数长度
    data_len, channels_num = trail_data.shape[0], trail_data.shape[1]
    k_end = (data_len - k_sta) // sample_len * sample_len + k_sta
    trail_data = trail_data[k_sta:k_end, :]

    # cutoff
    trail_data = np.reshape(trail_data, [-1, sample_len, channels_num])

    # 划分为5折数据
    # TODO: 检查数据是否为连续时间序列，方便后续的五折交叉验证
    trails_num = trail_data.shape[0] // 5 * 5
    trail_data = trail_data[0:trails_num, :, :]
    trail_data = np.reshape(trail_data, [5, int(trail_data.shape[0] / 5), sample_len, channels_num])

    if overlap != 0:
        trail_data = trail_data[:, 1:-1, :]

    return trail_data

def data_loader(dataset_name, sub_id, label_type):
    """
    Load the data of the specified database
    Args:
        dataset_name: dataset name
        sub_id: subject number
        label_type: category labels，direction or speaker
    Returns:
        data:Trail*Time*Channel
        label:Trail*1

    """
    data, label = [], []

    if dataset_name == 'SCUT':
        # basic Information
        fs = 1000
        trails_num, points_num, channels_num = 32, 55 * fs, 64

        data = np.empty((0, points_num, channels_num))

        # load
        data_path = f'{folder_path}/AAD_{dataset_name}/S{sub_id}/'
        files = os.listdir(data_path)
        files = sorted(files)
        for file in files:
            # input formatting
            data_mat = scipy.io.loadmat(data_path + file)
            for k_tra in range(data_mat['Markers'].shape[1] // 3):
                k_sta = data_mat['Markers'][0, 3 * k_tra + 2][3][0][0]

                trail_data = np.zeros((1, points_num, channels_num))

                if len(data_mat[channel_names_scut[0]]) >= k_sta + points_num:
                    for k_cha in range(len(channel_names_scut)):
                        trail_data[0, :, k_cha] = data_mat[channel_names_scut[k_cha]][k_sta:k_sta + points_num, 0]

                data = np.concatenate((data, trail_data), axis=0)
        
        label = copy.deepcopy(scut_label)
        label = scut_order(None, label, sub_id)
        label = select_label(label, label_type=label_type)
        data, label = scut_remove(data, None, label, sub_id)

    else:
        print('Error, check the "dataset_name"!')

    data = data[:, :, 0:64]

    return data, label


def data_ica(data, dataset_name, label_type):
    """
    ICA processing of data
    Args:
        label_type: category labels，direction or speaker
        data: raw data
        dataset_name: dataset name
    Returns:
        data: raw data

    """

    ica_dict = {'SCUT': [0, 1]}

    # prepare Electrode Information
    info = set_info(dataset_name)

    # load template data（S1-Trail1）
    data_tmp, label_tmp = data_loader(dataset_name, '1', label_type)
    data_tmp = np.transpose(data_tmp, (0, 2, 1))
    data_tmp = data_tmp[0]

    # calculate the ica channel
    raw_tmp = mne.io.RawArray(data_tmp, info)
    raw_tmp = raw_tmp.filter(l_freq=1, h_freq=None)
    raw_tmp.set_montage(montage_dict[dataset_name])
    ica_tmp = ICA(n_components=20, max_iter='auto', random_state=97)
    ica_tmp.fit(raw_tmp)

    # remove EOG
    is_verbose = True
    data = np.transpose(data, (0, 2, 1))
    for k_tra in range(data.shape[0]):
        print(f'data ica, trail: {k_tra}')

        # Convert raw data to raw format
        raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)

        # calculate ica data
        raw = raw.filter(l_freq=1, h_freq=None, verbose=is_verbose)
        ica = ICA(n_components=20, max_iter='auto', random_state=97, verbose=is_verbose)  # 97为随机种子
        ica.fit(raw)

        ica_exclude = []
        ica_s = [ica_tmp, ica]
        eog_channels = ica_dict[dataset_name]
        for k_ica in range(len(eog_channels)):
            corrmap(ica_s, template=(0, eog_channels[k_ica]), threshold=0.9, label=str(k_ica), plot=False,
                    verbose=is_verbose)
            ica_exclude += ica_s[1].labels_[str(k_ica)]

        ica.exclude = list(set(ica_exclude))
        ica.apply(raw, verbose=is_verbose)
        print(ica.exclude)
        del ica
        print(raw)

        data[k_tra] = raw.get_data()

        is_verbose = False

    data = np.transpose(data, (0, 2, 1))

    return data


def data_filter(data, dataset_name, l_freq, h_freq):
    """
    filter the data
    Args:
        data: raw data
        dataset_name: dataset name
        l_freq: bandpass filter parameters
        h_freq: bandpass filter parameters

    Returns:
        data: raw data

    """

    points_num = int(data.shape[1] / fs_dict[dataset_name] * 128)
    data_resample = np.empty((0, 64, points_num))

    # filter
    is_verbose = True
    info = set_info(dataset_name)
    data = np.transpose(data, (0, 2, 1))

    for k_tra in range(data.shape[0]):
        print(f'data filter, trail: {k_tra}')

        for k in range(len(l_freq)):
            # convert raw data to raw format
            raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)

            # set eeg reference, filtering, downsampling
            raw = raw.set_eeg_reference(ref_channels='average', verbose=is_verbose)

            raw = raw.filter(l_freq=l_freq[k], h_freq=h_freq[k], verbose=is_verbose)
            raw = raw.resample(128)

            trail_data = raw.get_data()[0:64, :]
            trail_data = (trail_data - np.average(trail_data, axis=-1)[..., None]) / np.std(trail_data, axis=-1)[..., None]
            trail_data = np.expand_dims(trail_data, 0)
            data_resample = np.concatenate([data_resample, trail_data], axis=0)

            is_verbose = False

    # to Trail*Band*Time*Channel
    data = np.transpose(data_resample, (0, 2, 1))

    return data


def set_info(dataset_name):
    """
    get electrode information
    Args:
        dataset_name: dataset name
    Returns:
          info：information

    """

    ch_names = channel_names_scut
    ch_types = list(['eeg' for _ in range(len(ch_names))])

    info = mne.create_info(ch_names, fs_dict[dataset_name], ch_types)
    info.set_montage(montage_dict[dataset_name])

    return info


if __name__ == '__main__':
    x, y = preprocess(dataset_name='SCUT', sub_id='11', l_freq=[1], h_freq=[50], is_ica=True,
                      label_type='direction')