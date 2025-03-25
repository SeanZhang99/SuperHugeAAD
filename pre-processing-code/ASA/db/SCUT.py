#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   SCUT.py.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/28 20:27   lintean      1.0         None
'''
import numpy as np

trail_number = 32

# the eeg sample rate is 1000
scut_eeg_fs = 1000

# the speech sample rate is 44100
scut_voice_fs = 44100

# direction and speaker
scut_label = [[0, 0], [1, 0], [0, 0], [1, 0],
         [0, 1], [1, 1], [0, 1], [1, 1],
         [1, 0], [0, 0], [1, 0], [0, 0],
         [1, 1], [0, 1], [1, 1], [0, 1],
         [0, 0], [1, 0], [0, 0], [1, 0],
         [0, 1], [1, 1], [0, 1], [1, 1],
         [1, 0], [0, 0], [1, 0], [0, 0],
         [1, 1], [0, 1], [1, 1], [0, 1]]

# S9-S17 voice and label order (1-32)
scut_suf_order = np.array([24, 8, 32, 31, 7, 15, 29, 6, 12, 26, 9, 1, 5, 28, 20, 11, 30, 21, 2, 4, 10, 13, 17, 25, 14, 22, 23, 19, 16, 27, 18, 3]) - 1

# remove some trails with poor answers. Note that the trail code is recorded here, and the trial subscript needs to be subtracted by 1
scut_remove_trail = {
    '1': [1, 2, 3, 9, 14, 24, 31],
    '2': [9, 29],
    '3': [1, 9, 19],
    '4': [31],
    '5': [14],
    '6': [7, 9, 10, 13, 15, 29],
    '7': [9, 25],
    '8': [25, 26, 27, 28, 29, 30, 31, 32],
    '9': [15],
    '10': [12, 15, 16],
    '11': [12, 17, 30],
    '13': [27, 29],
    '14': [9, 14, 26, 27],
    '16': [1]
}

# scut channel order
channel_names_scut = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                      'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10',
                      'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4',
                      'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
                      'FT9', 'FT10', 'Fpz', 'CPz', 'FCz']