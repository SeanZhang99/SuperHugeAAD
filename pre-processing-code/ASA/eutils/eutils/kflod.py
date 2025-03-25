#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   kflod.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/6 22:20   lintean      1.0         None
'''
from sklearn.model_selection import KFold
import numpy as np


def remove_repeated(train_index, test_index, split_index):
    delete_axis = []
    for i in range(len(test_index)):
        for j in range(len(train_index)):
            if j in delete_axis:
                continue
            train_left, train_right = split_index[train_index[j], 0], split_index[train_index[j], 1]
            test_left, test_right = split_index[test_index[i], 0], split_index[test_index[i], 1]
            
            if train_left < test_left < train_right or train_left < test_right < train_right:
                delete_axis.append(j)
    print(f"remove repeated training window: {np.array(train_index)[delete_axis]}")
    train_index = np.delete(np.array(train_index), delete_axis, axis=0)

    return list(train_index)


def five_fold(data_eeg, label, split_index, shuffle=True, keep_trial=False):
    # five-fold cross-validation
    train_flod, test_flod = [[], [], [], [], []], [[], [], [], [], []]

    sum = 0
    for i_kra in range(len(data_eeg)):
        fold = 0
        kf = KFold(n_splits=5, shuffle=shuffle)
        for train_index, test_index in kf.split(data_eeg[i_kra], label[i_kra]):
            train_index = remove_repeated(train_index, test_index, split_index[i_kra])

            if keep_trial == False:
                train_flod[fold].extend(list(np.array(train_index) + sum))
                test_flod[fold].extend(list(np.array(test_index) + sum))
            else:
                train_flod[fold].append(np.array(train_index) + sum)
                test_flod[fold].append(np.array(test_index) + sum)

            fold += 1
        sum += data_eeg[i_kra].shape[0]
  

    return train_flod, test_flod