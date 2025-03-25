#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   database.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/28 20:29   lintean      1.0         None
'''

db_path = f'E:\EEG dataset\ASA'
ch64 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
    'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10',
    'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4',
    'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
    'FT9', 'FT10', 'Fpz', 'CPz', 'FCz'
]
ch32 = [
    'Fp1', 'AF3', 'F3', 'F7',
    'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7',
    'PO3', 'O1', 'Oz', 'Pz',
    'Fp2', 'AF4', 'Fz', 'F4',
    'F8', 'FC6', 'FC2', 'Cz',
    'C4', 'T8', 'CP6', 'CP2',
    'P4', 'P8', 'PO4', 'O2'
]
ch32.sort()
ch16 = [
    'Fp1', 'F3', 'C3', 'T7',
    'P3', 'O1', 'Oz', 'Pz',
    'Fp2', 'Fz', 'F4', 'Cz',
    'C4', 'T8', 'P4', 'O2'
]
ch10 = [
    'FT7', 'FT8', 'FT9', 'FT10',
    'T7', 'T8', 'TP7', 'TP8',
    'TP9', 'TP10'
]
