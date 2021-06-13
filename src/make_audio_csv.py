# coding: utf-8
from collections import defaultdict
import torch
import torchaudio
import re
from transformers import BertConfig, BertModel
import time
from pandas.io.json import json_normalize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import optuna
from sklearn.metrics import classification_report
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import glob

ACTION_NUM = ["a001", "a002", "a018", "a033", "a094"]

ex_path = {"dataset_root":'../dataset/STAIR_ACTIONS_DATASET/'}


def make_csv(path):
    data_columns = ['Action_ID', 'Video_ID', 'label', 'path', 'colab_path', 'length', 'sample_rate', 'is_original']

    data_set = []

    action_dic = pd.read_csv(ex_path["dataset_root"] + 'caption/actionlist.csv',
                             encoding="shift-jis",
                             dtype={'Action ID': str, 'English category': str, 'Japanese category': str}).set_index(
        'Action ID').T.to_dict('list')

    for action in ACTION_NUM:
        print(action)
        recurrent = path + action
        files = glob.glob(recurrent + '/**/*.ogg', recursive=True)
        for file in files:
            file = file.replace("\\","/")
            print(file)
            data = dict.fromkeys(data_columns)
            data['Action_ID'] = action
            data['Video_ID'] = file.split('/')[-1].replace(".ogg","")
            data['label'] = action_dic[data['Action_ID']][0] # [1]だと日本語
            data['path'] = file
            data['colab_path'] = '/content/drive/My Drive/experiment_m/' + file[3:]
            data['is_original'] = True

            waveform, sample_rate = torchaudio.load(file)

            data['sample_rate'] = sample_rate
            data['length'] = waveform.size()[1] / sample_rate

            data_set.append(data)

    return json_normalize(data=data_set)



def main():
    path = ex_path["dataset_root"]+"extract/audios/ogg/"
    res = make_csv(path)

    print(res)
    res.to_csv(ex_path["dataset_root"] + "stair_audio_list.csv", encoding='utf-8')




if __name__ == '__main__':
    main()