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
ACTION_LABEL = ["drinking", "eating_meal", "washing_face", "gardening", "fighting"]
ACTION_DIC = {"drinking":0, "eating_meal":1, "washing_face":2, "gardening":3, "fighting":4}


ex_path = {"dataset_root":'../dataset/STAIR_ACTIONS_DATASET/'}


def make_csv():
    data_columns = ['Action_ID', 'Video_ID', 'label', 'place', 'person', 'action', 'is_original']

    data_set = []

    action_dic = pd.read_csv(ex_path["dataset_root"] + 'caption/actionlist.csv',
                             encoding="shift-jis",
                             dtype={'Action ID': str, 'English category': str, 'Japanese category': str}).set_index(
        'Action ID').T.to_dict('list')

    caption_data = pd.read_csv(ex_path["dataset_root"] + 'caption/caption.csv',
                             encoding='cp932',
                             dtype={'video': str, 'place': str, 'person': str, 'action': str})

    caption_data = caption_data.assign(Action_ID=None)
    caption_data = caption_data.assign(Video_ID=None)

    for i, v in caption_data.iterrows():

        caption_data.at[i, 'Action_ID'] = caption_data.at[i, 'video'].split("-")[0]
        caption_data.at[i, 'Video_ID'] = caption_data.at[i, 'video'].split("-")[1].split(".")[0]

    for action in ACTION_NUM:
        print(action)
        miria = caption_data[caption_data['Action_ID'] == action]
        if len(miria) < 1:
            continue

        for i, v in miria.iterrows():
            data = dict.fromkeys(data_columns)
            data['Action_ID'] = action
            data['Video_ID'] = miria.at[i, 'Video_ID']
            data['label'] = action_dic[data['Action_ID']][0] # [1]だと日本語
            data['is_original'] = True
            data['place'] = miria.at[i, 'place']
            data['person'] = miria.at[i, 'person']
            data['action'] = miria.at[i, 'action']

            data_set.append(data)

    return json_normalize(data=data_set)



def main():
    path = ex_path["dataset_root"]+"extract/audios/ogg/"
    res = make_csv()

    print(res)
    res.to_csv(ex_path["dataset_root"] + "stair_caption_list.csv", encoding='utf-8-sig')




if __name__ == '__main__':
    main()