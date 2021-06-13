import pandas as pd
import torch
import numpy as np
# coding: utf-8
from collections import defaultdict
import torch
import torchaudio
from torch.nn.functional import one_hot
from torchvision import transforms
from torchvision.models import resnet34
from transformers import BertConfig, BertModel
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from torchlibrosa.augmentation import SpecAugmentation
from torch.autograd import Variable
import optuna
from sklearn.metrics import classification_report
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import Pool
# List of Tuples
fruit_list = [ ('Orange', 34, 'Yes' , True)]
#Create a DataFrame object
df = pd.DataFrame(fruit_list, columns = ['Name' , 'Price', 'Stock', 'is_original'])
#Add new ROW
df.loc[1]=[ 'Mango', 4, 'No' , True]
df.loc[2]=[ 'Apple', 14, 'Yes' , True]
print(df)
print("")
for i, v in df.iterrows():
    if df.at[i, 'is_original']:
        a = df.iloc[i].copy()
        a.is_original = False
        a.Price *= 2
        df = df.append(a)
df = df.reset_index(drop=True)
print(df)

b = df.iloc[0].copy()
b.Price = 0
print(df)

# miria = [torch.tensor([1, 0, 2, 2, 1, 1, 2, 3, 2, 3, 4, 3, 1, 2, 2, 2, 2, 0, 1, 2, 0, 4, 2, 1,
#         1, 2, 4, 2, 4, 2, 4, 1]),
#  torch.tensor([4, 3, 2, 1, 0, 2, 1, 2, 2, 2, 2, 0, 2, 3, 1, 3, 0, 2, 1, 3, 1, 2, 1, 4,
#         2, 2, 1, 2, 2, 2, 4, 2]),
#  torch.tensor([0, 1, 0, 1, 2, 0, 2, 2, 2, 0, 2, 0, 1, 2, 3, 1, 4, 4, 0, 0, 2, 2, 2, 2,
#         2, 2, 2, 2, 4])]
# print(len(miria))
# for i in range(len(miria)):
#     miria[i] = miria[i].to('cpu').detach().numpy().copy()
# print(miria)
# import itertools
#
# print(list(itertools.chain.from_iterable(miria)))
# print(miria[1])
# print(miria[2])

hoge = np.array([1,2,3,4,5])
hoge = 1 / (hoge / sum(hoge))
print(hoge)

YuiSeries = pd.Series({"math":10, "japanese":50, "english":35})
print(YuiSeries)
print(YuiSeries["math"])

def aaaaa():
    b = [1,2,3,4,5,6,7,8,9,10]
    train_val, test = train_test_split(b, test_size=3, random_state=25, shuffle=True)
    train, val = train_test_split(train_val, test_size=2, random_state=25, shuffle=True)
    return train, val, test

tt = aaaaa()
print(tt)

ttt=aaaaa()
print(ttt)

ex_path = {"dataset_root":'../dataset/STAIR_ACTIONS_DATASET/'}

data = pd.read_csv(
    ex_path["dataset_root"] + "stair_audio_list.csv",
    index_col=0,
    dtype={'Action_ID': str, 'Video_ID': str, 'label': str, 'length': float, 'sample_rate': int, 'is_original': bool},
    usecols=lambda x: x is not 'index'
)

ACTION_NUM = ["a001", "a002", "a018", "a033", "a094"]

for a in ACTION_NUM:
    tmp = data[data['Action_ID']==a]
    print(tmp['length'].mean())
    print(tmp['length'].describe())