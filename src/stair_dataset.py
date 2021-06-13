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

ACTION_NUM = ["a001", "a002", "a018", "a033", "a094"]
ACTION_DIC = {"drinking":0,"eating_meal":1,"washing_face":2,"gardening":3,"fighting":4}
ex_path = {"dataset_root":'../dataset/STAIR_ACTIONS_DATASET/'}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

from gensim.models.doc2vec import Doc2Vec
d2v_model = Doc2Vec.load("../model/d2v/wiki_ex_doc2vec.model")

# 最後の1次元に指定サイズにCropし、長さが足りない時はCircularPadする
# 音声データの時間方向の長さを揃えるために使うtransform部品
class CircularPad1dCrop:
    def __init__(self, size):
        self.size = int(size)
    def __call__(self, x):
        n_repeat = int(self.size // x.size()[-1]) + 1
        repeat_sizes = ((1,) * (x.dim() - 1)) + (n_repeat,)
        out = x.repeat(*repeat_sizes).clone()
        return out.narrow(-1, 0, self.size)

class StairAudioDataset():
    resample_rate = 44100 // 4
    crop_time = 5.00 # [sec]
    def __init__(self, mode='train', test_rate=0.3, val_rate=0.2, augment=False, aug_size=10):
        self.action_dic = pd.read_csv(ex_path["dataset_root"]+'caption/actionlist.csv',
                    encoding="shift-jis",
                    dtype={'Action ID': str,'English category': str, 'Japanese category': str}).set_index('Action ID').T.to_dict('list')

        self.mode = mode
        self.augment = augment
        self.aug_size = aug_size
        dataset = self.load_audio()

        print(dataset)
        print(type(dataset))

        n_train_val = int(len(dataset) * (1-test_rate))
        n_test = len(dataset) - n_train_val
        torch.manual_seed(torch.initial_seed())  # 同じsplitを得るために必要
        #train_val_dataset, test_dataset = random_split(dataset, [n_train_val, n_test])
        train_val_dataset, test_dataset = train_test_split(dataset, test_size=n_test, random_state=25, shuffle=True)
        n_train = int(len(dataset) * (1-val_rate-test_rate))
        n_val = len(train_val_dataset) - n_train
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=n_val, random_state=25, shuffle=True)

        print(len(test_dataset) + len(val_dataset) + len(train_dataset))

        if self.mode == 'train':
            self.dataset = train_dataset
        elif self.mode == 'val':
            self.dataset = val_dataset
        elif self.mode == 'test':
            self.dataset = test_dataset
        else:
            self.dataset = None

        self.pre_transform = transforms.Compose([torchaudio.transforms.Resample(self.resample_rate),
                                                 CircularPad1dCrop(self.resample_rate*self.crop_time),
                                                 torchaudio.transforms.MelSpectrogram(self.resample_rate)
                                                 ])
        self.post_transform = transforms.Compose([torchaudio.transforms.AmplitudeToDB(),
                                                 transforms.Resize((128,128))])

        self.spec_augmenter = SpecAugmentation(time_drop_width=16,
                                               time_stripes_num=2,
                                               freq_drop_width=16,
                                               freq_stripes_num=2,
                                               )

        self.dataset = self.load_waveform(pre_transform=True, augment=self.augment, aug_size=self.aug_size)

    def load_audio(self):
        pth = ex_path["dataset_root"]+"extract/audios/mp3/"

        data = pd.read_csv(
                ex_path["dataset_root"] + "stair_audio_list.csv",
                index_col=0,
                dtype={'Action_ID': str, 'Video_ID': str, 'label': str, 'length': float, 'sample_rate': int, 'is_original': bool},
                usecols=lambda x: x is not 'index'
        )

        return data

    def load_waveform(self, pre_transform=True, augment=False, aug_size=10, post_transform=True):
        # waveform列の追加
        data = self.dataset.assign(waveform=None)
        data['waveform'] = data['waveform'].astype(object)
        data = data.reset_index(drop=True)

        curr_size = len(data)

        for i, v in data.iterrows():
            print("mode : {}, i : {}/{}".format(self.mode, i+1, curr_size))

            data.at[i, 'waveform'] = torchaudio.load(data.at[i, 'path'])[0]

            if data.at[i, 'is_original'] & pre_transform:
                data.at[i, 'waveform'] = self.pre_transform(data.at[i, 'waveform'])

            if augment:
                if data.at[i, 'is_original']:

                    for j in range(1, aug_size):
                        tmp = data.loc[i].copy()
                        tmp.is_original = False

                        tmp.waveform = (self.spec_augmenter(tmp.waveform.unsqueeze(0))).squeeze(0)
                        if post_transform:
                            tmp.waveform = self.post_transform(tmp.waveform)
                        data = data.append(tmp)
                        data = data.reset_index(drop=True)

            if post_transform:
                data.at[i, 'waveform'] = self.post_transform(data.at[i, 'waveform'])



        return data.reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset.iloc[idx]

        x.label = one_hot(torch.tensor(ACTION_DIC[x.label]), num_classes=5).long()

        return x.waveform, x.label

class StairCaptionDataset():
    def __init__(self, mode='train', test_rate=0.3, val_rate=0.2):
        self.action_dic = pd.read_csv(ex_path["dataset_root"]+'caption/actionlist.csv',
                    encoding="shift-jis",
                    dtype={'Action ID': str,'English category': str, 'Japanese category': str}).set_index('Action ID').T.to_dict('list')

        self.mode = mode

        dataset = self.load_caption()

        print(dataset)
        print(type(dataset))

        n_train_val = int(len(dataset) * (1-test_rate))
        n_test = len(dataset) - n_train_val
        torch.manual_seed(torch.initial_seed())  # 同じsplitを得るために必要
        #train_val_dataset, test_dataset = random_split(dataset, [n_train_val, n_test])
        train_val_dataset, test_dataset = train_test_split(dataset, test_size=n_test, random_state=25, shuffle=True)
        n_train = int(len(dataset) * (1-val_rate-test_rate))
        n_val = len(train_val_dataset) - n_train
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=n_val, random_state=25, shuffle=True)

        print(len(test_dataset) + len(val_dataset) + len(train_dataset))

        if self.mode == 'train':
            self.dataset = train_dataset
        elif self.mode == 'val':
            self.dataset = val_dataset
        elif self.mode == 'test':
            self.dataset = test_dataset
        else:
            self.dataset = None

        self.dataset = self.emb_wakati()

    def load_caption(self):
        data = pd.read_csv(
                ex_path["dataset_root"] + "stair_caption_dup_list.csv",
                index_col=0,
                dtype={'Action_ID': str, 'Video_ID': str, 'label': str, 'is_original': bool, 'wakati': str},
                usecols=lambda x: x is not 'index'
        )

        return data

    def emb_wakati(self):
        # waveform列の追加
        data = self.dataset.assign(emb=None)
        data['emb'] = data['emb'].astype(object)
        data = data.reset_index(drop=True)

        for i, v in data.iterrows():
            data.at[i, 'emb'] = d2v_model.infer_vector(data.at[i, 'wakati'].split(" "))
        return data.reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset.iloc[idx]

        x.label = one_hot(torch.tensor(ACTION_DIC[x.label]), num_classes=5).long()

        return x.emb, x.label

# a = StairAudioDataset(mode='train')
# print(len(a.dataset))
# b = StairAudioDataset(mode='val')
# print(len(b.dataset))
c = StairAudioDataset(mode='test', augment=True, aug_size=2)
# print(c.dataset)
print(len(c.dataset))


# print(c[0].waveform)
# print(c[0].Video_ID)
# print(c.dataset.waveform[204])
# print(c.dataset.waveform[204].size())
#
# print(c.dataset.waveform[0])
# print(c.dataset.waveform[0].size())

train_dataloader = DataLoader(c, batch_size=32, shuffle=True)

net = resnet34(pretrained=True)
net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# 最後の層の次元を今回のカテゴリ数に変更する
net.fc = nn.Linear(512,5)
# print(net)
net.to(device)
for x_train, y_train in train_dataloader:
    x = x_train.to(device)
    y = y_train.to(device)
    print(x.size())
    print(y)
    print(y.size())
    #print(x_train, y_train)
    out = net(x)
    #print(out)


# print(c.dataset['Action_ID'])
# print(c.dataset['waveform'])
#
#
# print(c.dataset.iloc[0])
# print(c.dataset.iloc[0].waveform)



