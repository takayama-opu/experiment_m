# coding: utf-8

import torch
import torchaudio
import matplotlib.pyplot as plt
path = '../dataset/STAIR_ACTIONS_DATASET/extract/audios/a094/0178C/0178C.mp3'
path = '../dataset/STAIR_ACTIONS_DATASET/for_test/yuyan.mp3'
waveform, sample_rate = torchaudio.load("../dataset/STAIR_ACTIONS_DATASET/extract/audios/mp3/a018/0031C/0031C.mp3")
amp = torchaudio.transforms.AmplitudeToDB()
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))
print(waveform)
plt.figure()
plt.plot(waveform.t().numpy())
plt.show()

specgram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow((amp(specgram.unsqueeze(0)).squeeze(0))[0,:,:].detach().numpy())
plt.show()

from torchlibrosa.augmentation import SpecAugmentation

spec_augmenter = SpecAugmentation(time_drop_width=8,time_stripes_num=2,freq_drop_width=8,freq_stripes_num=2)

tmp = spec_augmenter(specgram.unsqueeze(0)).squeeze(0)
plt.figure()
plt.imshow(amp(tmp)[0,:,:].detach().numpy())
plt.show()