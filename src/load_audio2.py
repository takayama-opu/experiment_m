# coding: utf-8

import torch
import torchaudio
import matplotlib.pyplot as plt
path = '../dataset/STAIR_ACTIONS_DATASET/extract/audios/a094/0178C/0178C.mp3'
path = '../dataset/STAIR_ACTIONS_DATASET/for_test/yuyan.mp3'
waveform, sample_rate = torchaudio.load("../dataset/STAIR_ACTIONS_DATASET/extract/audios/wav/a018/0031C/0031C.wav")

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))
print(waveform)
plt.figure()
plt.plot(waveform.t().numpy())
plt.show()