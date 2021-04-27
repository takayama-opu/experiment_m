# coding: UTF-8
import torch
import matplotlib.pyplot as plt
import librosa
import torchaudio
import librosa.display
from pathlib import Path
path = '../dataset/STAIR_ACTIONS_DATASET/extract/audios/a094/0178C/0178C.mp3'
path = '../dataset/STAIR_ACTIONS_DATASET/for_test/yuyan.mp3'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(device)
a, sr = librosa.load("0188C.mp3")
print(a)
print(sr)
librosa.display.waveplot(a, sr=sr)

plt.show()

import numpy as np
import wave
## wav ファイル読み込み
wr = wave.open("hamu.wav", "r")
data = wr.readframes(wr.getnframes())
numch = wr.getnchannels()
samplewidth = wr.getsampwidth()
dt = wr.getframerate()
N = wr.getnframes()
print("チャンネル数 = ", numch)
print("サンプル幅 (バイト数) = ", samplewidth)
print("サンプリングレート (Hz) =", dt)
print("サンプル数 =", N)
print("録音時間 =", N / dt)
wr.close()

from pydub import AudioSegment

# 音声ファイルの読み込み
sound = AudioSegment.from_file("test.mp3", "mp3")
print(np.array(sound.get_array_of_samples()))
# 情報の取得
time = sound.duration_seconds # 再生時間(秒)
rate = sound.frame_rate  # サンプリングレート(Hz)
channel = sound.channels  # チャンネル数(1:mono, 2:stereo)

# for i in np.array(sound.get_array_of_samples()):
#     print(i)

# 情報の表示
print('チャンネル数：', channel)
print('サンプリングレート：', rate)
print('再生時間：', time)

filename = "Music01.wav"
waveform, sample_rate = torchaudio.load("0188C.wav")

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))
# 32768
plt.figure()
plt.plot(waveform.t().numpy())
plt.show()