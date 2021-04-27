# coding: UTF-8
import torch
import msgpack

for msg in msgpack.Unpacker(open('video_fps_info.msgpack', 'rb')):
    print(msg)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(device)
print(torch.cuda.get_device_name())
