# coding: UTF-8

import msgpack

for msg in msgpack.Unpacker(open('video_fps_info.msgpack', 'rb')):
    print(msg)
