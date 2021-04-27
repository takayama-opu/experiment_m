# coding: utf-8
from collections import defaultdict
import cv2
import moviepy.editor as mp
import shutil
import os
import types


path = '../../dataset/STAIR_ACTIONS_DATASET/extract/images'  # 出力先
label = ["a001","a002","a018","a033","a094"]
for l in label:
    n = 0
    print(l)
    for current, subfolders, subfiles in os.walk(os.path.join(path,l), topdown=False):

        if len(subfolders) != 0:
            continue
        # print(f"現在のフォルダは{current}です。")
        # print(f"サブフォルダは{subfolders}です。")
        # print(f"サブファイルは{subfiles}です。")
        # print(len(subfiles))
        n += len(subfiles)

    print(n)

import glob
for l in label:
    print(len(glob.glob('../../dataset/STAIR_ACTIONS_DATASET/extract/images/' + l +'/*')))

