# coding: utf-8
from collections import defaultdict
import cv2
import moviepy.editor as mp
import shutil
import os
import types
d = defaultdict(lambda: defaultdict(float))
error_list = []

def save_all_frames(video_path, dir_path, file_name, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    d[os.path.splitext(file_name)[0]]['fps'] = cap.get(cv2.CAP_PROP_FPS)
    d[os.path.splitext(file_name)[0]]['frame_cnt'] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    d[os.path.splitext(file_name)[0]]['len_sec'] = d[os.path.splitext(file_name)[0]]['frame_cnt'] / d[os.path.splitext(file_name)[0]]['fps']

    file_name = os.path.splitext(file_name)[0].split('-', 1);

    if not cap.isOpened():
        print("ERROR CANT OPEN")
        return

    out_path = os.path.join(dir_path, file_name[0], file_name[1])

    print(out_path)

    #os.makedirs(out_path, exist_ok=True)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         cv2.imwrite('{}/{}_{}.{}'.format(out_path, file_name[1], str(n).zfill(digit), ext), frame)
    #         n += 1
    #     else:
    #         return



def substract_audio(video_path, dir_path, file_name, ext='mp3'):
    # Extract audio from input video.
    clip_input = mp.VideoFileClip(video_path).subclip()
    file_name = os.path.splitext(file_name)[0].split('-', 1);
    out_path = os.path.join(dir_path, file_name[0], file_name[1])
    #os.makedirs(out_path, exist_ok=True)

    if type(clip_input.audio) == type(None):
        print("AUDIO ERROR")
        error_list.append("-".join(file_name))
        print(error_list)
        #shutil.rmtree(out_path)
        #shutil.rmtree("../dataset/STAIR_ACTIONS_DATASET/extract/images/" + file_name[0] + "/" + file_name[1])
        return

    print(out_path)
    #clip_input.audio.write_audiofile('{}/{}.{}'.format(out_path, file_name[1], ext))


def main():
    bass_path = '../dataset/STAIR_ACTIONS_DATASET/original/videos'  # 入力
    out_bass_path = '../dataset/STAIR_ACTIONS_DATASET/extract'  # 出力先

    for current, subfolders, subfiles in os.walk(bass_path, topdown=False):

        if len(subfolders) != 0:
            continue
        print(f"現在のフォルダは{current}です。")
        print(f"サブフォルダは{subfolders}です。")
        print(f"サブファイルは{subfiles}です。")
        print("-------------------------------------------------------")

        num = len(subfiles)
        n = 0
        for file_name in subfiles:
            save_all_frames(os.path.join(current, file_name), os.path.join(out_bass_path, 'images'), file_name)
            substract_audio(os.path.join(current, file_name), os.path.join(out_bass_path, 'audios'), file_name)
            n = n + 1
            print(100*n/num)

    f = open('error_list.txt', 'w')
    for x in error_list:
        print("{}\n".format(x), file=f)
    f.close()

    import msgpack
    with open('video_fps_info.msgpack', 'wb') as f:
        msgpack.pack(d, f)


if __name__ == '__main__':
    main()