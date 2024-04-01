import os.path as osp
import numpy as np
from PIL import Image
import csv
import cv2
import os
import shutil

video_types = ["mp4"]

def sample_images(v_path, i_path, video_type, frame_rate, block_len):
    if video_type not in video_types:
        return
    cap = cv2.VideoCapture(v_path)
    frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if block_len >= int(frames_num / frame_rate):
        block_len = int(frames_num / frame_rate)
    start_idx = int(np.random.random() * (int(frames_num / frame_rate) - block_len)) * frame_rate
    #fps = cap.get(cv2.CAP_PROP_FPS)
    for idx in range(start_idx, start_idx + block_len * frame_rate, frame_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imwrite(osp.join(i_path, str(idx).zfill(8)+'.png'), frame)

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     if frame_count % frame_rate == 0 and int(frame_count / frame_rate) > start_idx and block_count <= block_len:
    #         cv2.imwrite(osp.join(i_path, str(frame_count).zfill(8)+'.png'), frame)
    #         block_count += 1
    #     elif block_count > block_len:
    #         break
    #     frame_count += 1
    cap.release()
    

def videos2images(video_folder, image_folder, frame_rate, block_len):
    video_type_folders = os.listdir(video_folder)
    for i, vt_folder in enumerate(video_type_folders):
        vt_path = osp.join(video_folder, vt_folder)
        it_path = osp.join(image_folder, vt_folder)
        if osp.isdir(vt_path):
            video_names = os.listdir(vt_path)
            for j, v_name in enumerate(video_names):
                file_sufix = v_name.split('.')[-1]
                v_path = osp.join(vt_path, v_name)
                i_path = osp.join(it_path, v_name[:-len(file_sufix)-1])
                print(v_path)
                print("sampling", i, "video_type from", len(video_type_folders), j, "videos from", len(video_names))
                if osp.exists(i_path):
                    if len(os.listdir(i_path)) >= block_len:
                        continue
                    else:
                        print(len(os.listdir(i_path)), "file exists in path", i_path)
                else:
                    os.makedirs(i_path)
                sample_images(v_path, i_path, file_sufix, frame_rate, block_len)


if __name__ == '__main__':
    video_folder = "/data/home/hao/OpenGenSurgery/surgical_dataset_videos"
    image_folder = "/data/home/hao/OpenGenSurgery/surgical_dataset_images_framerate2_block2000"
    videos2images(video_folder, image_folder, 2, 2000)