# @Time : 2023/10/18 10:06
# @Author : Chen Xie
# @Email : xiechen19@mails.jlu.edu.cn

import os
import shutil
import numpy as np


def main(root, mode, output_dir):
    img_path = os.path.join(root, mode, 'image_02')
    velo_path = os.path.join(root, mode, 'velodyne')

    seq_list = os.listdir(velo_path)
    start_frame = 0
    # each seq
    for seq in seq_list:
        img_files = os.listdir(os.path.join(img_path, seq))
        img_file_names = [int(img.split('.')[0]) for img in img_files]
        num_frame = np.max(np.array(img_file_names)) + 1

        pc_files = os.listdir(os.path.join(velo_path, seq))

        # each frame
        for file in pc_files:
            file_name = int(file.split('.')[0])
            correct_file_name = file_name + start_frame
            correct_file_name = str(correct_file_name).zfill(6) + '.bin'

            source_path = os.path.join(velo_path, seq, file)
            save_path = os.path.join(output_dir, correct_file_name)

            # copy the img file in source path to save path
            shutil.copyfile(source_path, save_path)

        start_frame += num_frame


if __name__ == '__main__':
    root = 'D:/dataset/KITTI/tracking'
    mode = 'testing'

    output_dir = os.path.join(root, mode + '_for_detection', 'velodyne')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(root, mode, output_dir)
