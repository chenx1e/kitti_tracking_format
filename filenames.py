# @Time : 2023/10/17 15:09
# @Author : Chen Xie
# @Email : xiechen19@mails.jlu.edu.cn

# @Time : 2023/10/17 10:57
# @Author : Chen Xie
# @Email : xiechen19@mails.jlu.edu.cn

"""
Split KITTI tracking training label from by sequence to by frame,
and the format of each label in tracking are transformed to the format of KITTI Detection.
"""
import os
import numpy as np


def main(root, mode, output_dir):
    image_path = os.path.join(root, mode, 'image_02')
    seqs = os.listdir(image_path)

    start_frame = 0
    file_name = []
    for seq in seqs:
        num_frame = len(os.listdir(os.path.join(image_path, seq)))

        velo_path = os.listdir(os.path.join(root, mode, 'velodyne', seq))
        file_names = [int(v.split('.')[0]) + start_frame for v in velo_path]
        file_name += file_names

        start_frame += num_frame

    save_file = os.path.join(output_dir, mode + '.txt')
    save_file = open(save_file, 'w')
    for id in file_name:
        infos = '%s\n' % (str(id).zfill(6))
        save_file.write(infos)


if __name__ == '__main__':
    root = 'D:/dataset/KITTI/tracking'
    mode = 'testing'

    output_dir = './sample_names'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(root, mode, output_dir)
