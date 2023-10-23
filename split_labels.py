# @Time : 2023/10/17 10:57
# @Author : Chen Xie
# @Email : xiechen19@mails.jlu.edu.cn

"""
Split KITTI tracking training label from by sequence to by frame,
and the format of each label in tracking are transformed to the format of KITTI Detection.
"""
import os
import glob
import numpy as np


def show_category_before(txt_list):
    category_list = []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ')
                    category_list.append(labeldata[2])
        except IOError as ioerr:
            print('File error:' + str(ioerr))
    print(set(category_list))


def show_category_after(txt_list):
    category_list = []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ')
                    category_list.append(labeldata[0])
        except IOError as ioerr:
            print('File error:' + str(ioerr))
    print(set(category_list))


def main(root, out_dir):
    label_path = os.path.join('./label_02/')
    label_list = os.listdir(label_path)

    txt_list_before = glob.glob(label_path + '*.txt')
    start_frame = 0
    # each sequence
    for file_name in label_list:
        label_file = os.path.join(label_path, file_name)

        img_path = os.path.join(root, 'image_02', file_name.split('.')[0])
        max_frame = len(os.listdir(img_path))

        # arr = np.loadtxt(label_file)
        with open(label_file) as f:
            lines = f.readlines()
            lines = [line.split(' ') for line in lines]

        frames = []
        for line in lines:
            frames.append(int(line[0]))
        frames = np.array(frames)

        # each frame
        for i in range(0, max_frame):
            idx = np.where(frames == i)[0]
            label_one_frame = [lines[id] for id in idx]

            start_id = start_frame + i
            save_file = os.path.join(out_dir, str(start_id).zfill(6) + '.txt')
            save_file = open(save_file, 'w')

            new_txt = []
            # each line in one frame
            for line in label_one_frame:
                # integrate class
                if line[2] in ['Truck', 'Van', 'Tram']:
                    line[2] = line[2].replace(line[2], 'Car')
                if line[2] in ['Person_sitting', 'Person']:
                    line[2] = line[2].replace(line[2], 'Pedestrian')
                if line[2] == 'DontCare':
                    continue
                if line[2] == 'Misc':
                    continue
                new_txt.append(line)

            # extract labels as KITTI detection format
            for line in new_txt:
                infos = '%s %d %d %f %f %f %f %f %f %f %f %f %f %f %f\n' % (
                    line[2], 0, int(line[4]), float(line[5]), float(line[6]), float(line[7]),
                    float(line[8]), float(line[9]), float(line[10]), float(line[11]), float(line[12]),
                    float(line[13]), float(line[14]), float(line[15]), float(line[16]))
                save_file.write(infos)

            save_file.close()

        start_frame += max_frame

    txt_list_after = glob.glob(out_dir + '*.txt')

    show_category_before(txt_list_before)
    show_category_after(txt_list_after)


if __name__ == '__main__':
    root = 'D:/dataset/KITTI/tracking/training'
    out_dir = 'D:/dataset/KITTI/tracking_for_detection/training/label_2/'
    main(root, out_dir)
