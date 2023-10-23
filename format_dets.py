# @Time : 2023/7/18 16:36
# @Author : Chen Xie
# @Email : xiechen19@mails.jlu.edu.cn

import os
import pickle
import numpy as np

# -----------------------------根据mmdet3d输出的pkl文件，按照tracking序列进行检测结果存储-----------------------------
detector_3d = 'pv_rcnn(train_using_objectdata)'  # todo
mode = 'training'  # todo

root = './detections'
pkl_path = os.path.join(root, detector_3d, mode, 'pred_instances_3d.pkl')

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

    kitti_root = 'D:/dataset/KITTI/tracking'
    kitti_root = os.path.join(kitti_root, mode, 'velodyne')
    seq_list = os.listdir(kitti_root)
    sample_num = 0
    for i in range(0, len(seq_list)):
        seq_id = str(i).zfill(4)
        sample_root = os.path.join(kitti_root, seq_id)
        id_list = os.listdir(sample_root)

        sample_num_last = sample_num
        sample_num += len(id_list)

        cars, peds = [], []
        # 提取一个序列的全部样本信息, 共15个元素
        # frame, cls, alpha, bbox[4], dimensions(lhw)[3], location(xyz)[3], ry, score
        data_tmp = data[sample_num_last: sample_num]
        assert len(id_list) == len(data_tmp), "length of prediction data should equal to the length of samples"

        for j, id in enumerate(id_list):
            frame = int(id.split('.')[0])
            for k in range(0, len(data_tmp[j]['name'])):
                if data_tmp[j]['name'][k] == 'Car':
                    arr_tmp = np.empty([15])
                    arr_tmp[0] = frame
                    arr_tmp[1] = 0  # Car-->0
                    arr_tmp[2] = data_tmp[j]['alpha'][k]
                    arr_tmp[3:7] = data_tmp[j]['bbox'][k]
                    arr_tmp[7:10] = data_tmp[j]['dimensions'][k]
                    arr_tmp[10:13] = data_tmp[j]['location'][k]
                    arr_tmp[13] = data_tmp[j]['rotation_y'][k]
                    arr_tmp[14] = data_tmp[j]['score'][k]
                    cars.append(arr_tmp)

                elif data_tmp[j]['name'][k] == 'Pedestrian':
                    arr_tmp = np.empty([15])
                    arr_tmp[0] = frame
                    arr_tmp[1] = 1  # Pedestrian-->1
                    arr_tmp[2] = data_tmp[j]['alpha'][k]
                    arr_tmp[3:7] = data_tmp[j]['bbox'][k]
                    arr_tmp[7:10] = data_tmp[j]['dimensions'][k]
                    arr_tmp[10:13] = data_tmp[j]['location'][k]
                    arr_tmp[13] = data_tmp[j]['rotation_y'][k]
                    arr_tmp[14] = data_tmp[j]['score'][k]
                    peds.append(arr_tmp)

        arr_car = np.array(cars)
        arr_ped = np.array(peds)

        # 分别将car, pedestrian的结果存入对应的路径
        save_car_path = os.path.join(root, detector_3d, mode, 'Car')
        save_ped_path = os.path.join(root, detector_3d, mode, 'Pedestrian')
        if not os.path.exists(save_car_path):
            os.makedirs(save_car_path)
        if not os.path.exists(save_ped_path):
            os.makedirs(save_ped_path)

        save_car = os.path.join(save_car_path, str(i).zfill(4) + '.txt')  # car dir
        save_ped = os.path.join(save_ped_path, str(i).zfill(4) + '.txt')  # pedestrians dir

        np.savetxt(save_car, arr_car, fmt="%.4f")  # 每个元素保留4位小数
        np.savetxt(save_ped, arr_ped, fmt="%.4f")
