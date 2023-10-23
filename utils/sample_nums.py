import os

# -----------------------------检查mmdet3d输出的样本数是否与数据集的样本数一致-----------------------------
kitti_root = 'D:/dataset/KITTI/tracking/testing/velodyne'
seq_list = os.listdir(kitti_root)
sample_num = 0
for i in range(0, len(seq_list)):
    seq_id = str(i).zfill(4)
    sample_root = os.path.join(kitti_root, seq_id)
    id_list = os.listdir(sample_root)
    sample_num += len(id_list)
print("KITTI tracking training样本数：", sample_num)

mmdet3d_root = './detections/3D/point_rcnn/testing/pred_instances_3d'
sample_num = len(os.listdir(mmdet3d_root))
print("mmdet3d输出的tracking training样本数：", sample_num)
