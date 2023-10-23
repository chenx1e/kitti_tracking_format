The following scripts are used to organize KITTI Tracking data into KITTI Object format for object detection training:

- integrate_img.py: used to convert png files stored in tracking sequence to per-frame storage and rename them sequentially.
- integrate_pc.py: used to convert bin files stored in tracking sequence to per-frame storage and rename them sequentially.
- split_labels.py: used to convert labels stored in tracking sequence to per-frame storage, and convert classes ['Truck', 'Van', 'Tram'] to Car, ['Person_sitting', 'Person'] to Pedestrian, remove 'DontCare ' and 'Misc' lines.
- split_calib.py: Used to convert calib files stored by tracking sequence to stored by frame and renamed in order.



Other scripts:

- filenames.py: used to generate the full filenames of the training and validation sets and store them as txt files for mmdetection.
- format_dets.py: to organize the pkl files from mmdetection output and convert the prediction results stored by frames to be stored by sequence.