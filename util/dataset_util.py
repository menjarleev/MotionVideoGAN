import os
from shutil import copyfile

def dataset_distribute(dataroot, gt, label, frames_per_folder, dest):
    dir_gt = os.path.join(dataroot, gt)
    dir_label = os.path.join(dataroot, label)
    gt_paths = sorted([f for f in os.listdir(dir_gt)])
    label_paths = sorted([f for f in os.listdir(dir_label)])
    if not os.path.exists(dest):
        os.makedirs(dest)
    assert len(gt_paths) == len(label_paths)
    for i in range(len(gt_paths)):
        dest_folder_gt = os.path.join(dest, gt, "{:0>5d}".format(i // frames_per_folder + 1))
        dest_folder_label = os.path.join(dest, label, "{:0>5d}".format(i // frames_per_folder + 1))
        if not os.path.exists(dest_folder_gt):
            os.makedirs(dest_folder_gt)
        if not os.path.exists(dest_folder_label):
            os.makedirs(dest_folder_label)
        source_gt = os.path.join(dir_gt, gt_paths[i])
        source_label = os.path.join(dir_label, label_paths[i])
        target_gt = os.path.join(dest_folder_gt, gt_paths[i])
        target_label = os.path.join(dest_folder_label, label_paths[i])
        copyfile(source_gt, target_gt)
        copyfile(source_label, target_label)


