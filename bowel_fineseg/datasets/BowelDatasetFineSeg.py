from __future__ import division
import numpy as np
import torch
import torch.utils.data as data
from glob import glob
import os
import os.path
import nibabel as nib
import random
import cv2


name_to_label = {'rectum': 1, 'sigmoid': 2, 'colon': 3, 'small': 4, 'duodenum': 5}


def split_dataset(dir, current_test, test_fraction, bowel_name, save_dir):

    test_split = []
    train_split = []

    sub_folder = ['Fully_labeled_5C', 'Colon_Sigmoid', 'Smallbowel']

    assert bowel_name in name_to_label.keys()

    if bowel_name == 'rectum':
        all_volumes = sorted(glob(os.path.join(dir, "Fully_labeled_5C/*/*/image_crop.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)

    if bowel_name == 'sigmoid':
        all_volumes = sorted(glob(os.path.join(dir, "Colon_Sigmoid/*/*/image_crop.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)
        all_volumes = sorted(glob(os.path.join(dir, "Fully_labeled_5C/*/*/image_crop.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)

    if bowel_name == 'colon':
        all_volumes = sorted(glob(os.path.join(dir, "Colon_Sigmoid/*/*/image_crop.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)
        all_volumes = sorted(glob(os.path.join(dir, "Fully_labeled_5C/*/*/image_crop.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)

    if bowel_name == 'small':
        all_volumes = sorted(glob(os.path.join(dir, "Smallbowel/*/*/image_crop.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)
        all_volumes = sorted(glob(os.path.join(dir, "Fully_labeled_5C/*/*/image_crop.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)

    if bowel_name == 'duodenum':
        all_volumes = sorted(glob(os.path.join(dir, "Fully_labeled_5C/*/*/image_crop.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)

    if save_dir is not None:
        with open(os.path.join(save_dir, bowel_name + '_train_' + str(current_test) + ".txt"), 'w') as f:
            for i in train_split:
                f.write(i + '\n')

        with open(os.path.join(save_dir, bowel_name + '_test_' + str(current_test) + ".txt"), 'w') as f:
            for i in test_split:
                f.write(i + '\n')

    return train_split, test_split


def load_image_and_label(img_file, transform=False):

    img_info = nib.load(img_file)
    img = img_info.get_fdata()
    # img = np.clip(img, -1024, 1000)    # remove abnormal intensity
    # img = normalize_volume(img)

    label = nib.load(img_file.replace('image_crop.nii.gz', 'masks_crop.nii.gz')).get_fdata()
    label = np.round(label)

    skele_x = nib.load(img_file.replace('image_crop.nii.gz', 'skele_crop_fluxx.nii.gz')).get_fdata()  # skele_crop_fluxx.nii.gz
    skele_y = nib.load(img_file.replace('image_crop.nii.gz', 'skele_crop_fluxy.nii.gz')).get_fdata()  # skele_crop_fluxy.nii.gz
    skele_z = nib.load(img_file.replace('image_crop.nii.gz', 'skele_crop_fluxz.nii.gz')).get_fdata()  # skele_crop_fluxz.nii.gz
    edge = nib.load(img_file.replace('image_crop.nii.gz', 'edge_reg_crop.nii.gz')).get_fdata()        # edge_heatmap.nii.gz

    if transform:
        op = random.choice(['ori', 'rotate', 'crop'])
        if op == 'rotate':
            img, label, edge, skele_x, skele_y, skele_z = rotate(img, label, edge, skele_x, skele_y, skele_z, degree=random.randint(-10, 10))
        if op == 'crop':
            img, label, edge, skele_x, skele_y, skele_z = crop_resize(img, label, edge, skele_x, skele_y, skele_z, shift_size_x=10, shift_size_y=10)

    # zero padding in case that cropped ROI is smaller than patch size
    img_arr, label_arr, edge_arr, skele_x_arr, skele_y_arr, skele_z_arr = padding_z(img, label, edge, skele_x, skele_y, skele_z, min_z=80)
    img_arr, label_arr, edge_arr, skele_x_arr, skele_y_arr, skele_z_arr = padding_x(img_arr, label_arr, edge_arr, skele_x_arr, skele_y_arr, skele_z_arr, min_x=120)
    img_arr, label_arr, edge_arr, skele_x_arr, skele_y_arr, skele_z_arr = padding_y(img_arr, label_arr, edge_arr, skele_x_arr, skele_y_arr, skele_z_arr, min_y=120)

    return img_arr.transpose((2, 0, 1)), label_arr.transpose((2, 0, 1)), edge_arr.transpose((2, 0, 1)), skele_x_arr.transpose((2, 0, 1)), skele_y_arr.transpose((2, 0, 1)), skele_z_arr.transpose((2, 0, 1))


def padding_z(img, label, edge, skele_x, skele_y, skele_z, min_z):
    x, y, z = img.shape
    if z >= min_z:
        return img, label, edge, skele_x, skele_y, skele_z
    else:
        num_pad = min_z - z
        num_top_pad = num_pad // 2
        top_pad = np.zeros((x, y, num_top_pad), dtype=np.float64)
        bottom_pad = np.zeros((x, y, num_pad - num_top_pad), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=2)
        label = np.concatenate((bottom_pad, label, top_pad), axis=2)
        edge = np.concatenate((bottom_pad, edge, top_pad), axis=2)
        skele_x = np.concatenate((bottom_pad, skele_x, top_pad), axis=2)
        skele_y = np.concatenate((bottom_pad, skele_y, top_pad), axis=2)
        skele_z = np.concatenate((bottom_pad, skele_z, top_pad), axis=2)
        return img, label, edge, skele_x, skele_y, skele_z


def padding_x(img, label, edge, skele_x, skele_y, skele_z, min_x):
    x, y, z = img.shape
    if x >= min_x:
        return img, label, edge, skele_x, skele_y, skele_z
    else:
        num_pad = min_x - x
        num_top_pad = num_pad // 2
        top_pad = np.zeros((num_top_pad, y, z), dtype=np.float64)
        bottom_pad = np.zeros((num_pad - num_top_pad, y, z), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=0)
        label = np.concatenate((bottom_pad, label, top_pad), axis=0)
        edge = np.concatenate((bottom_pad, edge, top_pad), axis=0)
        skele_x = np.concatenate((bottom_pad, skele_x, top_pad), axis=0)
        skele_y = np.concatenate((bottom_pad, skele_y, top_pad), axis=0)
        skele_z = np.concatenate((bottom_pad, skele_z, top_pad), axis=0)
        return img, label, edge, skele_x, skele_y, skele_z


def padding_y(img, label, edge, skele_x, skele_y, skele_z, min_y):
    x, y, z = img.shape
    if y >= min_y:
        return img, label, edge, skele_x, skele_y, skele_z
    else:
        num_pad = min_y - y
        num_top_pad = num_pad // 2
        top_pad = np.zeros((x, num_top_pad, z), dtype=np.float64)
        bottom_pad = np.zeros((x, num_pad - num_top_pad, z), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=1)
        label = np.concatenate((bottom_pad, label, top_pad), axis=1)
        edge = np.concatenate((bottom_pad, edge, top_pad), axis=1)
        skele_x = np.concatenate((bottom_pad, skele_x, top_pad), axis=1)
        skele_y = np.concatenate((bottom_pad, skele_y, top_pad), axis=1)
        skele_z = np.concatenate((bottom_pad, skele_z, top_pad), axis=1)
        return img, label, edge, skele_x, skele_y, skele_z


def rotate(img_ori, label_ori, edge_ori, skele_x_ori, skele_y_ori, skele_z_ori, degree):
    height, width, depth = img_ori.shape
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    imgRotation = np.zeros_like(img_ori)
    labelRotation = np.zeros_like(label_ori)
    skele_x_Rotation = np.zeros_like(skele_x_ori)
    skele_y_Rotation = np.zeros_like(skele_y_ori)
    skele_z_Rotation = np.zeros_like(skele_z_ori)
    edgeRotation = np.zeros_like(edge_ori)
    for z in range(depth):
        imgRotation[:, :, z] = cv2.warpAffine(img_ori[:, :, z], matRotation, (width, height), borderValue=0)
        labelRotation[:, :, z] = cv2.warpAffine(label_ori[:, :, z], matRotation, (width, height))
        edgeRotation[:, :, z] = cv2.warpAffine(edge_ori[:, :, z], matRotation, (width, height))
        skele_x_Rotation[:, :, z] = cv2.warpAffine(skele_x_ori[:, :, z], matRotation, (width, height))
        skele_y_Rotation[:, :, z] = cv2.warpAffine(skele_y_ori[:, :, z], matRotation, (width, height))
        skele_z_Rotation[:, :, z] = cv2.warpAffine(skele_z_ori[:, :, z], matRotation, (width, height))
    labelRotation = np.round(labelRotation)
    return imgRotation, labelRotation, edgeRotation, skele_x_Rotation, skele_y_Rotation, skele_z_Rotation


def crop_resize(img_ori, label_ori, edge_ori, skele_x_ori, skele_y_ori, skele_z_ori, shift_size_x, shift_size_y):
    H, W, C = img_ori.shape
    x_small = np.random.randint(0, shift_size_x)
    x_large = np.random.randint(H - shift_size_x, H)
    y_small = np.random.randint(0, shift_size_y)
    y_large = np.random.randint(W - shift_size_y, W)

    imgCropresize = np.zeros_like(img_ori)
    labelCropresize = np.zeros_like(label_ori)
    skele_x_Cropresize = np.zeros_like(skele_x_ori)
    skele_y_Cropresize = np.zeros_like(skele_y_ori)
    skele_z_Cropresize = np.zeros_like(skele_z_ori)
    edgeCropresize = np.zeros_like(edge_ori)
    for z in range(C):
         imgCropresize[:, :, z] = cv2.resize(img_ori[x_small:x_large, y_small:y_large, z], (W, H))
         labelCropresize[:, :, z] = cv2.resize(label_ori[x_small:x_large, y_small:y_large, z], (W, H))
         edgeCropresize[:, :, z] = cv2.resize(edge_ori[x_small:x_large, y_small:y_large, z], (W, H))
         skele_x_Cropresize[:, :, z] = cv2.resize(skele_x_ori[x_small:x_large, y_small:y_large, z], (W, H))
         skele_y_Cropresize[:, :, z] = cv2.resize(skele_y_ori[x_small:x_large, y_small:y_large, z], (W, H))
         skele_z_Cropresize[:, :, z] = cv2.resize(skele_z_ori[x_small:x_large, y_small:y_large, z], (W, H))
    labelCropresize = np.round(labelCropresize)
    return imgCropresize, labelCropresize, edgeCropresize, skele_x_Cropresize, skele_y_Cropresize, skele_z_Cropresize


def normalize_volume(img):
    img_array = (img - img.min()) / (img.max() - img.min())
    return img_array


class BowelFineSeg(data.Dataset):
    def __init__(self, root='', transform=None, mode="train", test_fraction=0.2, bowel_name='', save_dir=''):

        assert bowel_name in name_to_label.keys()

        current_test = 5
        train_split, test_split = split_dataset(root, current_test, test_fraction, bowel_name, save_dir)

        if mode == "infer" or mode == "test":
            self.imgs = test_split
        else:
            self.imgs = train_split

        self.bowel_name = bowel_name
        self.mode = mode
        self.root = root
        self.patch_size = (64, 192, 192)

        if bowel_name == 'colon':
            self.patch_size = (64, 192, 192)

        if bowel_name == 'sigmoid':
            self.patch_size = (64, 160, 160)

        if bowel_name == 'duodenum':
            self.patch_size = (64, 160, 160)

        if bowel_name == 'rectum':
            self.patch_size = (64, 96, 96)

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        patch_size = self.patch_size
        img_name = self.imgs[index]

        img, label, edge, skele_x, skele_y, skele_z = load_image_and_label(img_name, transform=self.transform)
        z, x, y = img.shape

        if np.random.choice([False, False, False, True]):
            force_fg = True
        else:
            force_fg = False

        if force_fg:
            # sample a foreground region
            z_range, x_range, y_range = np.where(label != 0)
            z_min, x_min, y_min = z_range.min(), x_range.min(), y_range.min()
            z_max, x_max, y_max = z_range.max(), x_range.max(), y_range.max()
            center_z = random.randint(z_min, z_max)
            center_x = random.randint(x_min, x_max)
            center_y = random.randint(y_min, y_max)

            zs = center_z - patch_size[0] // 2   # start
            ze = center_z + patch_size[0] // 2   # end
            xs = center_x - patch_size[1] // 2
            xe = center_x + patch_size[1] // 2
            ys = center_y - patch_size[2] // 2
            ye = center_y + patch_size[2] // 2

            if zs < 0:
                zs = 0
                ze = zs + patch_size[0]
            if ze > z:
                ze = z
                zs = ze - patch_size[0]

            if xs < 0:
                xs = 0
                xe = xs + patch_size[1]
            if xe > x:
                xe = x
                xs = xe - patch_size[1]

            if ys < 0:
                ys = 0
                ye = ys + patch_size[2]
            if ye > y:
                ye = y
                ys = ye - patch_size[2]
        else:
            zs = random.randint(0, z - patch_size[0])
            ze = zs + self.patch_size[0]
            xs = random.randint(0, x - patch_size[1])
            xe = xs + self.patch_size[1]
            ys = random.randint(0, y - patch_size[2])
            ye = ys + self.patch_size[2]

        img_p = img[zs:ze, xs:xe, ys:ye][np.newaxis, :]
        label_p = label[zs:ze, xs:xe, ys:ye][np.newaxis, :]
        edge_p = edge[zs:ze, xs:xe, ys:ye][np.newaxis, :]
        skele_x_p = skele_x[zs:ze, xs:xe, ys:ye][np.newaxis, :]
        skele_y_p = skele_y[zs:ze, xs:xe, ys:ye][np.newaxis, :]
        skele_z_p = skele_z[zs:ze, xs:xe, ys:ye][np.newaxis, :]

        img_p = torch.from_numpy(img_p.astype(np.float32))
        label_p = torch.from_numpy(label_p.astype(np.int32))
        edge_p = torch.from_numpy(edge_p.astype(np.float32))
        skele_x_p = torch.from_numpy(skele_x_p.astype(np.float32))
        skele_y_p = torch.from_numpy(skele_y_p.astype(np.float32))
        skele_z_p = torch.from_numpy(skele_z_p.astype(np.float32))

        return img_p, label_p, edge_p, skele_x_p, skele_y_p, skele_z_p

