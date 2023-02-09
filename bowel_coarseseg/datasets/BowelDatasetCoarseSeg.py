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


name_to_label = {1: 'rectum', 2: 'sigmoid', 3: 'colon', 4: 'small', 5: 'duodenum'}


def split_dataset(dir, current_test, test_fraction, dataset_name, save_dir):

    test_split = []
    train_split = []

    sub_folder = ['Fully_labeled_5C', 'Colon_Sigmoid', 'Smallbowel']

    if dataset_name == 'fully_labeled':
        all_volumes = sorted(glob(os.path.join(dir, "*/*/image.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)

    if dataset_name == 'smallbowel':
        all_volumes = sorted(glob(os.path.join(dir, "*/*/image.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)

    if dataset_name == 'colon_sigmoid':
        all_volumes = sorted(glob(os.path.join(dir, "*/image.nii.gz")))
        test_num = int(len(all_volumes) * test_fraction)
        test_volumes = all_volumes[test_num * (current_test - 1): test_num * current_test]
        train_volumes = sorted(list(set(all_volumes) - set(test_volumes)))
        test_split.extend(test_volumes)
        train_split.extend(train_volumes)

    if save_dir is not None:
        with open(os.path.join(save_dir, dataset_name + '_train_' + str(current_test) + ".txt"), 'w') as f:
            for i in train_split:
                f.write(i + '\n')

        with open(os.path.join(save_dir, dataset_name + '_test_' + str(current_test) + ".txt"), 'w') as f:
            for i in test_split:
                f.write(i + '\n')

    return train_split, test_split


def load_image_and_label(img_file, transform=False):

    # img_file = '/mnt/c/chong/data/Bowel/crop_downsample/Colon_Sigmoid/colon_sigmoid/0005_male/image.nii.gz'

    img = nib.load(img_file).get_fdata()
    img = np.clip(img, -1024, 1000)   # remove abnormal intensity
    img = normalize_volume(img)

    label = nib.load(img_file.replace('image.nii.gz', 'masks.nii.gz')).get_fdata()
    label = np.round(label)

    if transform:
        op = random.choice(['ori', 'rotate', 'crop'])
        if op == 'rotate':
            img, label = rotate(img, label, degree=random.uniform(-10, 10))
        if op == 'crop':
            img, label = crop_resize(img, label, shift_size_x=10, shift_size_y=10)

    # zero padding
    img_arr, label_arr = padding_z(img, label, min_z=100)
    img_arr, label_arr = padding_x(img_arr, label_arr, min_x=160)
    img_arr, label_arr = padding_y(img_arr, label_arr, min_y=160)

    return img_arr.transpose((2, 0, 1)), label_arr.transpose((2, 0, 1))


def padding_z(img, label, min_z):
    x, y, z = img.shape
    if z >= min_z:
        return img, label
    else:
        num_pad = min_z - z
        num_top_pad = num_pad // 2
        top_pad = np.zeros((x, y, num_top_pad), dtype=np.float64)
        bottom_pad = np.zeros((x, y, num_pad - num_top_pad), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=2)
        label = np.concatenate((bottom_pad, label, top_pad), axis=2)
        return img, label


def padding_x(img, label, min_x):
    x, y, z = img.shape
    if x >= min_x:
        return img, label
    else:
        num_pad = min_x - x
        num_top_pad = num_pad // 2
        top_pad = np.zeros((num_top_pad, y, z), dtype=np.float64)
        bottom_pad = np.zeros((num_pad - num_top_pad, y, z), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=0)
        label = np.concatenate((bottom_pad, label, top_pad), axis=0)
        return img, label


def padding_y(img, label, min_y):
    x, y, z = img.shape
    if y >= min_y:
        return img, label
    else:
        num_pad = min_y - y
        num_top_pad = num_pad // 2
        top_pad = np.zeros((x, num_top_pad, z), dtype=np.float64)
        bottom_pad = np.zeros((x, num_pad - num_top_pad, z), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=1)
        label = np.concatenate((bottom_pad, label, top_pad), axis=1)
        return img, label


def rotate(img_ori, label_ori, degree):
    height, width, depth = img_ori.shape
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    imgRotation = np.zeros_like(img_ori)
    labelRotation = np.zeros_like(label_ori)
    for z in range(depth):
        imgRotation[:, :, z] = cv2.warpAffine(img_ori[:, :, z], matRotation, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        temp = label_ori[:, :, z]
        unique_labels = np.unique(temp)
        result = np.zeros_like(temp, temp.dtype)
        for i, c in enumerate(unique_labels):
            res_new = cv2.warpAffine((temp == c).astype(float), matRotation, (width, height), flags=cv2.INTER_NEAREST)
            result[res_new > 0.5] = c
        labelRotation[:, :, z] = result
    return imgRotation, labelRotation


def crop_resize(img_ori, label_ori, shift_size_x, shift_size_y):
    H, W, C = img_ori.shape
    x_small = np.random.randint(0, shift_size_x)
    x_large = np.random.randint(H - shift_size_x, H)
    y_small = np.random.randint(0, shift_size_y)
    y_large = np.random.randint(W - shift_size_y, W)

    imgCropresize = np.zeros_like(img_ori)
    labelCropresize = np.zeros_like(label_ori)
    for z in range(C):
        imgCropresize[:, :, z] = cv2.resize(img_ori[x_small:x_large, y_small:y_large, z], (W, H), interpolation=cv2.INTER_NEAREST)
        temp = label_ori[x_small:x_large, y_small:y_large, z]
        unique_labels = np.unique(temp)
        result = np.zeros((H, W), label_ori.dtype)
        for i, c in enumerate(unique_labels):
            res_new = cv2.resize((temp == c).astype(float), (W, H), interpolation=cv2.INTER_NEAREST)
            result[res_new > 0.5] = c
        labelCropresize[:, :, z] = result
    return imgCropresize, labelCropresize


def normalize_volume(img):
    img_array = (img - img.min()) / (img.max() - img.min())
    return img_array


class BowelCoarseSeg(data.Dataset):
    def __init__(self, root='', transform=None, mode="train", test_fraction=0.2, dataset_name='', save_dir=''):

        assert dataset_name in ["fully_labeled", "smallbowel", "colon_sigmoid"]

        current_test = 5
        train_split, test_split = split_dataset(root, current_test, test_fraction, dataset_name, save_dir)

        if mode == "infer" or mode == "test":
            self.imgs = test_split
        else:
            self.imgs = train_split

        self.mode = mode
        self.root = root
        self.patch_size = (64, 128, 128)   # z, x, y
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        patch_size = self.patch_size
        image_name = self.imgs[index]

        # image_name = '/mnt/c/chong/data/Bowel/crop_downsample/Smallbowel/abdomen/260719/image.nii.gz'
        # image_name = '/mnt/c/chong/data/Bowel/crop_downsample/Fully_labeled_5C/rectum_sigmoid_colon_small_duodenum/20210310-092019-464/image.nii.gz'
        # image_name = '/mnt/c/chong/data/Bowel/crop_downsample/Colon_Sigmoid/colon_sigmoid/0013_female/image.nii.gz'

        image, label = load_image_and_label(image_name, transform=self.transform)

        positive_labels = sorted(np.unique(label)[1:].astype(int))
        for pos in positive_labels:
            assert pos in [1, 2, 3, 4, 5], image_name + ", wrong class label!!!"

        # name_to_label = {1: 'rectum', 2: 'sigmoid', 3: 'colon', 4: 'small', 5: 'duodenum'}

        z, x, y = image.shape
        # fully labeled and Colon_Sigmoid dataset
        if 'rectum' in image_name or 'duodenum' in image_name or 'sigmoid' in image_name:

            if np.random.choice([False, False, False, True]):
                force_fg = True
            else:
                force_fg = False

            if force_fg:

                pos_labels = sorted(np.unique(label)[1:].astype(int))
                label_will_be_chosen = pos_labels + list(set(pos_labels) - set([3, 4]))   # double chance to sample the 3 bowels above
                label_chosen = random.choice(label_will_be_chosen)

                z_range, x_range, y_range = np.where(label == label_chosen)
                z_min, x_min, y_min = z_range.min(), x_range.min(), y_range.min()
                z_max, x_max, y_max = z_range.max(), x_range.max(), y_range.max()
                center_z = random.randint(z_min, z_max)
                center_x = random.randint(x_min, x_max)
                center_y = random.randint(y_min, y_max)

                zs = center_z - patch_size[0] // 2     # start
                ze = center_z + patch_size[0] // 2     # end
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

        else:   # small bowel dataset

            if np.random.choice([False, False, False, True]):
                force_fg = True
            else:
                force_fg = False

            if force_fg:
                z_range, x_range, y_range = np.where(label != 0)
                z_min, x_min, y_min = z_range.min(), x_range.min(), y_range.min()
                z_max, x_max, y_max = z_range.max(), x_range.max(), y_range.max()
                center_z = random.randint(z_min, z_max)
                center_x = random.randint(x_min, x_max)
                center_y = random.randint(y_min, y_max)

                zs = center_z - patch_size[0] // 2
                ze = center_z + patch_size[0] // 2
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

        data = torch.from_numpy(image[zs:ze, xs:xe, ys:ye][np.newaxis, :].astype(np.float32))
        mask = torch.from_numpy(label[zs:ze, xs:xe, ys:ye][np.newaxis, :].astype(np.int32))

        return data, mask, image_name

