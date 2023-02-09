#!/usr/bin/env python3
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.ndimage.filters import gaussian_filter
from datasets.BowelDatasetCoarseSeg import BowelCoarseSeg
from tools.loss import *
import os, math
import numpy as np
import nibabel as nib
import loc_model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # "1, 2"
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])


name_to_label = {'rectum': 1, 'sigmoid': 2, 'colon': 3, 'small': 4, 'duodenum': 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)    # 8 (2 GPU)
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--nEpochs', type=int, default=1500, help='total training epoch')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-8, type=float, metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--eval_interval', default=5, type=int, help='evaluate interval on validation set')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--deterministic', type=bool, default=True)
    args = parser.parse_args()



    print("build Bowel Localisation Network")
    model = loc_model.BowelLocNet(elu=False)

    model_path = "./exp/BowelLocNet.20230208_1536/partial_5C_dict_1300.pth"
    print('load checkpoint:', model_path)
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    model = nn.parallel.DataParallel(model)
    model.eval()


    print("loading fully_labeled dataset")
    fully_labeled_dir = '/mnt/c/chong/data/Bowel/crop_downsample/Fully_labeled_5C'
    testSet_fully_labeled = BowelCoarseSeg(fully_labeled_dir, mode="test", transform=False, dataset_name="fully_labeled", save_dir=None)

    print("loading small dataset")
    smallbowel_dir = '/mnt/c/chong/data/Bowel/crop_downsample/Smallbowel'
    testSet_small = BowelCoarseSeg(smallbowel_dir, mode="test", transform=False, dataset_name="smallbowel", save_dir=None)

    print("loading colon_sigmoid dataset")
    colon_sigmoid_dir = '/mnt/c/chong/data/Bowel/crop_downsample/Colon_Sigmoid/colon_sigmoid'
    testSet_colon_sigmoid = BowelCoarseSeg(colon_sigmoid_dir, mode="test", transform=False, dataset_name="colon_sigmoid", save_dir=None)


    patch_size = (64, 128, 128)
    overlap = (32, 64, 64)


    n_test_samples = 0.
    dice_score = 0.0
    with torch.no_grad():
        # for data_name in testSet_fully_labeled.imgs:
        for data_name in testSet_small.imgs:
        # for data_name in testSet_colon_sigmoid.imgs:

            n_test_samples = n_test_samples + 1

            img_info = nib.load(data_name)
            img = img_info.get_fdata()
            img = np.clip(img, -1024, 1000)
            img = normalize_volume(img)

            label = nib.load(data_name.replace('image.nii.gz', 'masks.nii.gz')).get_fdata()
            label = np.round(label)

            img_arr, label_arr, zs_pad, ze_pad = padding_z(img, label, min_z=100)
            img_arr, label_arr, xs_pad, xe_pad = padding_x(img_arr, label_arr,  min_x=160)
            img_arr, label_arr, ys_pad, ye_pad = padding_y(img_arr, label_arr,  min_y=160)

            img_arr = img_arr.transpose((2, 0, 1))
            label_arr = label_arr.transpose((2, 0, 1))

            data = torch.from_numpy(img_arr[np.newaxis, np.newaxis, :].astype(np.float32))
            target = torch.from_numpy(label_arr[np.newaxis, np.newaxis, :].astype(np.int32))

            data, target = Variable(data.cuda()), Variable(target.cuda())

            b, _, z, x, y = data.shape

            zs = list(range(0, z, patch_size[0] - overlap[0]))
            xs = list(range(0, x, patch_size[1] - overlap[1]))
            ys = list(range(0, y, patch_size[2] - overlap[2]))

            gaussian_map = torch.from_numpy(_get_gaussian(patch_size, sigma_scale=1. / 4)).cuda()
            output_all = torch.zeros((b, 6, z, x, y)).cuda()
            for zzz in zs:
                for xxx in xs:
                    for yyy in ys:
                        if xxx + patch_size[1] > x:
                            xxx = x - patch_size[1]
                        if yyy + patch_size[2] > y:
                            yyy = y - patch_size[2]
                        if zzz + patch_size[0] > z:
                            zzz = z - patch_size[0]
                        candidate_patch = data[:, :, zzz:zzz+patch_size[0], xxx:xxx+patch_size[1], yyy:yyy+patch_size[2]]
                        output = model(candidate_patch)   # b, 6, z, x, y   prob output
                        output_all[:, :, zzz:zzz+patch_size[0], xxx:xxx+patch_size[1], yyy:yyy+patch_size[2]] += output * gaussian_map
            # remove padded slice
            output_all = output_all[:, :, zs_pad:ze_pad, xs_pad:xe_pad, ys_pad:ye_pad]
            target     =     target[:, :, zs_pad:ze_pad, xs_pad:xe_pad, ys_pad:ye_pad]
            result = torch.argmax(output_all, dim=1, keepdim=True)


            # pos_cls = [1, 2, 3, 4, 5]  # fully labeled dataset
            pos_cls = [4]  # small bowel dataset
            # pos_cls = [2, 3]  # colon_sigmoid dataset


            dice_c = []
            for cls in pos_cls:
                dice_c.append(dice_similarity((result == cls).contiguous(), (target == cls).contiguous()).item())
            dice_c = np.array(dice_c)
            dice_score += dice_c

            print('Name: {}, Dice: {}'.format(data_name, dice_c))

            # np_save = result.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.int32)
            # nib.save(nib.Nifti1Image(np_save, header=img_info.header, affine=img_info.affine), data_name.replace('image.nii.gz', 'masks_partial_C5.nii.gz'))

        dice_score /= n_test_samples

        print('\nMean test: Dice: {}\n'.format(dice_score))


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def normalize_volume(img):
    img_array = (img - img.min()) / (img.max() - img.min())
    return img_array


def padding_z(img, label, min_z):
    x, y, z = img.shape
    if z >= min_z:
        return img, label, 0, z
    else:
        num_pad = min_z - z
        num_top_pad = num_pad // 2
        top_pad = np.zeros((x, y, num_top_pad), dtype=np.float64)
        bottom_pad = np.zeros((x, y, num_pad - num_top_pad), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=2)
        label = np.concatenate((bottom_pad, label, top_pad), axis=2)
        return img, label, bottom_pad.shape[2], img.shape[2] - top_pad.shape[2]


def padding_x(img, label, min_x):
    x, y, z = img.shape
    if x >= min_x:
        return img, label, 0, x
    else:
        num_pad = min_x - x
        num_top_pad = num_pad // 2
        top_pad = np.zeros((num_top_pad, y, z), dtype=np.float64)
        bottom_pad = np.zeros((num_pad - num_top_pad, y, z), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=0)
        label = np.concatenate((bottom_pad, label, top_pad), axis=0)
        return img, label, bottom_pad.shape[0], img.shape[0] - top_pad.shape[0]


def padding_y(img, label, min_y):
    x, y, z = img.shape
    if y >= min_y:
        return img, label, 0, y
    else:
        num_pad = min_y - y
        num_top_pad = num_pad // 2
        top_pad = np.zeros((x, num_top_pad, z), dtype=np.float64)
        bottom_pad = np.zeros((x, num_pad - num_top_pad, z), dtype=np.float64)
        img = np.concatenate((bottom_pad, img, top_pad), axis=1)
        label = np.concatenate((bottom_pad, label, top_pad), axis=1)
        return img, label, bottom_pad.shape[1], img.shape[1] - top_pad.shape[1]


if __name__ == '__main__':
    main()
