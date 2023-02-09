#!/usr/bin/env python3
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
from tools.loss import *
from tools import utils
import os, math
import nibabel as nib
import seg_model
from datasets.BowelDatasetFineSeg import BowelFineSeg


os.environ['CUDA_VISIBLE_DEVICES'] = '1'   # "1, 2"
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])


name_to_label = {'rectum': 1, 'sigmoid': 2, 'colon': 3, 'small': 4, 'duodenum': 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)    # 4 (single GPU)
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--nEpochs_base', type=int, default=501, help='total epoch number for base segmentor')
    parser.add_argument('--nEpochs_meta', type=int, default=201, help='total epoch number for meta segmentor')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float, metavar='W', help='weight decay')
    parser.add_argument('--eval_interval', default=1, type=int, help='evaluate interval on validation set')
    parser.add_argument('--temp', default=0.7, type=float, help='temperature for meta segmentor')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--deterministic', type=bool, default=True)
    args = parser.parse_args()


    print("build BowelNet")
    model = seg_model.BowelNet(elu=False)

    model_path = "./BowelNet.20230207_1744/rectum_meta_195.pth"
    print('load checkpoint:', model_path)
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    model = nn.parallel.DataParallel(model)
    model.eval()


    # data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_small/'
    # bowel_name = 'small'

    # data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_colon/'
    # bowel_name = 'colon'

    # data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_sigmoid/'
    # bowel_name = 'sigmoid'

    # data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_duodenum/'
    # bowel_name = 'duodenum'

    data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_rectum/'
    bowel_name = 'rectum'


    print("loading test set")
    testSet = BowelFineSeg(root=data_dir, mode="test", transform=False, bowel_name=bowel_name, save_dir=None)

    if bowel_name == 'small':
        patch_size = (64, 192, 192)
        overlap = (32, 96, 96)
    if bowel_name == 'colon':
        patch_size = (64, 192, 192)
        overlap = (32, 96, 96)
    if bowel_name == 'sigmoid':
        patch_size = (64, 160, 160)
        overlap = (32, 80, 80)
    if bowel_name == 'duodenum':
        patch_size = (64, 160, 160)
        overlap = (32, 80, 80)
    if bowel_name == 'rectum':
        patch_size = (64, 96, 96)
        overlap = (32, 48, 48)

    n_test_samples = 0.
    dice_score_meta = 0.0
    dice_score_edge = 0.0
    dice_score_skele = 0.0
    assd_score_meta = 0.0
    assd_score_edge = 0.0
    assd_score_skele = 0.0
    with torch.no_grad():
        for iii, data_name in enumerate(testSet.imgs):

            n_test_samples = n_test_samples + 1

            img_info = nib.load(data_name)
            img = img_info.get_fdata()
            # img = np.clip(img, -1024, 1000)
            # img = normalize_volume(img)

            label = nib.load(data_name.replace('image_crop.nii.gz', 'masks_crop.nii.gz')).get_fdata()
            label = np.round(label)

            img_arr, label_arr, zs_pad, ze_pad = padding_z(img, label, min_z=80)
            img_arr, label_arr, xs_pad, xe_pad = padding_x(img_arr, label_arr, min_x=120)
            img_arr, label_arr, ys_pad, ye_pad = padding_y(img_arr, label_arr, min_y=120)
            
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
            out_meta_all = torch.zeros((b, 2, z, x, y)).cuda()
            out_edge_all = torch.zeros((b, 2, z, x, y)).cuda()
            out_skele_all = torch.zeros((b, 2, z, x, y)).cuda()
            for zzz in zs:
                for xxx in xs:
                    for yyy in ys:
                        if xxx + patch_size[1] > x:
                            xxx = x - patch_size[1]
                        if yyy + patch_size[2] > y:
                            yyy = y - patch_size[2]
                        if zzz + patch_size[0] > z:
                            zzz = z - patch_size[0]
                        candidate_patch = data[:, :, zzz:zzz + patch_size[0], xxx:xxx + patch_size[1], yyy:yyy + patch_size[2]]
                        out_meta, out_edge, out_skele = model(candidate_patch, 'meta')  # b, 2, z, x, y
                        prob_meta = F.softmax(out_meta, dim=1)
                        prob_edge = F.softmax(out_edge, dim=1)
                        prob_skele = F.softmax(out_skele, dim=1)
                        out_meta_all[:, :, zzz:zzz + patch_size[0], xxx:xxx + patch_size[1], yyy:yyy + patch_size[2]] += prob_meta * gaussian_map
                        out_edge_all[:, :, zzz:zzz + patch_size[0], xxx:xxx + patch_size[1], yyy:yyy + patch_size[2]] += prob_edge * gaussian_map
                        out_skele_all[:, :, zzz:zzz + patch_size[0], xxx:xxx + patch_size[1], yyy:yyy + patch_size[2]] += prob_skele * gaussian_map
            # remove padded slice
            out_meta_all = out_meta_all[:, :, zs_pad:ze_pad, xs_pad:xe_pad, ys_pad:ye_pad]
            out_edge_all = out_edge_all[:, :, zs_pad:ze_pad, xs_pad:xe_pad, ys_pad:ye_pad]
            out_skele_all = out_skele_all[:, :, zs_pad:ze_pad, xs_pad:xe_pad, ys_pad:ye_pad]
            target       =       target[:, :, zs_pad:ze_pad, xs_pad:xe_pad, ys_pad:ye_pad]

            dice_meta = dice_score_metric(out_meta_all, target)
            dice_edge = dice_score_metric(out_edge_all, target)
            dice_skele = dice_score_metric(out_skele_all, target)
            dice_score_meta += dice_meta
            dice_score_edge += dice_edge
            dice_score_skele += dice_skele

            assd_meta = utils.assd(torch.argmax(out_meta_all, dim=1).squeeze().permute(1, 2, 0).cpu().numpy(), target.squeeze().permute(1, 2, 0).cpu().numpy())  # x, y, z
            assd_edge = utils.assd(torch.argmax(out_edge_all, dim=1).squeeze().permute(1, 2, 0).cpu().numpy(), target.squeeze().permute(1, 2, 0).cpu().numpy())
            assd_skele = utils.assd(torch.argmax(out_skele_all, dim=1).squeeze().permute(1, 2, 0).cpu().numpy(), target.squeeze().permute(1, 2, 0).cpu().numpy())
            assd_score_meta += assd_meta
            assd_score_edge += assd_edge
            assd_score_skele += assd_skele

            print('Name: {}, Dice_meta:{:.4f}, Dice_edge:{:.4f}, Dice_skele:{:.4f}, ASSD_meta:{:.4f}, ASSD_edge:{:.4f}, ASSD_skele:{:.4f}'.
                  format(data_name, dice_meta, dice_edge, dice_skele, assd_meta, assd_edge, assd_skele))

            # np_save = torch.argmax(out_meta_all, dim=1).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.int32)
            # np_save[np_save == 1] = name_to_label[bowel_name]
            # nib.save(nib.Nifti1Image(np_save, header=img_info.header, affine=img_info.affine), data_name.replace('image_crop.nii.gz', 'masks_crop_meta.nii.gz'))
            #
            # np_save = torch.argmax(out_edge_all, dim=1).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.int32)
            # np_save[np_save == 1] = name_to_label[bowel_name]
            # nib.save(nib.Nifti1Image(np_save, header=img_info.header, affine=img_info.affine), data_name.replace('image_crop.nii.gz', 'masks_crop_edge.nii.gz'))
            #
            # np_save = torch.argmax(out_skele_all, dim=1).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.int32)
            # np_save[np_save == 1] = name_to_label[bowel_name]
            # nib.save(nib.Nifti1Image(np_save, header=img_info.header, affine=img_info.affine), data_name.replace('image_crop.nii.gz', 'masks_crop_skele.nii.gz'))

            # break

        dice_score_meta /= n_test_samples
        dice_score_edge /= n_test_samples
        dice_score_skele /= n_test_samples
        assd_score_meta /= n_test_samples
        assd_score_edge /= n_test_samples
        assd_score_skele /= n_test_samples

        print('\nMean test: Dice_meta:{:.4f}, Dice_edge:{:.4f}, Dice_skele:{:.4f}, '
              'ASSD_meta: {:.4f}, ASSD_edge: {:.4f}, ASSD_skele: {:.4f}\n'.
              format(dice_score_meta, dice_score_edge, dice_score_skele,
                     assd_score_meta, assd_score_edge, assd_score_skele))


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
