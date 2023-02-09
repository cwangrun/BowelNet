from matplotlib import pylab as plt
import nibabel as nib
from glob import glob
from cc3d import connected_components
import numpy as np
import os, cv2
import scipy
from scipy import ndimage
from skimage import morphology
from scipy.ndimage import gaussian_filter
from shutil import copy
import scipy.ndimage as ndi



def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



files = glob('/mnt/c/chong/data/Bowel/crop_stage1_ROI_small/*/*/*/masks_crop.nii.gz')



for iii, file in enumerate(files):


    # file = '/home/chong/Desktop/small bowel skeleton/masks_crop.nii.gz'


    mask_info = nib.load(file)
    mask_arr_ori = np.round(mask_info.get_fdata()).astype(np.uint8)

    skeleton = mask_arr_ori.copy()

    ####################
    # filtering for smoothing, larger sigma, more blur
    skeleton = ndi.gaussian_filter(skeleton.astype(np.float), sigma=(1.5, 1.5, 1.5), order=0, truncate=2.0)
    skeleton = (skeleton > 0.5).astype(np.uint8)
    #####################

    skeleton = morphology.skeletonize_3d(skeleton).astype(np.uint8)

    save_path = file.replace('.nii.gz', '_skele.nii.gz')
    skeleton_dilate = ndimage.binary_dilation(skeleton, structure=np.ones((3, 3, 5)))
    nib.save(nib.Nifti1Image(skeleton_dilate, header=mask_info.header, affine=mask_info.affine), save_path)


    pseudo_one_mask = np.ones_like(mask_arr_ori).astype(np.uint8)
    pseudo_one_mask[skeleton == 1] = 0
    skeleton_dist, skeleton_index = ndi.distance_transform_edt(pseudo_one_mask, return_indices=True)    # distance to the nearest zero point

    grid = np.indices(skeleton_dist.shape).astype(float)
    diff = grid - skeleton_index
    dist = np.sqrt(np.sum(diff ** 2, axis=0))

    direction_0 = np.divide(diff[0, ...], dist + 1e-7)   # avoid divide zero
    direction_1 = np.divide(diff[1, ...], dist + 1e-7)
    direction_2 = np.divide(diff[2, ...], dist + 1e-7)

    skeleton_dist[mask_arr_ori != 1] = 0
    direction_0[mask_arr_ori != 1] = 0
    direction_1[mask_arr_ori != 1] = 0
    direction_2[mask_arr_ori != 1] = 0

    save_path = file.replace('.nii.gz', '_skele_dist.nii.gz')
    nib.save(nib.Nifti1Image(skeleton_dist, header=mask_info.header, affine=mask_info.affine), save_path)

    save_path = file.replace('.nii.gz', '_skele_fluxx.nii.gz')
    nib.save(nib.Nifti1Image(direction_0, header=mask_info.header, affine=mask_info.affine), save_path)

    save_path = file.replace('.nii.gz', '_skele_fluxy.nii.gz')
    nib.save(nib.Nifti1Image(direction_1, header=mask_info.header, affine=mask_info.affine), save_path)

    save_path = file.replace('.nii.gz', '_skele_fluxz.nii.gz')
    nib.save(nib.Nifti1Image(direction_2, header=mask_info.header, affine=mask_info.affine), save_path)

    np_save = np.sqrt(direction_0 ** 2 + direction_1 ** 2 + direction_2 ** 2)
    nib.save(nib.Nifti1Image(np_save, header=mask_info.header, affine=mask_info.affine), file.replace('.nii.gz', '_skele_fluxmag.nii.gz'))


    break

