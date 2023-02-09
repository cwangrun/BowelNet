from matplotlib import pylab as plt
import nibabel as nib
from glob import glob
from cc3d import connected_components
import numpy as np
import os, cv2
import shutil
import scipy
from scipy import ndimage
from skimage import morphology
from scipy.ndimage import gaussian_filter
from shutil import copy



def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



files = glob('/mnt/c/chong/data/Bowel/crop_stage1_ROI_small/*/*/*/masks_crop.nii.gz')



for iii, file in enumerate(files):


    # file = '/home/chong/Desktop/small bowel skeleton/masks_crop.nii.gz'


    mask_info = nib.load(file)
    mask_arr_ori = np.round(mask_info.get_fdata()).astype(np.uint8)

    mask_arr = mask_arr_ori.copy()

    erosion_bin_map = ndimage.binary_erosion(mask_arr.astype(int), structure=np.ones((3, 3, 1)))
    # erosion_bin_map = ndimage.binary_erosion(mask_arr.astype(int), structure=np.ones((3, 3, 3)))

    edge_map = mask_arr - erosion_bin_map.astype(int)

    edge_heatmap = np.zeros_like(mask_arr).astype(np.float32)
    for z in range(mask_arr.shape[2]):
        gray = edge_map[:, :, z].astype(np.float32).copy()
        edge_heatmap[:, :, z] = gaussian_filter(gray, sigma=1.0, truncate=2.0)   # larger sigma, more blur

    save_path = file.replace('.nii.gz', '_edge_heatmap.nii.gz')
    nib.save(nib.Nifti1Image(edge_heatmap, header=mask_info.header, affine=mask_info.affine), save_path)

    save_path = file.replace('.nii.gz', '_edge.nii.gz')
    nib.save(nib.Nifti1Image(edge_map, header=mask_info.header, affine=mask_info.affine), save_path)

    break
