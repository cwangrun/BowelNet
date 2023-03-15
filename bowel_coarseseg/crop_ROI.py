from __future__ import division
import numpy as np
from glob import glob
import os.path
import nibabel as nib
import scipy
from cc3d import connected_components


name_to_label = {'rectum': 1, 'sigmoid': 2, 'colon': 3, 'small': 4, 'duodenum': 5}


def img2box(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax


def normalize_volume(img):
    img_array = (img - img.min()) / (img.max() - img.min())
    return img_array


def crop_ROI_using_coarse_seg_mask(mask_ori, bowel_name):
    assert bowel_name in name_to_label.keys()

    pos_label = name_to_label[bowel_name]
    mask = mask_ori.copy()
    mask[mask != pos_label] = 0
    mask[mask == pos_label] = 1

    # use the largest connected region to eliminate small noisy points, or use the mask directly
    labels_out, N = connected_components(mask, connectivity=26, return_N=True)
    numPix = []
    for segid in range(1, N + 1):
        numPix.append([segid, (labels_out == segid).astype(np.int8).sum()])
    numPix = np.array(numPix)

    if len(numPix) != 0:
        max_connected_image = np.int8(labels_out == numPix[np.argmax(numPix[:, 1]), 0])
        min_x, max_x, min_y, max_y, min_z, max_z = img2box(max_connected_image)
    else:
        print('coarse stage does not detect the organ, will skip this case')
        return (None, None), (None, None), (None, None), None

    # extend
    x_extend, y_extend, z_extend = (30, 30, 30)

    max_x = min(max_x + x_extend, mask.shape[0])
    max_y = min(max_y + y_extend, mask.shape[1])
    max_z = min(max_z + z_extend, mask.shape[2])
    min_x = max(min_x - x_extend, 0)
    min_y = max(min_y - y_extend, 0)
    min_z = max(min_z - z_extend, 0)

    return (min_x, max_x), (min_y, max_y), (min_z, max_z), mask


if __name__ == '__main__':

    imgs = glob('/mnt/c/chong/data/Bowel/crop_ori_res/*/*.nii.gz')

    bowel_name = 'rectum'
    # bowel_name = 'sigmoid'
    # bowel_name = 'colon'
    # bowel_name = 'small'
    # bowel_name = 'duodenum'

    for iii, img_file in enumerate(imgs):

        img_info = nib.load(img_file)
        img = img_info.get_fdata()
        img = np.clip(img, -1024, 1000)
        img = normalize_volume(img)

        coarse_mask_file = img_file.replace('image.nii.gz', 'masks_partial_C5.nii.gz')
        assert os.path.isfile(coarse_mask_file)
        coarse_mask = nib.load(coarse_mask_file).get_fdata()
        coarse_mask = scipy.ndimage.interpolation.zoom((coarse_mask).astype(np.float32),
                                                       zoom=np.array(img.shape) / np.array(coarse_mask.shape),
                                                       mode='nearest',
                                                       order=0)  # order = 0 nearest interpolation
        coarse_mask = np.round(coarse_mask).astype(np.int32)

        crop_x, crop_y, crop_z, mask_return = crop_ROI_using_coarse_seg_mask(coarse_mask, bowel_name)

        if mask_return is None:
           continue

        img = img[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]
        mask_return = mask_return[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]

        save_dir = os.path.dirname(img_file.replace('crop_ori_res', 'crop_ori_res_' + bowel_name))
        os.makedirs(save_dir, exist_ok=True)
        ############################################################################################
        crop_coords_patient = [crop_x[0], ',', crop_x[1], ',', crop_y[0], ',', crop_y[1], ',', crop_z[0], ',', crop_z[1]]
        with open(save_dir + '/crop_info_' + bowel_name + '.txt', 'w') as file:
            file.writelines(map(lambda x: str(x), crop_coords_patient))
            file.close()
        nib.save(nib.Nifti1Image(img, header=img_info.header, affine=img_info.affine), os.path.join(save_dir, 'image_crop.nii.gz'))
        nib.save(nib.Nifti1Image(mask_return, header=img_info.header, affine=img_info.affine), os.path.join(save_dir, 'masks_partial_C5_crop.nii.gz'))
        ############################################################################################

        # print(iii, crop_x[1] - crop_x[0], crop_y[1] - crop_y[0], crop_z[1] - crop_z[0], img_file)

