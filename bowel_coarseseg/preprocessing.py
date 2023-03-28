import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
import SimpleITK as sitk
from glob import glob
from cc3d import connected_components
import numpy as np
import os, cv2
from scipy import ndimage
from skimage import morphology
from skimage.morphology import convex_hull_image, disk, binary_closing


def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def resample_img(image_file, out_spacing=[0.97, 0.97, 1.5], is_label=False):

    itk_image = sitk.ReadImage(image_file)

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    itk_new = resample.Execute(itk_image)

    return sitk.GetArrayFromImage(itk_new).transpose(2, 1, 0)


def select_largest_region(img_bin):

    # N is the number of connected components
    labels_out, N = connected_components(img_bin, connectivity=26, return_N=True)
    num_labels = []
    for segid in range(1, N + 1):
        extracted_image = labels_out * (labels_out == segid)
        num_labels.append(np.array([segid, extracted_image.sum()]))

    num_labels = np.array(num_labels)
    topk = np.argsort(num_labels, axis=0)[::-1]
    top1 = num_labels[topk[0][1]][0]

    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[labels_out == top1] = 255

    return largest_mask


def crop_body(image):

    image = ((image - image.min()) / (image.max() - image.min())) * 255
    image = image.astype("uint8")
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    thresh, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_body = select_largest_region(mask)   # remove CT bed

    # plt.imshow(image)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(mask_body)
    # plt.show()

    x_range, y_range = np.where(mask_body != 0)
    x_min, y_min = x_range.min(), y_range.min()
    x_max, y_max = x_range.max(), y_range.max()

    return x_min, x_max, y_min, y_max, mask_body


files = glob('/mnt/c/chong/data/Bowel/alldata/small_bowel/*/*/image.nii.gz')   # resampled data

for iii, image_file in enumerate(files):

    label_file = image_file.replace('image.nii.gz', 'masks.nii.gz')

    image_info = nib.load(image_file)
    image_arr = image_info.get_fdata()

    label_info = nib.load(label_file)
    label_arr = np.round(label_info.get_fdata()).astype(int)

    x, y, z = image_arr.shape

    image = image_arr[:, :, :].copy()
    image = np.clip(image, -1024, 1000)   # remove abnormal intensity

    image_mean = np.mean(image, axis=2)
    body_x_min, body_x_max, body_y_min, body_y_max, mask_body = crop_body(image_mean)

    xs = body_x_min - 0
    xe = body_x_max + 0
    ys = body_y_min - 20
    ye = body_y_max + 0

    xs = 0 if xs < 0 else xs
    xe = x if xe > x else xe

    ys = 0 if ys < 0 else ys
    ye = y if ye > y else ye

    image_body = image_arr[xs:xe, ys:ye, :]
    masks_body = label_arr[xs:xe, ys:ye, :]

    print(iii, 'crop_shape', image_body.shape, 'ori_shape', image_arr.shape, image_file)

    # save preprocessed data
    #######################################################################################################
    if not os.path.exists(os.path.dirname(image_file.replace('/alldata', '/crop_preprocessed'))):
        os.makedirs(os.path.dirname(image_file.replace('/alldata', '/crop_preprocessed')))

    mask_save_path = os.path.dirname(os.path.dirname(image_file)).replace('/alldata', '/crop_preprocessed')
    cv2.imwrite(mask_save_path + '/' + image_file.split('/')[-2] + '.jpg', mask_body.transpose())

    nib.save(nib.Nifti1Image(image_body, header=image_info.header, affine=image_info.affine), image_file.replace('/alldata', '/crop_preprocessed'))
    nib.save(nib.Nifti1Image(masks_body, header=label_info.header, affine=label_info.affine), label_file.replace('/alldata', '/crop_preprocessed'))
    #######################################################################################################

    # save downsampled data
    #######################################################################################################
    image_downsampled = image_body[::2, ::2, ::2]
    masks_downsampled = masks_body[::2, ::2, ::2]

    if not os.path.exists(os.path.dirname(image_file.replace('/alldata', '/crop_downsample'))):
        os.makedirs(os.path.dirname(image_file.replace('/alldata', '/crop_downsample')))

    nib.save(nib.Nifti1Image(image_downsampled, header=image_info.header, affine=image_info.affine), image_file.replace('/alldata', '/crop_downsample'))
    nib.save(nib.Nifti1Image(masks_downsampled, header=label_info.header, affine=label_info.affine), label_file.replace('/alldata', '/crop_downsample'))
    #######################################################################################################

    # break
