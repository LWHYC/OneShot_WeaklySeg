#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import nibabel
import numpy as np
import random
from scipy import ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
import time

def extract_certain_class(volume, wanted_class):
    nvolume = np.zeros_like(volume, dtype=np.uint8)
    for i in range(len(wanted_class)):
        nvolume[volume==wanted_class[i]]=i+1
    return nvolume

def recover_certain_class(volume, wanted_class):
    nvolume = np.zeros_like(volume, dtype=np.uint8)
    for i in range(len(wanted_class)):
        nvolume[volume==i+1]=wanted_class[i]
    return nvolume

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist

def mkdir(path):
    """
    创建path所给文件夹
    :param path:
    :return:
    """
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")

        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")

def resize_ND_volume_to_given_shape(volume, zoom_factor, order=3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    out_volume = ndimage.interpolation.zoom(volume, zoom_factor, order=order)
    return out_volume


def resize_Multi_label_to_given_shape(volume, zoom_factor, class_number, order=1):
    """
    resize an multi class label to a given shape
    :param volume: the input label, an tensor
    :param zoom_factor: the zoom fatcor of z,x,y
    :param class_number: the number of classes
    :param order:  the order of the interpolation
    :return:   shape = zoom_factor*original shape z,x,y
    """
    volume_one = convert_to_one_hot(volume, class_number)
    volum_one_reshape = [ndimage.interpolation.zoom(volume_one[i + 1], zoom_factor, order=order) for i in
                         range(class_number - 1)]
    output = np.zeros_like(volum_one_reshape[0])
    for i in range(class_number - 1):
        output = np.int8(np.rint(volum_one_reshape[i]))*(i + 1) + output
    return output

def convert_to_one_hot(volume, class_number):
    '''
    :param volume: label
    :param C: class number
    :return:
    '''
    shape = [class_number] + list(volume.shape)
    volume_one = np.eye(class_number)[volume.reshape(-1)].T
    volume_one = volume_one.reshape(shape)
    return volume_one.astype(np.uint8)

def get_bound_coordinate(file, pad=[0, 0, 0]):
    file_size = file.shape
    nonzeropoint = np.asarray(np.nonzero(file)) 
    maxpoint = np.max(nonzeropoint, 1).tolist()
    minpoint = np.min(nonzeropoint, 1).tolist()
    for i in range(len(pad)):
        maxpoint[i] = min(maxpoint[i] + pad[i], file_size[i]-1)
        minpoint[i] = max(minpoint[i] - pad[i], 0)
    return [minpoint, maxpoint]

def img_multi_thresh_normalized(file, thresh_lis=[0], norm_lis=[0], data_type = np.float32):
    """
    :param file: np array
    :param upthresh:
    :param downthresh:
    :param norm: norm or not
    :return:
    """
    new_file = np.zeros_like(file, dtype=data_type)

    for i in range(1, len(thresh_lis)):
        mask = np.where((file < thresh_lis[i]) & (file >= thresh_lis[i - 1]))
        k = (norm_lis[i] - norm_lis[i - 1]) / (thresh_lis[i] - thresh_lis[i - 1])
        b = norm_lis[i - 1]
        new_file[mask] = file[mask] - thresh_lis[i - 1]
        new_file[mask] = k * new_file[mask] + b
    new_file[np.where(file >= thresh_lis[-1])] = norm_lis[-1]
    return new_file

def save_array_as_nifty_volume(data, filename, transpose=False, pixel_spacing=[3, 1, 1], origin=[0,0,0]):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Channel, Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    if transpose:
        data = data.transpose(2, 1, 0)
    else:
        pixel_spacing = pixel_spacing[::-1]
        origin = origin[::-1]
    data = sitk.GetImageFromArray(data)
    data.SetOrigin(origin)
    data.SetSpacing(pixel_spacing)
    sitk.WriteImage(data, filename)
    # img = nibabel.Nifti1Image(data, None)
    # img.header.set_zooms(pixel_spacing)
    # nibabel.save(img, filename)

def load_volfile(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        vol = load_nifty_volume_as_array(filename)
        affine = None
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if add_feat_axis:
        vol = vol[np.newaxis, ...]

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol

def load_nifty_volume_as_array(filename, transpose=True, return_spacing=False, respacing=False, target_spacing=1,
                               mode='image', order=3):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    spacing = list(img.header.get_zooms())
    if transpose:
        data = data.transpose(2, 1, 0)
        spacing.reverse()
    if respacing:
        zoomfactor = list(np.array(spacing) / np.array(target_spacing))
        spacing = target_spacing
        if mode == 'image':
            data = ndimage.zoom(data, zoom=zoomfactor, order=order)
        elif mode == 'label':
            data = np.int8(data)
            data = np.int8(resize_Multi_label_to_given_shape(data, zoom_factor=zoomfactor, order=order, class_number=np.max(data)+1))
            # data = Erosion_Multi_label(data, np.ones([1,3,3]), class_number=np.max(data)+1)
            # data = np.int8(Dilation_Multi_label(data, np.ones([1, 4, 4]), class_number=np.max(data) + 1))
        else:
            ValueError('Please choose the right data mode! ( \'label\', or \'image\')')
    


    if return_spacing:
        return data, spacing
    else:
        return data

def load_volume_as_array(filename, transpose=True, return_spacing=False, return_origin=False, respacing=False, target_spacing=1,
                               mode='image', order=3):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
    """
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the spacing along each dimension
    spacing = list(itkimage.GetSpacing())

    origin = list(itkimage.GetOrigin())

    if not transpose:
        ct_scan = ct_scan.transpose(2, 1, 0)
    else:
        spacing.reverse()
        origin.reverse()
    if respacing:
        zoomfactor = list(np.array(spacing) / np.array(target_spacing))
        if mode != 'image':
            order = 0
        ct_scan = ndimage.zoom(ct_scan, zoom=zoomfactor, order=order)
        spacing = target_spacing
    

    if not return_spacing or not return_origin:
        data = ct_scan
    else:
        data = [ct_scan]

    if return_spacing:
        data.append(spacing)
    if return_origin:
        data.append(origin)
    return data

def get_largest_component(img, print_info=False, threshold=False):
    """
    Get the largest component of a binary volume
    inputs:
        img: the input 3D_train volume
        threshold: a size threshold
    outputs:
        out_img: the output volume
    """
    s = ndimage.generate_binary_structure(3, 1)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if (print_info):
        print('component size', sizes_list)
    if (len(sizes) <= 1):
        out_img = img
    else:
        if threshold:
            out_img = np.zeros_like(img, dtype=np.int8)
            for temp_size in sizes_list:
                if (temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            out_img = labeled_array == max_label1
    return out_img

def one_hot(img, nb_classes):
    hot_img = np.zeros([nb_classes]+list(img.shape))
    for i in range(nb_classes):
        hot_img[i][np.where(img == i)] = 1
    return hot_img