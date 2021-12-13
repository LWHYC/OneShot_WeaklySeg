#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import nibabel
import numpy as np
import random
from scipy import ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
import math


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


def search_file_in_folder_list(folder_list, file_name):
    """
    Find the full filename from a list of folders
    inputs:
        folder_list: a list of folders
        file_name:  filename
    outputs:
        full_file_name: the full filename
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if (os.path.isfile(full_file_name)):
            file_exist = True
            break
    if (file_exist == False):
        raise ValueError('{0:} is not found in {1:}'.format(file_name, folder))
    return full_file_name


def show_numpy(file_path, file_name, nrows, ncols, img_slip=1, dim=2):
    """
    show numpy image ,shape = h*w*l or w*l
    :param file_path:  e.g '/lyc/RTData/123.npy'
    :param file_name: list,contains the file name wanted
    :param mode:   '2D'or'3D'
    """
    img_list = [np.load(file_path + '/' + file) for file in file_name]
    f, plots = plt.subplots(nrows, ncols, figsize=(60, 60))
    if dim == 2:
        for i in range(len(img_list)):
            assert len(img_list[i].shape) == 2
            plots[divmod(i, nrows)[0], divmod(i, nrows)[1]].imshow(img_list[i])
            plots[divmod(i, nrows)[0], divmod(i, nrows)[1]].set_title(file_name[i])
        plt.show()

    elif dim == 3:
        for ii in np.arange(0, img_list[0].shape[0], img_slip):
            for iii in range(len(img_list)):
                assert len(img_list[iii].shape) == 2
                plots[divmod(iii, nrows)[0], divmod(iii, nrows)[1]].imshow(img_list[iii][ii])
                plots[divmod(iii, nrows)[0], divmod(iii, nrows)[1]].set_title(file_name[iii][ii])
            plt.show()


def save_array_as_nifty_volume(data, filename, transpose=True, pixel_spacing=[1, 1, 3]):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Channel, Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    if transpose:
        data = data.transpose(2, 1, 0)
    img = nibabel.Nifti1Image(data, None)
    img.header.set_zooms(pixel_spacing)
    nibabel.save(img, filename)


def save_list_as_nifty_volume(data, filename, pixel_spacing=[1, 1, 3]):
    """
    save a list as nifty image
    inputs:
        data: a list contains array [Channel, Depth, Height, Width]
        filename: the output file name
    :return:
    """
    data = np.asarray(data)
    img = nibabel.Nifti1Image(data)
    img.header.set_zooms(pixel_spacing)
    nibabel.save(img, filename)

def save_array_as_volume(data, filename, transpose=True, pixel_spacing=[1, 1, 3]):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Channel, Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    if transpose:
        data = data.transpose(2, 1, 0)
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(pixel_spacing)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(img)

def save_list_as_array_volume(data, filename):
    """
    save a list as numpy array
    :param data: list contains array
    :param filename: where to save
    :return:
    """
    data = np.asarray(data, dtype=np.float16)
    np.save(filename, data)


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if (source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume > 0] = convert_volume[mask_volume > 0]
    return out_volume


def fill_array(array, divisor):
    """
    由于下采样操作，需要对输入图像进行填充，使其满足采样比例的整数倍
    :param array: np array: depth*length*height
    :param divisor: The shape of the output file can be divided by divisor
    :return:
    """
    shape = array.shape
    pad_num = [[0, divisor[i] - shape[i] % divisor[i]] for i in range(len(divisor))]
    pad_array = np.pad(array, pad_num, 'constant')
    return pad_array


def get_random_roi_sampling_center(input_shape, output_shape, sample_mode, bounding_box=None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'full': the entire roi should be inside the input volume
                     'valid': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if (sample_mode[i] == 'full'):  # 不同轴向的裁取方式不同,z轴为full,裁剪范围需全部在输入中
            if (bounding_box):
                x0 = bounding_box[i * 2];
                x1 = bounding_box[i * 2 + 1]
            else:
                x0 = 0;
                x1 = input_shape[i]
        else:
            if (bounding_box):
                x0 = bounding_box[i * 2] + int(output_shape[i] / 2)
                x1 = bounding_box[i * 2 + 1] - int(output_shape[i] / 2)
            else:
                x0 = int(output_shape[i] / 2)
                x1 = input_shape[i] - x0
        if (x1 <= x0):  # 如果输出大于输入,后期会随机填充或0填充
            centeri = int((x0 + x1) / 2)
        else:
            centeri = random.randint(x0, x1)  # 如输出小于输入,可在[x0,l-x0]范围内任选点
        center.append(centeri)
    return center


def get_bound_coordinate(file, pad=[0, 0, 0]):
    '''
    输出array非0区域的各维度上下界坐标+-pad
    :param file: groundtruth图,
    :param pad: 各维度扩充的大小
    :return: bound: [min,max]
    '''
    file_size = file.shape
    nonzeropoint = np.asarray(np.nonzero(file))  # 得到非0点坐标,输出为一个3*n的array，3代表3个维度，n代表n个非0点在对应维度上的坐标
    maxpoint = np.max(nonzeropoint, 1).tolist()
    minpoint = np.min(nonzeropoint, 1).tolist()
    for i in range(len(pad)):
        maxpoint[i] = min(maxpoint[i] + pad[i], file_size[i]-1)
        minpoint[i] = max(minpoint[i] - pad[i], 0)
    return [minpoint, maxpoint]


def labeltrans(labelpair, file):
    '''
    :param labelpair: labelpair list
    :param file: np array
    :return:
    '''
    newfile = np.zeros_like(file)
    for label in labelpair:
        newfile[np.where(file == label[0])] = label[1]
    return newfile


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


def load_origin_nifty_volume_as_array(filename):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
        zoomfactor:
    """
    img = nibabel.load(filename)
    pixelspacing = img.header.get_zooms()
    zoomfactor = list(pixelspacing)
    zoomfactor.reverse()
    data = img.get_data()
    data = data.transpose(2, 1, 0)

    return data, zoomfactor


def load_and_respacing_nifty_volume_as_array(filename, mode='img', target_spacing=1, order=3):
    img = nibabel.load(filename)
    pixelspacing = list(img.header.get_zooms())
    pixelspacing.reverse()
    zoomfactor = list(np.array(pixelspacing) / np.array(target_spacing))
    data = img.get_data()
    data = data.transpose(2, 1, 0)

    if mode != 'img':
        order = 0
    data = ndimage.zoom(data, zoom=zoomfactor, order=order)

    return data

def load_volume_as_array(filename, transpose=True, return_spacing=False, respacing=False, target_spacing=1,
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

    if not transpose:
        ct_scan = ct_scan.transpose(2, 1, 0)
    else:
        spacing.reverse()
    if respacing:
        zoomfactor = list(np.array(spacing) / np.array(target_spacing))
        if mode != 'image':
            order = 0
        ct_scan = ndimage.zoom(ct_scan, zoom=zoomfactor, order=order)
        spacing = target_spacing
    if return_spacing:
        return ct_scan, spacing
    else:
        return ct_scan

def img_normalized(file, upthresh=0, downthresh=0, norm=True, thresh=True):
    """
    :param file: np array
    :param upthresh:
    :param downthresh:
    :param norm: norm or not
    :return:
    """
    if thresh:
        assert upthresh > downthresh
        file[np.where(file > upthresh)] = upthresh
        file[np.where(file < downthresh)] = downthresh
    if norm:
        file = (file - downthresh) / (upthresh - downthresh)
    return file


def img_multi_thresh_normalized(file, thresh_lis=[0], norm_lis=[0], data_type = np.float32):
    """
    :param file: np array
    :param upthresh:
    :param downthresh:
    :param norm: norm or not
    :return:
    """
    new_file = np.zeros_like(file).astype(data_type)

    for i in range(1, len(thresh_lis)):
        mask = np.where((file < thresh_lis[i]) & (file >= thresh_lis[i - 1]))
        k = (norm_lis[i] - norm_lis[i - 1]) / (thresh_lis[i] - thresh_lis[i - 1])
        b = norm_lis[i - 1]
        new_file[mask] = file[mask] - thresh_lis[i - 1]
        new_file[mask] = k * new_file[mask] + b
    new_file[np.where(file >= thresh_lis[-1])] = norm_lis[-1]
    return new_file


def transpose_volumes(volumes, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volumes
    elif (slice_direction == 'sagittal'):
        tr_volumes = [np.transpose(x, (2, 0, 1)) for x in volumes]
    elif (slice_direction == 'coronal'):
        tr_volumes = [np.transpose(x, (1, 0, 2)) for x in volumes]
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volumes
    return tr_volumes


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
    one hot编码
    :param volume: label
    :param C: class number
    :return:
    '''
    shape = [class_number] + list(volume.shape)
    volume_one = np.eye(class_number)[volume.reshape(-1)].T
    volume_one = volume_one.reshape(shape)
    return volume_one


def convert_one_hot_to_multi_class(one_hot, class_num):
    """
    input size:1*class_num*h*w*l or class_num*h*w*l
    :param one_hot: the one hot coder array
    :return: h*w*l
    """
    one_hot = one_hot.squeeze()
    assert (class_num == one_hot.shape[0])
    img = np.ones(one_hot.shape[1::])
    for i in range(class_num):
        img += one_hot[i] * i
    return img

def Erosion_Multi_label(volume, structure,class_number):
    """
    Erosion an multi class label
    :param volume: the input label, an tensor
    :param zoom_factor: the zoom fatcor of z,x,y
    :param class_number: the number of classes
    :param order:  the order of the interpolation
    :return:   shape = zoom_factor*original shape z,x,y
    """
    volume_one = convert_to_one_hot(volume, class_number)
    volum_one_erosion = [1*ndimage.binary_erosion(volume_one[i + 1],structure=structure) for i in
                         range(class_number-1)] # (n-1)*z*x*y
    output = np.zeros_like(volum_one_erosion[0])
    for i in range(class_number - 1):
        output[np.where(volum_one_erosion[i]==1)] = (i + 1)
    return output

def Dilation_Multi_label(volume, structure,class_number):
    """
    Dilation an multi class label
    :param volume: the input label, an tensor
    :param zoom_factor: the zoom fatcor of z,x,y
    :param class_number: the number of classes
    :param order:  the order of the interpolation
    :return:   shape = zoom_factor*original shape z,x,y
    """
    volume_one = convert_to_one_hot(volume, class_number)
    volum_one_erosion = [1*ndimage.binary_dilation(volume_one[i + 1],structure=structure) for i in
                         range(class_number-1)] # (n-1)*z*x*y
    output = np.zeros_like(volum_one_erosion[0])
    for i in range(class_number - 1):
        output[np.where(volum_one_erosion[i]==1)] = (i + 1)
    return output

def extract_roi_from_volume(volume, in_center, output_shape, fill='random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D_train volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape
    if (fill == 'random'):
        output = np.random.normal(0, 1, size=output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x / 2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output


def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D_train/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D_train/4D volume
    """
    volume_shape = volume.shape
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if (center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x / 2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if (len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif (len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output_volume


def get_roi(volume, margin):
    """
    get the roi bounding box of a 3D_train volume
    inputs:
        volume: the input 3D_train volume
        margin: an integer margin along each axis
    output:
        [mind, maxd, minh, maxh, minw, maxw]: a list of lower and upper bound along each dimension
    """
    [d_idxes, h_idxes, w_idxes] = np.nonzero(volume)
    [D, H, W] = volume.shape
    mind = max(d_idxes.min() - margin, 0)
    maxd = min(d_idxes.max() + margin, D)
    minh = max(h_idxes.min() - margin, 0)
    maxh = min(h_idxes.max() + margin, H)
    minw = max(w_idxes.min() - margin, 0)
    maxw = min(w_idxes.max() + margin, W)
    return [mind, maxd, minh, maxh, minw, maxw]

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
            out_img = np.zeros_like(img)
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

def get_largest_two_component(img, print_info=False, threshold=None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D_train volume
        threshold: a size threshold
    outputs:
        out_img: the output volume
    """
    s = ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if (print_info):
        print('component size', sizes_list)
    if (len(sizes) == 1):
        out_img = img
    else:
        if (threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if (temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if (max_size2 * 10 > max_size1):
                component1 = (component1 + component2) > 0
            out_img = component1
    return out_img


def fill_holes(img):
    """
    filling small holes of a binary volume with morphological operations
    """
    neg = 1 - img
    s = ndimage.generate_binary_structure(3, 1)  # iterate structure
    labeled_array, numpatches = ndimage.label(neg, s)  # labeling
    sizes = ndimage.sum(neg, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component


def remove_external_core(lab_main, lab_ext):
    """
    remove the core region that is outside of whole tumor
    """

    # for each component of lab_ext, compute the overlap with lab_main
    s = ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(lab_ext, s)  # labeling
    sizes = ndimage.sum(lab_ext, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    new_lab_ext = np.zeros_like(lab_ext)
    for i in range(len(sizes)):
        sizei = sizes_list[i]
        labeli = np.where(sizes == sizei)[0] + 1
        componenti = labeled_array == labeli
        overlap = componenti * lab_main
        if ((overlap.sum() + 0.0) / sizei >= 0.5):
            new_lab_ext = np.maximum(new_lab_ext, componenti)
    return new_lab_ext


def binary_dice3d(s, g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert (len(s.shape) == 3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert (Ds == Dg and Hs == Hg and Ws == Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = 2.0 * s0 / (s1 + s2 + 1e-10)
    return dice


def make_overlap_weight(overlap_num):
    """
    考虑到网络感受野可能超过图像厚度，故子图边界预测结果相对不可信。
    在叠加时应考虑加权，对每张图中心区域预测结果给予高权重，边界低权重。
    :return:
    """

    if overlap_num % 2 == 0:
        weight = [1 / (1 + abs(i - overlap_num // 2 - 0.5)) for i in range(1, overlap_num + 1)]
    else:
        weight = [1 / (1 + abs(i - overlap_num // 2)) for i in range(1, overlap_num + 1)]

    return weight


def zoom_data(file, mode='img', zoom_factor=[1, 1, 1], class_number=0):
    """
    对数据进行插值并储存，
    :param data_root: 数据所在上层目录
    :param save_root: 存储的顶层目录
    :zoom_factor:   缩放倍数
    :return:
    """

    if mode == 'label':
        intfile = np.int16(file)
        # zoom_file = np.int16(resize_Multi_label_to_given_shape(intfile, zoom_factor, class_number, order=2))
        zoom_file = ndimage.interpolation.zoom(file, zoom_factor, order=0)
    elif mode == 'img':
        zoom_file = ndimage.interpolation.zoom(file, zoom_factor, order=3)
    else:
        KeyError('please choose img or label mode')
    return zoom_file