#!/usr/bin/env python
from __future__ import absolute_import, print_function
import os
import sys
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data_process import *
from multiprocessing import Pool
import os
import SimpleITK as sitk
from scipy import ndimage

def resample_and_normalize(data_save_path, normdata_save_path, label_save_path, data_path, label_path, target_spacing, thresh_lis, norm_lis):
    data = sitk.ReadImage(data_path)
    img = sitk.GetArrayFromImage(data).astype(np.float32)
    label = sitk.ReadImage(label_path)
    mask = sitk.GetArrayFromImage(label).astype(np.int8)
    class_num = np.max(mask)+1
    zoom_factor = np.array(data.GetSpacing())/np.array(target_spacing)
    zoom_img = resize_ND_volume_to_given_shape(img, zoom_factor, order=1)
    normzoom_img = img_multi_thresh_normalized(zoom_img, thresh_lis=thresh_lis, norm_lis=norm_lis)
    zoom_mask = resize_Multi_label_to_given_shape(mask, zoom_factor, class_number=class_num)
    zoom_data = sitk.GetImageFromArray(zoom_img)
    zoom_data.SetSpacing(data.GetSpacing())
    zoom_data.SetOrigin(data.GetOrigin())
    zoom_data.SetDirection(data.GetDirection())
    sitk.WriteImage(zoom_data, data_save_path)
    normzoom_data = sitk.GetImageFromArray(normzoom_img)
    normzoom_data.SetSpacing(data.GetSpacing())
    normzoom_data.SetOrigin(data.GetOrigin())
    normzoom_data.SetDirection(data.GetDirection())
    sitk.WriteImage(normzoom_data, normdata_save_path)
    zoom_label = sitk.GetImageFromArray(zoom_mask)
    zoom_label.SetSpacing(data.GetSpacing())
    zoom_label.SetOrigin(data.GetOrigin())
    zoom_label.SetDirection(data.GetDirection())
    sitk.WriteImage(zoom_label, label_save_path)
    return

if __name__ == '__main__':
    data_root = '../../../Data/StructSeg' # change it to your dataroot
    modelist = ['train', 'valid', 'test']
    imgname = 'data.nii.gz'
    img_save_name = 'rdata.nii.gz'
    norm_img_save_name = 'norm_rdata.nii.gz'
    labelname = 'label.nii.gz'
    label_save_name = 'rlabel.nii.gz'
    target_spacing = [3,1,1]
    thresh_lis = [-1000, -250, 200, 800]
    norm_lis = [0,0.3,0.8,1]

    for mode in modelist:
        filelist =os.listdir(os.path.join(data_root, mode))
        for file in filelist:
            data_save_path = os.path.join(data_root, mode, file, img_save_name)
            normdata_save_path = os.path.join(data_root, mode, file, norm_img_save_name)
            data_path = os.path.join(data_root, mode, file, imgname)
            label_save_path = os.path.join(data_root, mode, file, label_save_name)
            label_path = os.path.join(data_root, mode, file, labelname)
            resample_and_normalize(data_save_path, normdata_save_path, label_save_path, data_path, label_path, target_spacing, thresh_lis, norm_lis)
            print('sucessfully save', file)
    print('---done!')