#!/usr/bin/env python
from __future__ import absolute_import, print_function
import os
import sys
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data_process_func import *
import os
import SimpleITK as sitk

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
    data_root = '/nas/leiwenhui/Data/HaN_structseg' # change it to your dataroot
    modelist = ['test']
    imgname = 'data.nii.gz'
    norm_img_save_name = 'rdata.nii.gz'
    labelname = 'label.nii.gz'
    label_save_name = 'rlabel.nii.gz'
    target_spacing = [3,1,1]
    thresh_lis = [-500, -100, 200, 1500]
    norm_lis = [0,0.2,0.8,1]

    for mode in modelist:
        filelist =os.listdir(os.path.join(data_root, mode))
        for file in filelist:
            normdata_save_path = os.path.join(data_root, mode, file, norm_img_save_name)
            data_path = os.path.join(data_root, mode, file, imgname)
            data = sitk.ReadImage(data_path)
            img = sitk.GetArrayFromImage(data).astype(np.float32)
            norm_img = img_multi_thresh_normalized(img, thresh_lis=thresh_lis, norm_lis=norm_lis)
            norm_data = sitk.GetImageFromArray(norm_img)
            norm_data.SetSpacing(data.GetSpacing())
            norm_data.SetOrigin(data.GetOrigin())
            norm_data.SetDirection(data.GetDirection())
            sitk.WriteImage(norm_data, normdata_save_path)
            print('sucessfully save', file)
    print('---done!')