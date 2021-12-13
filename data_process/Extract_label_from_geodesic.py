#!/usr/bin/env python
import os
import sys
import argparse
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data_process.data_process_func import *
from multiprocessing import Pool
import os
import SimpleITK as sitk
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage import measure
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

def extract_label(file_path, file_name_ls, label_ls):
    file_ls = []
    for file_name in file_name_ls:
        data_path = os.path.join(file_path, file_name)
        data = sitk.ReadImage(data_path)
        img = sitk.GetArrayFromImage(data)
        file_ls.append(np.exp(10*img))
    
    file = np.array(file_ls)
    nfile = np.array(file_ls)
    for i in [1,2,3]:
        file[i] -= 0.3*nfile[0]+0.1*np.delete(nfile, (0,i), axis=0).sum(0)
    file[0] *= 0.5
    label = np.argmax(file, axis=0).astype(np.float32)
    label[label==3]=6
    label[label==2]=3
    for clabel in label_ls:
        nlabel = np.zeros_like(label)
        nlabel[label==clabel]=1
        nlabel = morphology.binary_erosion(nlabel, np.ones([5,7,7])).astype(np.int16)
        nlabel = get_largest_component(nlabel)
        nlabel = morphology.binary_dilation(nlabel, np.ones([4,6,6])).astype(np.int16)
        label[label==clabel]=0
        label[nlabel==1]=clabel
    save_path=os.path.join(file_path, 'geo_seg.nii.gz')
    ndata = sitk.GetImageFromArray(label)
    ndata.SetSpacing(data.GetSpacing())
    ndata.SetOrigin(data.GetOrigin())
    ndata.SetDirection(data.GetDirection())
    sitk.WriteImage(ndata, save_path)
    return


p = Pool(12)
root_path = '../../../Data/TCIA'
file_name_ls = ['geo_dis_back.nii.gz', 'geo_dis_spleen.nii.gz', 'geo_dis_liver.nii.gz', 'geo_dis_leftkidney.nii.gz']
label_ls = [1, 3, 6]
for fold in ['train', 'valid', 'test']:
    cur_ls = os.listdir(os.path.join(root_path, fold))
    for file in cur_ls:
        file_path = os.path.join(root_path, fold, file)
        #extract_label(file_path, file_name_ls, label_ls)
        p.apply_async(extract_label, args=(file_path, file_name_ls, label_ls, ))
        print(file)
p.close()
p.join()