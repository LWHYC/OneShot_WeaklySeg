#!/usr/bin/env python
import os
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data_process.data_process_func import *
import SimpleITK as sitk

data_root = '../../../Data/TCIA/'
scribblename = 'os_scribble.nii.gz'
nscribblename = 'os_nscribble.nii.gz'
noisename = 'noise.nii.gz'
modelist = [ 'train', 'valid']
label_wanted_ls = [1,3,6,15]
noise_thresh_ls = [0.4,0.4,0.4,0.4]
precision_sum = 0
num = 0
for mode in modelist:
    filelist = os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        scribble_path = os.path.join(data_root, mode, filelist[ii], scribblename)
        scribble = sitk.ReadImage(scribble_path)
        scribble = sitk.GetArrayFromImage(scribble)
        noise_path = os.path.join(data_root, mode, filelist[ii], noisename)
        noise = sitk.ReadImage(noise_path)
        noise = sitk.GetArrayFromImage(noise)
        nscribble = np.zeros_like(scribble)
        for i in range(len(label_wanted_ls)):
            label_wanted = label_wanted_ls[i]
            thresh = noise_thresh_ls[i]
            cscribble = (scribble==label_wanted).astype(np.int16)
            cur_noise = noise*cscribble
            noise_mask = (cur_noise>thresh).astype(np.int16)
            print(i, noise_mask.sum())
            if noise_mask.sum()<=2:
                noise_mask = (cur_noise>cur_noise.mean()).astype(np.int16)
            cscribble*= noise_mask
            nscribble[cscribble==1]=label_wanted
        data = sitk.GetImageFromArray(nscribble)
        nscribble_path = os.path.join(data_root, mode, filelist[ii], nscribblename)
        sitk.WriteImage(data, nscribble_path)
        print(nscribble_path)

