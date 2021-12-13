#!/usr/bin/env python
import os
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data_process.data_process_func import *
import SimpleITK as sitk

data_root = '../../../Data/TCIA/'
scribblename = 'o_nscribble2.nii.gz'
labelname = 'rex_label.nii.gz'
modelist = [ 'train', 'valid']
label_wanted_ls = [1,3,6]

for mode in modelist:
    filelist = os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for label_wanted in label_wanted_ls:
        precision_sum = 0
        num = 0
        for ii in range(filenum):
            scribble_path = os.path.join(data_root, mode, filelist[ii], scribblename)
            scribble = sitk.ReadImage(scribble_path)
            scribble = sitk.GetArrayFromImage(scribble)
            nscribble = np.zeros_like(scribble)
            nscribble[scribble==label_wanted]=1
            label_path = os.path.join(data_root, mode, filelist[ii], labelname)
            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label)
            nlabel = np.zeros_like(label)
            nlabel[label==label_wanted]=1
            precision = np.sum(nlabel*nscribble)/np.sum(nscribble)
            precision_sum += precision
            print(scribble_path, precision)
            num+=1
        print('mean precision:', precision_sum/num)

