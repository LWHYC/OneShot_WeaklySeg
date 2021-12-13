#!/usr/bin/env python
import os
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data_process.data_process_func import *
import SimpleITK as sitk

data_root = '../../../Data/TCIA/'
scribblename = 'geo_seg.nii.gz'
labelname = 'rlabel.nii.gz'
modelist = [ 'train','valid', 'test']
label_wanted_ls = [1,3,6]

for mode in modelist:
    for label_wanted in label_wanted_ls:
        dice_sum = 0
        num = 0
        filelist = os.listdir(os.path.join(data_root, mode))
        filenum = len(filelist)
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
            dice = 2*np.sum(nlabel*nscribble)/(np.sum(nscribble)+np.sum(nlabel))
            dice_sum += dice
            print(scribble_path, dice)
            num+=1
        print(mode, label_wanted, 'mean dice:', dice_sum/num)

