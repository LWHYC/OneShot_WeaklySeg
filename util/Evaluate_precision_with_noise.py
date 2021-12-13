#!/usr/bin/env python
import os
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data_process.data_process_func import *
import SimpleITK as sitk

if __name__ == '__main__':

    data_root = '../../../Data/HaN_structseg/'
    scribblename = 'o_scribble1.nii.gz'
    nscribblename = 'o_nscribble.nii.gz'
    labelname = 'rlabel.nii.gz'
    noisename = 'o_noise1.nii.gz'
    modelist = [ 'train','valid', 'test']
    label_wanted = 1
    precision_sum = 0
    num = 0
    for mode in modelist:
        filelist = os.listdir(os.path.join(data_root, mode))
        filenum = len(filelist)
        for ii in range(filenum):
            scribble_path = os.path.join(data_root, mode, filelist[ii], scribblename)
            scribble = sitk.ReadImage(scribble_path)
            scribble = sitk.GetArrayFromImage(scribble)
            label_path = os.path.join(data_root, mode, filelist[ii], labelname)
            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label)
            noise_path = os.path.join(data_root, mode, filelist[ii], noisename)
            noise = sitk.ReadImage(noise_path)
            noise = sitk.GetArrayFromImage(noise)
            nscribble = np.zeros_like(scribble)
            nscribble[scribble==label_wanted]=1
            nlabel = np.zeros_like(label)
            nlabel[label==label_wanted]=1
            nnoise = np.zeros_like(noise)
            nnoise[noise>0.5]=1
            nscribble*= nnoise
            precision = np.sum(nlabel*nscribble)/(np.sum(nscribble)+0.00001)
            precision_sum += precision
            print(scribble_path, precision)
            num+=1
        print('mean precision:', precision_sum/num)

