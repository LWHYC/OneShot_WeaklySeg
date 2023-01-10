#!/usr/bin/env python
import os
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )



for dataset in ['train','test','valid']:
    file_root = '../data/HaN_structseg/{0:}'.format(dataset)

    image_tx=open("../config/data/HaN_structseg/{0:}_image.txt".format(dataset), "w")
    label_tx=open("../config/data/HaN_structseg/{0:}_label.txt".format(dataset), "w")

    file_list = os.listdir(file_root)
    file_list.sort()
    for ii in file_list:
        image_tx.writelines(os.path.join(file_root, ii, 'rdata.nii.gz')+"\n")
        label_tx.writelines(os.path.join(file_root, ii, 'rlabel.nii.gz')+"\n")
        
