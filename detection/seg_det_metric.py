#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
#os._exit(00)
import sys
sys.path.append(os.path.abspath(__file__))  #返回当前.py文件的绝对路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.ndimage.measurements import label
import torch
import torch.tensor
import torch.backends.cudnn as cudnn
from dataloaders.Position_multi_scale_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from data_process.data_process_func import *
from util.visualization.show_param import show_param
from prefetch_generator import BackgroundGenerator
from detection.detection_functions import *
import numpy as np

def extractclass(label, class_wanted):
    nlabel = np.zeros_like(label)
    for i in range(len(class_wanted)):
            nlabel[np.where(label==class_wanted[i])]=i+1
    return nlabel


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def test(data_root):#, label_wanted):
    # 2, load data
    mean_iou_lis = []
    mean_error_lis = []
    error_dis = []
    volume_num = len(os.listdir(data_root))
    for ognb in [1,2,3,4]:
        label_wanted = ognb
        print('lw', label_wanted)
        iou_sum = 0
        error_sum = np.zeros([2, 3])
        for volume_name in os.listdir(data_root):
            label_path = os.path.join(data_root, volume_name, 'crop_label.nii.gz')
            label = load_nifty_volume_as_array(label_path)
            label = extractclass(label, [1,3,6,7])
            predic_path = os.path.join(data_root, volume_name, 'crop_coarseg.nii.gz')
            predic = load_nifty_volume_as_array(predic_path)
            error_dis.append([])
            real_corner_cor = extract_certain_organ_cor(label, label_wanted=label_wanted,extreme_point_num=2)
            predic_corner_cor = extract_certain_organ_cor(predic, label_wanted=label_wanted,extreme_point_num=2)
            pred_iou = iou(real_corner_cor,  predic_corner_cor)
            iou_sum += pred_iou
            error = np.abs(np.asarray(real_corner_cor-predic_corner_cor)) *np.asarray([3,1,1])
            error_sum += error
        mean_iou = np.around(iou_sum/volume_num, decimals=3)
        mean_error = np.around(np.mean(error_sum/volume_num), decimals=2)
        mean_error_each = np.around(error_sum/volume_num, decimals=2)
        mean_iou_lis.append(mean_iou)
        mean_error_lis.append(mean_error)
        print('mean iou:',mean_iou, 'mean error:',mean_error, 'mean error each:',mean_error_each)
    print(np.mean(np.array(mean_iou_lis)), np.mean(np.array(mean_error_lis)))
    



if __name__ == '__main__':
    data_root = '/home/disk/LWH/Data/HaN/valid'
    test(data_root) 
