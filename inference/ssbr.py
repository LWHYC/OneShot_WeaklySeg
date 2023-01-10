# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import sys
import os
sys.path.append("..")
sys.path.append(os.path.realpath(__file__)) 
sys.path.append(os.path.dirname(os.path.realpath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from dataloaders.Torchio_dataloader import *
from util.train_test_func import *
from test_single.test import test_single_case
import numpy as np
from inference.localization_functions import *
from scipy.ndimage import morphology
from numpy.linalg import norm
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

def selfsuport_bundaryrefine(net, sample_batch, class_wanted, stride, test_patch_size):
    kernel_size = np.array([2,2,2])
    prediction = np.zeros(sample_batch['image'].shape).astype(np.int16)

    for class_n in range(len(class_wanted)):
        cur_class_wanted = class_wanted[class_n]
        extractclass = ExtractCertainClass(class_wanted=[cur_class_wanted])
        cropbound = CropBound(pad=[7,7,7], mode='coarseg')
        totensor = ToTensor()
        cur_sample = totensor(cropbound(extractclass(sample_batch.copy())))
        img_batch = cur_sample['image'].cpu().data.numpy().squeeze()
        minpoint = cur_sample['minpoint']
        maxpoint = cur_sample['maxpoint']
        if 'coarseg' in cur_sample.keys():
            coarseg = cur_sample['coarseg'].cpu().data.numpy().squeeze()
        else:
            coarseg = False

        img_batch = img_batch[np.newaxis, :].astype(np.float16)
        predic_all = test_single_case(net, img_batch, stride=stride, patch_size=test_patch_size)
        feature_map = predic_all['fine_feature'].astype(np.float16)
        feature_map = feature_map/(0.0001+norm(feature_map, axis=0))

        ero_bg_coarseg_small = morphology.binary_dilation(coarseg, np.ones(kernel_size)).astype(np.int16)
        ero_bg_coarseg_large = morphology.binary_dilation(coarseg, np.ones(2*kernel_size)).astype(np.int16)
        print(np.sum(ero_bg_coarseg_small), np.sum(ero_bg_coarseg_large))
        cur_coarseg = (ero_bg_coarseg_large-ero_bg_coarseg_small).reshape(-1)
        bg_feature = np.take(feature_map.reshape(feature_map.shape[0], -1), np.where(cur_coarseg==1)[0],axis=1).transpose(1,0)
        k_means = KMeans(n_clusters=5).fit(bg_feature)
        qu_bg_gl_temp = k_means.cluster_centers_/(0.0001+norm(k_means.cluster_centers_, axis=1)[:, np.newaxis])
        bg_sim_map = np.einsum('ij,jklm->iklm', qu_bg_gl_temp,feature_map).max(axis=0)


        ero_fg_coarseg_small = morphology.binary_erosion(coarseg, np.ones(kernel_size)).astype(np.int16)
        ero_fg_coarseg_large = morphology.binary_erosion(coarseg, np.ones(2*kernel_size)).astype(np.int16)
        print(np.sum(ero_fg_coarseg_small), np.sum(ero_fg_coarseg_large))
        cur_coarseg = ero_fg_coarseg_small-ero_fg_coarseg_large
        
        qu_fg_gl_temp = (feature_map*cur_coarseg[np.newaxis,:]).sum(axis=(1,2,3))/cur_coarseg.sum() # c
        fg_sim_map = (feature_map*qu_fg_gl_temp[:, np.newaxis, np.newaxis, np.newaxis]).sum(axis=0)
        unsure_region = ero_bg_coarseg_small-ero_fg_coarseg_small
        unsure_region[unsure_region>1]=0
        cur_coarseg = np.clip(ero_fg_coarseg_small + np.argmax(np.array([bg_sim_map, fg_sim_map]), axis=0)*unsure_region, 0, 1)
        prediction[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]] += cur_class_wanted*cur_coarseg

    return prediction



