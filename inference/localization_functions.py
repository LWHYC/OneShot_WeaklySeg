#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from os.path import abspath, join, dirname
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, join(abspath(dirname(__file__)), 'src'))
import  numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib
matplotlib.use('WebAgg')
from data_process.data_process_func import *
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

class Relative_distance(object):
    def __init__(self,network,out_mode='fc_position', feature_refine=False, judge_noise=False, distance_mode='linear', \
                center_patch_size=[8,8,8], distance_ratio=100):
        self.network = network
        self.center_patch_size = center_patch_size
        self.distance_ratio = distance_ratio
        self.distance_mode = distance_mode
        self.out_mode = out_mode
        self.feature_refine = feature_refine
        self.judge_noise = judge_noise

    def cal_support_position(self, support_patch):
        '''
        support_patch: [b*1*d*w*h] 
        '''
        self.support_patch = support_patch
        self.support_all = self.network(torch.from_numpy(support_patch).float().half())
        self.support_position = self.support_all[self.out_mode].cpu().numpy()
        if self.feature_refine:
            self.support_feature = self.support_all['feature'].cpu().numpy() #[n,c,d,w,h]
            self.shape = self.support_feature.shape
            self.support_center_feature = self.support_feature[:,:,self.shape[2]//2,self.shape[3]//2, \
                                                    self.shape[4]//2] #[n,c,4]
        if self.judge_noise:
            # support_feature = self.support_all['feature'].cpu().numpy()
            # self.shape = support_feature.shape
            self.support_center_feature_0 = self.support_all['feature0'].cpu().numpy() #[n,c]
            self.support_center_feature_1 = self.support_all['feature1'].cpu().numpy()
        if self.out_mode == 'position':
            if self.center_patch_size =='all':
                self.support_position = np.mean(self.support_position, axis=(2,3,4)) #[6,3]
            else:
                self.support_position = np.mean(crop_patch_around_center(self.support_position, r=self.center_patch_size), axis=(2,3,4)) #[6,3]

    def cal_RD(self, query_patch, mean=False):
        '''
        query_patch:[b*1*d*w*h]
        '''
        result = {}
        query_all = self.network(torch.from_numpy(query_patch).float().half())
        quer_position = query_all[self.out_mode].cpu().numpy()# [b, 3]
        if self.out_mode == 'position':
            if self.center_patch_size =='all':
                quer_position = np.mean(quer_position, axis=(2,3,4)) #[6,3]
            else:
                quer_position = np.mean(crop_patch_around_center(quer_position, r=self.center_patch_size), axis=(2,3,4))
        if self.feature_refine:
            querry_feature = query_all['feature'].cpu().numpy()#[b,c,d,w,h]
            feature_sim = []
            feature_sim_rela_pos = []
            for i in range(self.shape[0]):
                n_querry_feature = np.transpose(querry_feature[i], (1,2,3,0)).reshape(-1, self.shape[1]) #[d*w*h,c]
                n_support_feature = self.support_center_feature[i].squeeze().transpose(1,0) #[16,c]
                
                cos_sim = np.sum(cosine_similarity(n_querry_feature, n_support_feature), axis=1).squeeze().reshape(self.shape[2::])
                print(np.max(cos_sim))
                plt.figure(figsize=(10, 10))
                plt.subplot(131)
                plt.imshow(self.support_patch[i,0,8].astype(np.float32), cmap='gray')
                plt.title('support patch')
                plt.subplot(132)
                plt.imshow(query_patch[i,0,8].astype(np.float32), cmap='gray')
                plt.title('query patch')
                plt.subplot(133)
                plt.imshow(cos_sim[8].astype(np.float32))
                plt.title('cos sim')
                plt.show()
                plt.close()
                feature_sim.append(cos_sim)
                feature_sim_rela_pos.append(np.asarray(np.where(cos_sim==np.max(cos_sim))).squeeze())
            feature_sim = np.asarray(feature_sim)
            feature_sim_rela_pos = np.asarray(feature_sim_rela_pos)-np.asarray(self.shape[2::])//2
            result['feature_sim']=feature_sim
            result['feature_sim_rela_pos']=feature_sim_rela_pos.astype(np.float16)

        if mean:
            quer_position = np.mean(quer_position, axis=0)
        if self.distance_mode=='linear':
            relative_position = self.distance_ratio*(self.support_position-quer_position)
        elif self.distance_mode=='tanh':
            relative_position = self.distance_ratio*np.tanh(self.support_position-quer_position)
        else:
            raise ValueError('Please select a correct distance mode!!！')
        result['relative_position']=relative_position
        return result
    
    def cal_noise(self, query_patch):
        query_all = self.network(torch.from_numpy(query_patch).float().half())
        query_center_feature_0 = query_all['feature0'].cpu().numpy() #[n,c]
        query_center_feature_1 = query_all['feature1'].cpu().numpy()
        noise = []
        for i in range(query_center_feature_0.shape[0]):
            noise0 = dot(query_center_feature_0[i], self.support_center_feature_0[i])/(0.0001+norm(query_center_feature_0[i])*norm(self.support_center_feature_0[i]))
            noise1 = dot(query_center_feature_1[i], self.support_center_feature_1[i])/(0.0001+norm(query_center_feature_1[i])*norm(self.support_center_feature_1[i]))
            noise.append(noise1*noise0)
        return noise


def crop_patch_around_center(img, r):
    '''
    img: array, c*w*h*d
    r: list
    crop a patch around the center point with shape r
    '''
    if len(img.shape)==4:
        shape = img.shape[1::]
        patch = img[:, shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    elif len(img.shape)==3:
        shape = img.shape
        patch = img[shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    elif len(img.shape)==5:
        shape = img.shape[2::]
        patch = img[:, :, shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    return patch
            


def random_all(random_seed):
    random.seed(random_seed)  # 给定seed value,决定了后面的伪随机序列
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def pad(img, pad_size):
    img = np.pad(img, [(pad_size[0] // 2, pad_size[0] // 2), (pad_size[1] // 2, pad_size[1] // 2),
                (pad_size[2] // 2, pad_size[2] // 2)], mode='constant', constant_values=0)
    return img

