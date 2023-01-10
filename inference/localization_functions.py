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
from data_process.data_process_func import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

class Dataloader_(Dataset):
    def __init__(self, config=None, num=None, transform=None):
        self._iternum = config['iter_num']
        self.support_image, self.spacing = load_nifty_volume_as_array(config['support_image'], return_spacing=True)
        self.support_scribble = load_nifty_volume_as_array(config['support_scribble'])
        self.transform = transform
        self.sample_list = []
        self.image_dic_list = []
        self.image_list = read_file_list(config['image_list'])

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
        spacing = np.asarray(spacing)
        sample = {'image': image.astype(np.float16), 'spacing':spacing, 'image_path':image_path}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomPositionCrop(object):
    """
    Randomly crop a image in one sample;
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, padding=True):
        self.output_size = output_size
        self.padding = padding
        self.max_outsize_each_axis = self.output_size

    def random_cor(self, shape):
        position = []
        for i in range(len(shape)):
            position.append(np.random.randint(shape[i]//2-10, shape[i]//2+10))
        return np.asarray(position)

    def __call__(self, sample):
        image,spacing= sample['image'],sample['spacing']
        # pad the sample if necessary
        if self.padding:
            image = np.pad(image, [(self.max_outsize_each_axis[0]//2, self.max_outsize_each_axis[0]//2), (self.max_outsize_each_axis[1]//2,
                self.max_outsize_each_axis[1]//2), (self.max_outsize_each_axis[2]//2, self.max_outsize_each_axis[2]//2)], mode='constant', constant_values=0)
        if image.shape[0] <= self.max_outsize_each_axis[0] or image.shape[1] <= self.max_outsize_each_axis[1] or image.shape[2] <= \
                self.max_outsize_each_axis[2]:
            pw = max((self.max_outsize_each_axis[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.max_outsize_each_axis[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.max_outsize_each_axis[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        shape = image.shape
        background_chosen = True
        while background_chosen:
            random_cor = self.random_cor(shape)
            if image[random_cor[0] , random_cor[1] ,random_cor[2]] >= 0.001:
                background_chosen = False
        sample['random_position']=random_cor*np.asarray(spacing)
        image_patch = image[random_cor[0]-self.output_size[0]//2:random_cor[0] + self.output_size[0]//2,
                            random_cor[1]-self.output_size[1]//2:random_cor[1] + self.output_size[1]//2,
                            random_cor[2]-self.output_size[2]//2:random_cor[2] + self.output_size[2]//2]
        sample['random_crop_image']=image_patch
        return sample

class Relative_distance(object):
    def __init__(self,network, out_mode='fc_position', config=None, judge_noise=False, noise_wanted_ls=[],distance_mode='linear', distance_ratio=0):
        self.network = network
        self.patch_size = config['patch_size']
        self.distance_ratio = np.array(distance_ratio)[np.newaxis, :]
        self.step = config['step']
        self.class_wanted = config['class_wanted']
        self.support_image = pad(load_volume_as_array(config['support_image']), pad_size=self.patch_size)
        self.support_mask = pad(load_volume_as_array(config['support_scribble']), pad_size=self.patch_size)
        self.distance_mode = distance_mode
        self.out_mode = out_mode
        self.judge_noise = judge_noise
        self.noise_wanted_ls = noise_wanted_ls
        self.support_position = {}
        self.support_center_feature = {}
        self.support_center_feature_1 = {}

    def cal_support_position(self):
        for class_cur in range(len(self.class_wanted)):
            ognb = self.class_wanted[class_cur]
            support_mask = np.array(np.where(self.support_mask==ognb)).transpose()
            for iii in range(0, support_mask.shape[0]-1, 32):
                key = '{0:}_{1:}'.format(ognb, iii)
                '''crop several support patch'''
                support_batch = []
                cur_support_cor = support_mask[iii:min(iii+32, support_mask.shape[0]):self.step[class_cur]] # 4,3
                for i in range(cur_support_cor.shape[0]):
                    support_cor = cur_support_cor[i]
                    support_batch.append(self.support_image[support_cor[0] - self.patch_size[0] // 2:support_cor[0] + self.patch_size[0] // 2,
                                support_cor[1] - self.patch_size[1] // 2:support_cor[1] + self.patch_size[1] // 2,
                                support_cor[2] - self.patch_size[2] // 2:support_cor[2] + self.patch_size[2] // 2][np.newaxis])
                support_batch = np.asarray(support_batch) # 32*1*d*w*h
                self.add_support_position(support_batch, key) # k, 3

    def add_support_position(self, support_patch, key):
        '''
        support_patch: [b*1*d*w*h] 
        '''
        self.support_patch = support_patch
        self.support_all = self.network(torch.from_numpy(support_patch).float().half(), self.judge_noise)
        self.support_position[key] = self.support_all[self.out_mode].cpu().numpy() # 
        if self.judge_noise:
            self.support_center_feature[key] = {}
            for nkey in self.support_all.keys():
                if nkey in self.noise_wanted_ls:
                    self.support_center_feature[key][nkey] = self.support_all[nkey].cpu().numpy()/(norm(self.support_all[nkey].cpu().numpy(), axis=1, keepdims=True)+0.0001) #[n,c]

    def cal_query_position(self, query_batch, mean=False):
        '''
        query_batch:[b*1*d*w*h]
        '''
        query_all = self.network(torch.from_numpy(query_batch).float().half())
        self.quer_position = query_all[self.out_mode].cpu().numpy()# [b, 3]
        if mean:
            self.quer_position = np.mean(self.quer_position, axis=0, keepdims=True) # [1, 3]
        return

    def cal_RD(self, key=None):
        if self.distance_mode=='linear':
            relative_position = self.distance_ratio*(self.support_position[key]-self.quer_position)
        elif self.distance_mode=='tanh':
            relative_position = self.distance_ratio*np.tanh(self.support_position[key]-self.quer_position)
        else:
            raise ValueError('Please select a correct distance mode!!！')
        return relative_position
    
    def cal_noise(self, query_batch, key):
        '''
        query_batch: 32*1*d*w*h
        '''
        query_all = self.network(torch.from_numpy(query_batch).float().half(), self.judge_noise)
        query_center_feature = {}
        for nkey in query_all.keys():
            if nkey in self.noise_wanted_ls:
                query_center_feature[nkey] = query_all[nkey].cpu().numpy()/(norm(query_all[nkey].cpu().numpy(), axis=1, keepdims=True)+0.00001) #[n,c]
        noise_dic = {}
        for i in range(query_center_feature[list(query_center_feature.keys())[0]].shape[0]): #
            for nkey in query_center_feature.keys():
                noise = dot(query_center_feature[nkey][i], self.support_center_feature[key][nkey][i])
                if i ==0:
                    noise_dic[nkey]=[]
                noise_dic[nkey].append(noise)
        return noise_dic


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
    random.seed(random_seed)  
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def pad(img, pad_size):
    img = np.pad(img, [(pad_size[0] // 2, pad_size[0] // 2), (pad_size[1] // 2, pad_size[1] // 2),
                (pad_size[2] // 2, pad_size[2] // 2)], mode='constant', constant_values=0)
    return img


def process_bar(num, total):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('*'*ratenum,' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()