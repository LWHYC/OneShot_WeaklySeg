#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from data_process.data_process_func import *
from multiprocessing import Pool, cpu_count

def create_sample(image_path, out_size):
    image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
    image = image.astype(np.float16)
    spacing = np.array(spacing).astype(np.float32)
    spacing = spacing[:,np.newaxis,np.newaxis,np.newaxis]
    if out_size:
        if image.shape[0] <= out_size[0] or image.shape[1] <= out_size[1] or image.shape[2] <= \
                out_size[2]:
            pw = max((out_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((out_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((out_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
    shape = list(image.shape)
    sample = {'image': torch.from_numpy(image[np.newaxis]), 'spacing':torch.from_numpy(spacing),'image_path':image_path}
    return sample
        

class PositionDataloader(Dataset):
    """ Dataset position """
    def __init__(self, config=None, image_name_list='train', num=None, transform=None, 
                random_sample=True, load_aug=False, load_memory=True, out_size=None):
        self._iternum = config['iter_num']
        self.out_size = out_size
        self.transform = transform
        self.sample_list = []
        self.image_dic = {}
        image_task_dic = {}
        self.iternum = 0
        self.load_aug = load_aug
        self.load_memory = load_memory
        self.random_sample = random_sample
        self.image_name_list = read_file_list(image_name_list)

        if load_memory:
            #p = Pool(2)
            p = Pool(cpu_count())
            for image_name in self.image_name_list:
                image_task_dic[image_name]= p.apply_async(create_sample, args=(image_name, out_size, ))
            p.close()
            p.join()
            for image_name in image_task_dic.keys():
                self.image_dic[image_name]=image_task_dic[image_name].get()
        
        if num is not None:
            self.image_name_list = self.image_name_list[:num]
        print("total {} samples".format(len(self.image_name_list)))


    def __len__(self):
        if self.random_sample:
            return self._iternum
        else:
            return len(self.image_name_list)

    def __getitem__(self, idx):
        if self.load_memory:
            sample = self.image_dic[random.sample(self.image_name_list, 1)[0]].copy()
        else:
            if self.random_sample:
                image_name = random.sample(self.image_name_list, 1)[0]
            else:
                image_name = self.image_name_list[idx]
            sample = create_sample(image_name, self.out_size)
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomDoubleCrop(object):
    """
    Randomly crop several images in one sample;
    distance is a vector(could be positive or pasitive), representing the vector
    from image1 to image2.
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, foreground_only=True, small_move=False, fluct_range=[0,0,0]):
        self.output_size = torch.tensor(output_size, dtype=torch.int16)
        self.img_pad = torch.div(self.output_size, 2, rounding_mode='trunc')
        self.foregroung_only = foreground_only
        self.fluct_range = fluct_range # distance, mm
        self.small_move = small_move
        cor_x,cor_z,cor_y = np.meshgrid(np.arange(self.output_size[1]), 
                                        np.arange(self.output_size[0]), 
                                        np.arange(self.output_size[2]))
        self.cor_grid = torch.from_numpy(np.concatenate((cor_z[np.newaxis], \
                                                        cor_x[np.newaxis], \
                                                        cor_y[np.newaxis]), axis=0))

    def random_position(self, shape, initial_position=[0,0,0], spacing=[1,1,1], small_move=False):
        position = []
        for i in range(len(shape)):
            if small_move:
                position.append(random.randint(max(0, initial_position[i]-np.int(self.fluct_range[i]/spacing[i])), \
                                               min(shape[i]-1, initial_position[i]+np.int(self.fluct_range[i]/spacing[i]))))
            else:
                position.append(random.randint(0, shape[i]-1))
        return torch.tensor(position)

    def __call__(self, sample):
        image= sample['image']
        nsample ={}
        nsample['image_path']=sample['image_path']
        nsample['spacing']=sample['spacing']
        spacing = sample['spacing'].squeeze().numpy()
        
        background_chosen = True
        shape_n = torch.tensor(image.shape[1::])
        choose_num = 0
        while background_chosen:
            choose_num +=1
            random_pos0 = self.random_position(shape_n)
            if image[0, random_pos0[0], random_pos0[1], random_pos0[2]]>0.0001:
                background_chosen = False
        pad_size_ = [torch.maximum(-random_pos0+torch.div(self.output_size, 2, rounding_mode='trunc'), torch.tensor(0)).to(torch.int16), 
                     torch.maximum(random_pos0+torch.div(self.output_size, 2, rounding_mode='trunc')-shape_n, torch.tensor(0)).to(torch.int16)]
        pad_size = []
        for i in range(3):
            pad_size.append(pad_size_[0][2-i])
            pad_size.append(pad_size_[1][2-i])
        min_random_pos0 = torch.maximum(random_pos0-torch.div(self.output_size, 2, rounding_mode='trunc'), torch.tensor(0)).to(torch.int16)
        max_random_pos0 = torch.minimum(random_pos0+torch.div(self.output_size, 2, rounding_mode='trunc'), shape_n).to(torch.int16)
        nsample['random_crop_image_0']=F.pad(image[:, min_random_pos0[0]:max_random_pos0[0],
                                                    min_random_pos0[1]:max_random_pos0[1],
                                                    min_random_pos0[2]:max_random_pos0[2]], pad = pad_size)
        
        nsample['random_position_0'] = random_pos0
        nsample['random_fullsize_position_0'] = self.cor_grid + random_pos0.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        background_chosen = True
        choose_num = 0
        while background_chosen:
            choose_num +=1
            random_pos1= self.random_position(shape_n, nsample['random_position_0'], spacing, self.small_move)
            if image[0, random_pos1[0], random_pos1[1], random_pos1[2]]>0.0001:
                background_chosen = False
        pad_size_ = [torch.maximum(-random_pos1+torch.div(self.output_size, 2, rounding_mode='trunc'), torch.tensor(0)).to(torch.int16), 
                     torch.maximum(random_pos1+torch.div(self.output_size, 2, rounding_mode='trunc')-shape_n, torch.tensor(0)).to(torch.int16)]
        pad_size = []
        for i in range(3):
            pad_size.append(pad_size_[0][2-i])
            pad_size.append(pad_size_[1][2-i])
        min_random_pos1 = torch.maximum(random_pos1-torch.div(self.output_size, 2, rounding_mode='trunc'), torch.tensor(0)).to(torch.int16)
        max_random_pos1 = torch.minimum(random_pos1+torch.div(self.output_size, 2, rounding_mode='trunc'), shape_n).to(torch.int16)
        nsample['random_crop_image_1']=F.pad(image[:, min_random_pos1[0]:max_random_pos1[0],
                                                    min_random_pos1[1]:max_random_pos1[1],
                                                    min_random_pos1[2]:max_random_pos1[2]], pad=pad_size)
        nsample['random_position_1'] = random_pos1
        nsample['random_fullsize_position_1'] = self.cor_grid + random_pos1.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        key_ls = list(nsample.keys())
        for key in key_ls:
            if torch.is_tensor(nsample[key]):
                nsample[key]=nsample[key].to(torch.float32)
        return nsample


class RandomDoubleMask(object):
    def __init__(self, max_round=1, include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], mask_size=[0,0,0]):
        self.include_0 = include_0
        self.include_1 = include_1
        self.max_round = max_round
        self.mask_size = mask_size
    def __call__(self, sample):
        for include in self.include_0[0], self.include_1[0]:
            max_round = np.random.randint(1, self.max_round)
            sample[include.replace('random','random_mask')] = torch.clone(sample[include])
            sample[include.replace('random_crop_image','random_mask')] = torch.zeros_like(sample[include.replace('random','random_mask')])
            mask_image = sample[include].clone()
            shape = mask_image.shape[1::]
            for _ in range(max_round):
                min_cor = []
                for i in range(3):  
                    min_cor.append(np.random.randint(0, shape[i]-self.mask_size[i]))
                sample[include.replace('random','random_mask')][min_cor[0]:min_cor[0]+self.mask_size[0],min_cor[1]:min_cor[1]+self.mask_size[1],min_cor[2]:min_cor[2]+self.mask_size[2]]=0
                sample[include.replace('random_crop_image','random_mask')][min_cor[0]:min_cor[0]+self.mask_size[0],min_cor[1]:min_cor[1]+self.mask_size[1],min_cor[2]:min_cor[2]+self.mask_size[2]]=0
        return sample

class ToPositionTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        shape = sample['random_fullsize_position_0'].shape
        sample['random_position_0']=sample['random_fullsize_position_0'][:,shape[0]//2, shape[1]//2, shape[2]//2]
        sample['random_position_1']=sample['random_fullsize_position_1'][:,shape[0]//2, shape[1]//2, shape[2]//2]
        spacing = sample['spacing']
        sample['rela_distance']=(sample['random_position_0']-sample['random_position_1'])*spacing.squeeze()
        sample['rela_fullsize_distance'] = (sample['random_fullsize_position_0']-sample['random_fullsize_position_1'])*spacing
        return sample

