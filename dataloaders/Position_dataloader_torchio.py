#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from numpy.lib.function_base import append
import torch
import numpy as np
from torch.utils.data import Dataset
import time
import random
import itertools
from torch.utils.data.sampler import Sampler
from data_process.data_process_func import *
import torchio as tio
from multiprocessing import Pool, cpu_count

def create_sample(image_path, out_size):
    image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
    image = image.astype(np.float16)
    spacing = np.array(spacing).astype(np.float32)
    spacing = torch.tensor(spacing[:,np.newaxis,np.newaxis,np.newaxis])
    if out_size:
        if image.shape[0] <= out_size[0] or image.shape[1] <= out_size[1] or image.shape[2] <= \
                out_size[2]:
            pw = max((out_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((out_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((out_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        image = np.pad(image, [(out_size[0]//2, out_size[0]//2), (out_size[1]//2, 
                out_size[1]//2), (out_size[2]//2, out_size[2]//2)], mode='constant', constant_values=0)
    shape = list(image.shape)
    cor_x,cor_z,cor_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
    cor_z,cor_x,cor_y = cor_z.astype(np.float16),cor_x.astype(np.float16),cor_y.astype(np.float16)
    cor = np.concatenate((cor_z[np.newaxis], cor_x[np.newaxis], cor_y[np.newaxis]), axis=0)
    sample = {'image': image[np.newaxis], 'coordinate':cor, 'spacing':spacing,'image_path':image_path}
    return sample
        

class PositionDataloader(Dataset):
    """ Dataset position """
    def __init__(self, config=None, split='train', num=None, transform=None, 
                random_sample=True, load_aug=False, load_memory=True, out_size=None):
        self._data_root = config['data_root']
        self._image_filename = config['image_name']
        self._iternum = config['iter_num']
        self.out_size = out_size
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.image_dic = {}
        image_task_dic = {}
        self.iternum = 0
        self.load_aug = load_aug
        self.load_memory = load_memory
        self.random_sample = random_sample
        
        self.image_name_list = os.listdir(os.path.join(self._data_root,split))
        if load_memory:
            #p = Pool(2)
            p = Pool(cpu_count())
            for i in range(len(self.image_name_list)):
                image_name = self.image_name_list[i]
                if self._image_filename is None:
                    image_path = os.path.join(self._data_root, self.split, image_name)
                else:
                    image_path = os.path.join(self._data_root, self.split, image_name, self._image_filename)
                image_task_dic[image_name]= p.apply_async(create_sample, args=(image_path, out_size, ))
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
            if self._image_filename is None:
                image_path = os.path.join(self._data_root, self.split, image_name)
            else:
                image_path = os.path.join(self._data_root, self.split, image_name, self._image_filename)
            sample = create_sample(image_path, self.out_size)
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
        self.output_size = output_size
        self.foregroung_only = foreground_only
        self.fluct_range=fluct_range
        self.small_move = small_move

    def random_position(self, shape, initial_position=[0,0,0], small_move=False):
        position = []
        for i in range(len(shape)):
            if small_move:
                position.append(random.randint(max(0, initial_position[i]-self.fluct_range[i]), min(shape[i]-self.output_size[i], initial_position[i]+self.fluct_range[i])))
            else:
                position.append(random.randint(0, shape[i] - self.output_size[i]))
        return np.asarray(position)

    def __call__(self, sample):
        image,cor= sample['image'],sample['coordinate']
        nsample ={}
        nsample['spacing']=sample['spacing']
        background_chosen = True
        shape_n = image.shape[1::]
        while background_chosen:
            random_pos0 = self.random_position(shape_n)
            if image[0, random_pos0[0] + self.output_size[0]//2, 
                    random_pos0[1] + self.output_size[1]//2, random_pos0[2] + self.output_size[2]//2]!=0:
                background_chosen = False
        nsample['random_crop_image_0']=image[:, random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]
        nsample['random_position_0'] = cor[:,random_pos0[0],random_pos0[1],random_pos0[2]] # 3
        nsample['random_fullsize_position_0'] = cor[:, random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]
        background_chosen = True
        while background_chosen:
            random_pos1= self.random_position(shape_n, nsample['random_position_0'], self.small_move)
            if image[0, random_pos1[0] + self.output_size[0] // 2, random_pos1[1] + self.output_size[1] // 2,
                     random_pos1[2] + self.output_size[2] // 2] != 0:
                background_chosen = False
        nsample['random_crop_image_1'] = image[:, random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                     random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                     random_pos1[2]:random_pos1[2] + self.output_size[2]]
        nsample['random_position_1'] = cor[:, random_pos1[0],random_pos1[1],random_pos1[2]]
        nsample['random_fullsize_position_1'] = cor[:, random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                     random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                     random_pos1[2]:random_pos1[2] + self.output_size[2]]
        for key in ['random_crop_image_0', 'random_position_0', 'random_fullsize_position_0', 
                    'random_crop_image_1', 'random_position_1', 'random_fullsize_position_1']:
                nsample[key]=torch.from_numpy(nsample[key].astype(np.float32))
        return nsample

# class RandomDoubleCrop(object):
#     """
#     Randomly crop several images in one sample;
#     distance is a vector(could be positive or pasitive), representing the vector
#     from image1 to image2.
#     Args:
#     output_size (int): Desired output size
#     """

#     def __init__(self, output_size, foreground_only=True, small_move=False, fluct_range=[0,0,0]):
#         self.output_size = output_size
#         self.foregroung_only = foreground_only
#         self.fluct_range=fluct_range
#         self.small_move = small_move

#     def random_position(self, shape, initial_position=[0,0,0], fluct_range=[5, 10,10], small_move=False):
#         position = []
#         for i in range(len(shape)):
#             if small_move:
#                 position.append(np.random.randint(max(0, initial_position[i]-fluct_range[i]),
#                                                   min(shape[i] - self.output_size[i], initial_position[i]+fluct_range[i])))
#             else:
#                 position.append(
#                     np.random.randint(0, shape[i] - self.output_size[i]))
#         return np.asarray(position)

#     def __call__(self, sample):
#         image,cor= sample['image'],sample['coordinate']
#         nsample ={}
#         nsample['spacing']=sample['spacing']
#         background_chosen = True
#         shape_n = image.shape[1::]
#         while background_chosen:
#             random_pos0 = self.random_position(shape_n)
#             if image[0, random_pos0[0] + self.output_size[0]//2, 
#                     random_pos0[1] + self.output_size[1]//2, random_pos0[2] + self.output_size[2]//2]!=0:
#                 background_chosen = False
#         nsample['random_crop_image_0']=image[:, random_pos0[0]:random_pos0[0] + self.output_size[0],
#                                                     random_pos0[1]:random_pos0[1] + self.output_size[1],
#                                                     random_pos0[2]:random_pos0[2] + self.output_size[2]].astype(np.float32)
#         nsample['random_position_0'] = cor[:,random_pos0[0],random_pos0[1],random_pos0[2]].astype(np.float32) # 3
#         nsample['random_fullsize_position_0'] = cor[:, random_pos0[0]:random_pos0[0] + self.output_size[0],
#                                                     random_pos0[1]:random_pos0[1] + self.output_size[1],
#                                                     random_pos0[2]:random_pos0[2] + self.output_size[2]].astype(np.float32)
#         background_chosen = True
#         while background_chosen:
#             random_pos1= self.random_position(shape_n,nsample['random_position_0'],
#                         fluct_range=self.fluct_range, small_move=self.small_move)
#             if image[0, random_pos1[0] + self.output_size[0] // 2, random_pos1[1] + self.output_size[1] // 2,
#                      random_pos1[2] + self.output_size[2] // 2] != 0:
#                 background_chosen = False
#         nsample['random_crop_image_1'] = image[:, random_pos1[0]:random_pos1[0] + self.output_size[0],
#                                                      random_pos1[1]:random_pos1[1] + self.output_size[1],
#                                                      random_pos1[2]:random_pos1[2] + self.output_size[2]].astype(np.float32)
#         nsample['random_position_1'] = cor[:, random_pos1[0],random_pos1[1],random_pos1[2]].astype(np.float32)
#         nsample['random_fullsize_position_1'] = cor[:, random_pos1[0]:random_pos1[0] + self.output_size[0],
#                                                      random_pos1[1]:random_pos1[1] + self.output_size[1],
#                                                      random_pos1[2]:random_pos1[2] + self.output_size[2]].astype(np.float32)
#         return nsample

class RandomDoubleNoise(object):
    def __init__(self, mean=0, std=0.1,include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], prob=0):
        self.prob = prob
        self.add_noise_0 = tio.RandomNoise(mean=mean, std=std, include=include_0)
        self.add_noise_1 = tio.RandomNoise(mean=mean, std=std, include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample= self.add_noise_0(sample)
        if torch.rand(1)<self.prob:
            sample= self.add_noise_1(sample)
        return sample

class RandomDoubleFlip(object):
    def __init__(self, include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], prob=0):
        self.flip_probability = prob
        self.include0 = include_0
        self.include1 = include_1
    def __call__(self, sample):
        axes = np.random.randint(0, 2)
        flip = tio.RandomFlip(axes=axes, flip_probability=self.flip_probability, include = self.include0)
        axes = np.random.randint(0, 2)
        flip = tio.RandomFlip(axes=axes, flip_probability=self.flip_probability, include = self.include1)
        sample= flip(sample)
        return sample

class RandomDoubleSpike(object):
    def __init__(self, num_spikes=3, intensity=1.2,include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], prob=0):
        self.prob = prob
        self.add_spike_0 = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity,include=include_0)
        self.add_spike_1 = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity,include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_spike_0(sample)
        if torch.rand(1)<self.prob:
            sample=self.add_spike_1(sample)
        return sample

class RandomDoubleGhosting(object):
    def __init__(self, intensity=0.8,include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], prob=0):
        self.prob = prob
        self.add_ghost_0 = tio.RandomGhosting(intensity=intensity, include=include_0)
        self.add_ghost_1 = tio.RandomGhosting(intensity=intensity, include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_ghost_0(sample)
        if torch.rand(1)<self.prob:
            sample=self.add_ghost_1(sample)
        return sample

class RandomDoubleElasticDeformation(object):
    def __init__(self, num_control_points=[5,10,10], max_displacement=[7,7,7],
                include_0=['random_crop_image_0','random_fullsize_position_0'], 
                include_1=['random_crop_image_1','random_fullsize_position_1'], prob=0):
        self.prob = prob
        self.add_elas_0 = tio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement = max_displacement,
            include=include_0)
        self.add_elas_1 = tio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement = max_displacement,
            include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_elas_0(sample)
        if torch.rand(1)<self.prob:
            sample=self.add_elas_1(sample)
        return sample

class RandomDoubleAffine(object):
    def __init__(self, scales=[0.2,0.2,0.2], degrees=[10,10,10],
                include_0=['random_crop_image_0','random_fullsize_position_0'], 
                include_1=['random_crop_image_1','random_fullsize_position_1'], prob=0):
        self.prob = prob
        self.add_elas_0 = tio.RandomAffine(
            scales=scales,
            degrees=degrees,
            include=include_0)
        self.add_elas_1 = tio.RandomAffine(
            scales=scales,
            degrees=degrees,
            include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_elas_0(sample)
        if torch.rand(1)<self.prob:
            sample=self.add_elas_1(sample)
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
        # key_float_list = ['rela_distance','rela_fullsize_distance','rela_fullsize_aug_distance',
        #         'rela_poi','random_position','random_fullsize_position','random_fullsize_aug_position','consis_fullsize_distance']
        # for key in sample.keys():
        #     if 'random_crop_image' in key or 'random_crop_aug_image' in key:
        #         image = sample[key]
        #         image= image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        #         sample[key] = torch.from_numpy(image)
        #     else:
        #         for key_float in  key_float_list:
        #             if key_float in key:
        #                 sample[key]=torch.from_numpy(sample[key]).float()
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
