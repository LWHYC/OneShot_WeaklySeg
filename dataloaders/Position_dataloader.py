#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import numbers
from scipy import ndimage
from glob import glob
from torch.utils.data import Dataset
import tqdm
from tqdm import trange
import random
import h5py
import itertools
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data.sampler import Sampler
from data_process.data_process_func import *

class PositionDataloader(Dataset):
    """ structseg Dataset position """
    def __init__(self, config=None, split='train', num=None, transform=None, 
                random_sample=True, load_aug=False, load_memory=True):
        self._data_root = config['data_root']
        self._image_filename = config['image_name']
        self._iternum = config['iter_num']
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.image_dic = {}
        self.iternum = 0
        self.load_aug = load_aug
        self.load_memory = load_memory
        self.random_sample = random_sample
        if load_aug:
            self._aug_image_filename = config['aug_image_name']
            self._aug_cor_x_filename = config['aug_cor_x_name']
            self._aug_cor_y_filename = config['aug_cor_y_name']
            self._aug_cor_z_filename = config['aug_cor_z_name']
        self.image_name_list = os.listdir(os.path.join(self._data_root,split))
        if load_memory:
            for i in trange(len(self.image_name_list)):
                image_name = self.image_name_list[i]
                if self._image_filename is None:
                    image_path = os.path.join(self._data_root, self.split, image_name)
                else:
                    image_path = os.path.join(self._data_root, self.split, image_name, self._image_filename)
                image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
                spacing = np.asarray(spacing)
                shape = list(image.shape)
                cor_x,cor_z,cor_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
                cor_z,cor_x,cor_y = cor_z.astype(np.float32),cor_x.astype(np.float32),cor_y.astype(np.float32)
                cor = np.concatenate((cor_z[np.newaxis,:], cor_x[np.newaxis,:], cor_y[np.newaxis,:]), axis=0)
                sample = {'image': image.astype(np.float16), 'coordinate':cor, 'spacing':spacing,'image_path':image_path}
                if self.load_aug:
                    aug_image_path = os.path.join(self._data_root, self.split, image_name, self._aug_image_filename)
                    aug_cor_x_path = os.path.join(self._data_root, self.split, image_name, self._aug_cor_x_filename)
                    aug_cor_y_path = os.path.join(self._data_root, self.split, image_name, self._aug_cor_y_filename)
                    aug_cor_z_path = os.path.join(self._data_root, self.split, image_name, self._aug_cor_z_filename)
                    aug_image = load_nifty_volume_as_array(aug_image_path, return_spacing=False)
                    aug_cor_x = load_nifty_volume_as_array(aug_cor_x_path, return_spacing=False)
                    aug_cor_y = load_nifty_volume_as_array(aug_cor_y_path, return_spacing=False)
                    aug_cor_z = load_nifty_volume_as_array(aug_cor_z_path, return_spacing=False)
                    sample['aug_image'] = aug_image
                    sample['aug_cor_x'] = aug_cor_x
                    sample['aug_cor_y'] = aug_cor_y
                    sample['aug_cor_z'] = aug_cor_z
                self.image_dic[image_name]=sample
        
        if num is not None:
            self.image_name_list = self.image_name_list[:num]
        print("total {} samples".format(len(self.image_name_list)))

    # def get_image_dic(self):
    #     return self.image_dic_list

    def __len__(self):
        if self.random_sample:
            return self._iternum
        else:
            return len(self.image_name_list)

    def __getitem__(self, idx):
        if self.load_memory:
            sample = self.image_dic[random.sample(self.image_name_list, 1)[0]]
        else:
            if self.random_sample:
                image_name = random.sample(self.image_name_list, 1)[0]
            else:
                image_name = self.image_name_list[idx]
            if self._image_filename is None:
                image_path = os.path.join(self._data_root, self.split, image_name)
            else:
                image_path = os.path.join(self._data_root, self.split, image_name, self._image_filename)
            image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
            print(image.shape)
            spacing = np.asarray(spacing)
            shape = list(image.shape)
            cor_x,cor_z,cor_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
            cor_z,cor_x,cor_y = cor_z.astype(np.float32),cor_x.astype(np.float32),cor_y.astype(np.float32)
            cor = np.concatenate((cor_z[np.newaxis,:], cor_x[np.newaxis,:], cor_y[np.newaxis,:]), axis=0)
            sample = {'image': image.astype(np.float16)} #, 'coordinate':cor, 'spacing':spacing,'image_path':image_path}
            if self.load_aug:
                aug_image_path = os.path.join(self._data_root, self.split, image_name, self._aug_image_filename)
                aug_cor_x_path = os.path.join(self._data_root, self.split, image_name, self._aug_cor_x_filename)
                aug_cor_y_path = os.path.join(self._data_root, self.split, image_name, self._aug_cor_y_filename)
                aug_cor_z_path = os.path.join(self._data_root, self.split, image_name, self._aug_cor_z_filename)
                aug_image = load_nifty_volume_as_array(aug_image_path, return_spacing=False)
                aug_cor_x = load_nifty_volume_as_array(aug_cor_x_path, return_spacing=False)
                aug_cor_y = load_nifty_volume_as_array(aug_cor_y_path, return_spacing=False)
                aug_cor_z = load_nifty_volume_as_array(aug_cor_z_path, return_spacing=False)
                sample['aug_image'] = aug_image
                sample['aug_cor_x'] = aug_cor_x
                sample['aug_cor_y'] = aug_cor_y
                sample['aug_cor_z'] = aug_cor_z

        if self.transform:
            sample = self.transform(sample)

        return sample

class PositionDoublePatientDataloader(Dataset):
    """ structseg Dataset position """
    def __init__(self, config=None, split='train', num=None, transform=None):
        self._data_root = config['data_root']
        self._image_filename = config['image_name']
        self._iternum = config['iter_num']
        self.split = split
        self.transform = transform
        self.sample_list = []
        if split=='train':
            self.image_name_list = os.listdir(self._data_root+'/'+'train')
        elif split == 'valid':
            self.image_name_list = os.listdir(self._data_root + '/' + 'valid')
        elif split == 'test':
            self.image_name_list = os.listdir(self._data_root + '/' + 'test')
        else:
            ValueError('please input choose correct mode! i.e."train" "valid" "test"')
        if num is not None:
            self.image_name_list = self.image_name_list[:num]
        print("total {} samples".format(len(self.image_name_list)))

    def __len__(self):
        if self.split == 'train':
            return self._iternum
        else:
            return len(self.image_name_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            double_image_fold = random.sample(self.image_name_list, 2)
        else:
            # double_image_fold = [self.image_name_list[idx], random.sample(self.image_name_list, 1)[0]]
            double_image_fold = [ self.image_name_list[idx], '/home/disk/LWH/Data/Pancreas/test/Pancreas071']
            #double_image_fold = [self.image_name_list[idx], self.image_name_list[idx-1]]
        image_path_0 = os.path.join(self._data_root, self.split, double_image_fold[0], self._image_filename)
        image_path_1 = os.path.join(self._data_root, self.split, double_image_fold[1], self._image_filename)
        image_0, spacing_0 = load_nifty_volume_as_array(image_path_0, return_spacing=True)
        image_1, spacing_1 = load_nifty_volume_as_array(image_path_1, return_spacing=True)
        spacing_0 = np.asarray(spacing_0)
        spacing_1 = np.asarray(spacing_1)
        sample = {'image_0': image_0, 'spacing_0':spacing_0, 'spacing_1':spacing_1, 'image_1':image_1,
                  'image_path_0':image_path_0,'image_path_1':image_path_1}
        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomPositionSeveralCrop(object):
    """
    Randomly Crop several  the image from one sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, crop_num):
        self.output_size = output_size
        self.crop_num = crop_num
    def __call__(self, sample):
        image= sample['image']
        n_sample = {}
        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        for i in range(self.crop_num):
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

            label = w1+self.output_size//2
            n_image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            n_sample['image{}'.format(i)]=n_image
            n_sample['label{}'.format(i)]=label
        return n_sample

class RandomPositionDoubleCrop(object):
    """
    Randomly crop several images in one sample;
    distance is a vector(could be positive or pasitive), representing the vector
    from image1 to image2.
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, padding=True, foreground_only=True, small_move=False, fluct_range=[0,0,0]
                                ,elastic_prob=0,scale_prob=0, scale_ratio=0.2, rotate_prob=0, rotate_ratio=30, axes=[0,1,2]):
        self.output_size = output_size
        self.padding = padding
        self.foregroung_only = foreground_only
        self.fluct_range=fluct_range
        self.elastic_prob = elastic_prob
        self.scale_prob = scale_prob
        self.scale_ratio = scale_ratio
        self.rotate_prob = rotate_prob
        self.rotate_ratio = rotate_ratio
        self.axes = axes
        self.small_move = small_move
    def random_position(self, shape, initial_position=[0,0,0], fluct_range=[5, 10,10], small_move=False):
        position = []
        for i in range(len(shape)):
            if small_move:
                position.append(np.random.randint(max(0, initial_position[i]-fluct_range[i]),
                                                  min(shape[i] - self.output_size[i], initial_position[i]+fluct_range[i])))
            else:
                position.append(
                    np.random.randint(0, shape[i] - self.output_size[i]))
        return np.asarray(position)

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state=None, cor=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         alpha 控制高斯模糊的幅度, 越大，总体
         sigma控制高斯模糊的方差，越小，坐标模糊时局部弹性变化越剧烈
         alpha_affine控制仿射变换的幅度，越大，放射变换幅度可能越大
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
        if cor == None:
            x, z, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
            shape_size = shape[:2]
        else:
            x,z,y = cor[1],cor[0],cor[2]
        shape = image.shape
        x, z, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
        shape_size = shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])  # 选取三个点
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(
            np.float32)  # 输出三个点仿射变换后的坐标
        M = cv2.getAffineTransform(pts1, pts2)  # 得到仿射变换矩阵
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)  # 对原图进行仿射变化
        wx,wz,wy = cv2.warpAffine(x.astype(np.float32), M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101),\
                   cv2.warpAffine(z.astype(np.float32), M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101),\
                   cv2.warpAffine(y.astype(np.float32), M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        # dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
        #                      sigma) * alpha  # alpha 控制高斯模糊的幅度,sigma控制高斯模糊的方差，越小局部变化越剧烈
        # dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        # # dz = np.zeros_like(dx)

        
        # indices = np.reshape(z+dz, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))  # 弹性变换后的坐标
        # image = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
        # wx = map_coordinates(wx, indices, order=1, mode='constant').reshape(shape)
        # wy = map_coordinates(wy, indices, order=1, mode='constant').reshape(shape)
        # wz = map_coordinates(wz, indices, order=1, mode='constant').reshape(shape)
        return image, M, [wz,wx,wy]

    def __call__(self, sample):
        image,spacing= sample['image'],sample['spacing']
        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        rela_position = np.zeros(3)
        if self.padding:
            image = np.pad(image, [(self.output_size[0]//2, self.output_size[0]//2), (self.output_size[1]//2, 
                self.output_size[1]//2), (self.output_size[2]//2, self.output_size[2]//2)], mode='constant', constant_values=0)

        shape = list(image.shape)
        
        cor_x,cor_z,cor_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
        cor_z,cor_x,cor_y = cor_z.astype(np.float32),cor_x.astype(np.float32),cor_y.astype(np.float32)
        cor_z_n,cor_x_n,cor_y_n = cor_z.copy(),cor_x.copy(),cor_y.copy()
        image_n = image.copy().astype(np.float32)
        #print(cor_z.shape, np.max(cor_z), np.mean(cor_z), cor_x.shape, np.max(cor_x),np.mean(cor_x), cor_y.shape, np.max(cor_y),np.mean(cor_y))
        '''
        如随机数超过elastic_rate，进行弹性变化
        如进行弹性变换，将第一步裁减图片的坐标投影至弹性变换后的坐标系
        '''
        if np.random.uniform(0, 1) < self.elastic_prob:
            image_n, M, [cor_z_n,cor_x_n,cor_y_n] = self.elastic_transform(image_n, alpha=shape[0], sigma=shape[0] * 0.1,
                                                        alpha_affine=shape[0] * 0.01)
            cor_z_n,cor_x_n,cor_y_n = cor_z_n.astype(np.float32),cor_x_n.astype(np.float32),cor_y_n.astype(np.float32)

        '''
        如随机数超过scale_rate，进行缩放变化
        如进行缩放变换，将第一步裁减图片的坐标投影至弹性变换后的坐标
        '''
        if np.random.uniform(0, 1) < self.scale_prob:
            zoomfactor = np.random.uniform(1-self.scale_ratio, 1+self.scale_ratio,size=3)
            image_n = ndimage.zoom(image_n, zoom=zoomfactor, order=1)
            cor_z_n = ndimage.zoom(cor_z_n, zoom=zoomfactor, order=1)
            cor_x_n = ndimage.zoom(cor_x_n, zoom=zoomfactor, order=1)
            cor_y_n = ndimage.zoom(cor_y_n, zoom=zoomfactor, order=1)

        '''
        如随机数超过rotate_rate，进行翻转变化
        '''
        if np.random.uniform(0, 1) < self.rotate_prob:
            angle = random.uniform(-self.rotate_ratio, self.rotate_ratio)
            if len(self.axes) != 2:
                axes = random.sample(self.axes, 2)
            else:
                axes = self.axes
            image_n = ndimage.rotate(image_n, angle, axes=axes, order=1, reshape=False)
            cor_z_n = ndimage.rotate(cor_z_n, angle, axes=axes, order=1, reshape=False)
            cor_x_n = ndimage.rotate(cor_x_n, angle, axes=axes, order=1, reshape=False)
            cor_y_n = ndimage.rotate(cor_y_n, angle, axes=axes, order=1, reshape=False)

        cor_n = np.concatenate((cor_z_n[np.newaxis,:], cor_x_n[np.newaxis,:], cor_y_n[np.newaxis,:]), axis=0)
        background_chosen = True
        shape_n = image_n.shape
        while background_chosen:
            random_pos0 = self.random_position(shape_n)
            if image_n[random_pos0[0] + self.output_size[0]//2, random_pos0[1] + self.output_size[1]//2, random_pos0[2] + self.output_size[2]//2]!=0:
                background_chosen = False
        sample['random_crop_image_0']=image_n[random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]
        sample['random_position_0'] = cor_n[:,random_pos0[0],random_pos0[1],random_pos0[2]]
        sample['random_fullsize_position_0'] = cor_n[:, random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]


        cor = np.concatenate((cor_z[np.newaxis,:], cor_x[np.newaxis,:], cor_y[np.newaxis,:]), axis=0)
        background_chosen = True
        shape = image.shape
        #print(shape, shape_t)
        while background_chosen:
            random_pos1= self.random_position(shape,sample['random_position_0'],fluct_range=self.fluct_range, small_move=self.small_move)
            if image[random_pos1[0] + self.output_size[0] // 2, random_pos1[1] + self.output_size[1] // 2,
                     random_pos1[2] + self.output_size[2] // 2] != 0:
                background_chosen = False
        sample['random_crop_image_1'] = image[random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                     random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                     random_pos1[2]:random_pos1[2] + self.output_size[2]]
        sample['random_position_1'] = cor[:, random_pos1[0],random_pos1[1],random_pos1[2]]
        sample['random_fullsize_position_1'] = cor[:, random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                     random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                     random_pos1[2]:random_pos1[2] + self.output_size[2]]

        #print(random_pos0, random_pos1, (random_pos0-random_pos1)*spacing)
        for i in range(len(rela_position)):
            if sample['random_position_0'][i]<sample['random_position_1'][i]:
                rela_position[i] = 1
        spacing = np.asarray(spacing).squeeze()
        sample['rela_distance']=(sample['random_position_0']-sample['random_position_1'])*spacing
        sample['rela_fullsize_distance'] = (sample['random_fullsize_position_0']-sample['random_fullsize_position_1'])*spacing[:,np.newaxis,np.newaxis,np.newaxis]

        sample['rela_poi']=rela_position
        return sample

class RandomZoom(object):
    def __init__(self, zoom_prob=0, zoom_ratio=0.2):
        self.zoom_prob = zoom_prob
        self.zoom_ratio = zoom_ratio

    def __call__(self, sample):
        image, cor = sample['image'], sample['coordinate']
        if np.random.uniform(0, 1) < self.zoom_prob:
            zoomfactor = np.random.uniform(1-self.scale_ratio, 1+self.scale_ratio,size=3)
            image = ndimage.zoom(image, zoom=zoomfactor, order=1)
            for i in range(cor.shape[0]):
                cor[i] = ndimage.zoom(cor[i], zoom=zoomfactor, order=1)
        return sample

class RandomRotate(object):
    def __init__(self, rotate_prob=0, rotate_ratio=0.2, axes=[0,1,2]):
        self.rotate_prob = rotate_prob
        self.rotate_ratio = rotate_ratio
        self.axes = axes
    def __call__(self, sample):
        image, cor = sample['image'], sample['coordinate']
        if np.random.uniform(0, 1) < self.rotate_prob:
            angle = random.uniform(-self.rotate_ratio, self.rotate_ratio)
            if len(self.axes) != 2:
                axes = random.sample(self.axes, 2)
            else:
                axes = self.axes
            image = ndimage.rotate(image, angle, axes=axes, order=1, reshape=False)
            for i in range(image.shape[0]):
                cor[i] = ndimage.rotate(cor[i], angle, axes=axes, order=1, reshape=False)
        return sample

class RandomPositionCrop(object):
    """
    Randomly crop several images in one sample;
    distance is a vector(could be positive or pasitive), representing the vector
    from image1 to image2.
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, padding=True, foreground_only=True, small_move=False, fluct_range=[0,0,0]):
        self.output_size = output_size
        self.padding = padding
        self.foregroung_only = foreground_only
        self.fluct_range=fluct_range
        self.small_move = small_move
        self.cur_position = [0,0,0]
    def random_position(self, shape, initial_position=[0,0,0], fluct_range=[5, 10,10], small_move=False):
        position = []
        for i in range(len(shape)):
            if small_move:
                position.append(np.random.randint(max(0, initial_position[i]-fluct_range[i]),
                                                  min(shape[i] - self.output_size[i], initial_position[i]+fluct_range[i])))
            else:
                position.append(
                    np.random.randint(0, shape[i] - self.output_size[i]))
        return np.asarray(position)

    def __call__(self, sample):
        image, cor= sample['image'], sample['coordinate']
        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        if self.padding:
            image = np.pad(image, [(self.output_size[0]//2, self.output_size[0]//2), (self.output_size[1]//2, 
                self.output_size[1]//2), (self.output_size[2]//2, self.output_size[2]//2)], mode='constant', constant_values=0)

        shape = list(image.shape)
        cor_x,cor_z,cor_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
        cor_z,cor_x,cor_y = cor_z.astype(np.float32),cor_x.astype(np.float32),cor_y.astype(np.float32)
        cor = np.concatenate((cor_z[np.newaxis,:], cor_x[np.newaxis,:], cor_y[np.newaxis,:]), axis=0)
        background_chosen = True
        shape_n = image.shape

        while background_chosen:
            random_pos = self.random_position(shape_n)
            if image[random_pos[0] + self.output_size[0]//2, random_pos[1] + self.output_size[1]//2, random_pos[2] + self.output_size[2]//2]!=0:
                background_chosen = False
        sample['image']=image[random_pos[0]:random_pos[0] + self.output_size[0],
                                                    random_pos[1]:random_pos[1] + self.output_size[1],
                                                    random_pos[2]:random_pos[2] + self.output_size[2]]
        sample['center_coordinate'] = cor[:,random_pos[0],random_pos[1],random_pos[2]]
        sample['coordinate'] = cor[:, random_pos[0]:random_pos[0] + self.output_size[0],
                                                    random_pos[1]:random_pos[1] + self.output_size[1],
                                                    random_pos[2]:random_pos[2] + self.output_size[2]]

        return sample


class RandomPositionAugDoubleCrop(object):
    """
    Randomly crop several images in one sample;
    distance is a vector(could be positive or pasitive), representing the vector
    from image1 to image2.
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, padding=True, foreground_only=True, small_move=False, augment_consistency=False,fluct_range=[0,0,0]):
        self.output_size = output_size
        self.padding = padding
        self.foregroung_only = foreground_only
        self.fluct_range=fluct_range
        self.small_move = small_move
        self.augment_consistency = augment_consistency
    def random_position(self, shape, initial_position=[0,0,0], fluct_range=[5, 10,10], small_move=False):
        position = []
        for i in range(len(shape)):
            if small_move:
                position.append(np.random.randint(max(0, initial_position[i]-fluct_range[i]),
                                                  min(shape[i] - self.output_size[i], initial_position[i]+fluct_range[i])))
            else:
                position.append(
                    np.random.randint(0, shape[i] - self.output_size[i]))
        return np.asarray(position)

    def __call__(self, sample):
        aug_image,aug_cor_z,aug_cor_x,aug_cor_y, image,spacing= sample['aug_image'], sample['aug_cor_z'], sample['aug_cor_x'], \
                                                                        sample['aug_cor_y'], sample['image'],sample['spacing']
        
        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        rela_position = np.zeros(3)
        if self.padding:
            image = np.pad(image, [(self.output_size[0]//2, self.output_size[0]//2), (self.output_size[1]//2, 
                self.output_size[1]//2), (self.output_size[2]//2, self.output_size[2]//2)], mode='constant', constant_values=0)
        shape = list(image.shape)

        cor_x,cor_z,cor_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
        cor = np.concatenate((cor_z[np.newaxis,:], cor_x[np.newaxis,:], cor_y[np.newaxis,:]), axis=0)
        aug_cor = np.concatenate((aug_cor_z[np.newaxis,:], aug_cor_x[np.newaxis,:], aug_cor_y[np.newaxis,:]), axis=0)
        background_chosen = True

        while background_chosen:
            random_aug_pos0 = self.random_position(shape)
            if aug_image[random_aug_pos0[0] + self.output_size[0]//2, random_aug_pos0[1] + self.output_size[1]//2, random_aug_pos0[2] + self.output_size[2]//2]!=0:
                background_chosen = False
        sample['random_crop_aug_image_0']=aug_image[random_aug_pos0[0]:random_aug_pos0[0] + self.output_size[0],
                                                    random_aug_pos0[1]:random_aug_pos0[1] + self.output_size[1],
                                                    random_aug_pos0[2]:random_aug_pos0[2] + self.output_size[2]]
        sample['random_fullsize_aug_position_0'] = aug_cor[:, random_aug_pos0[0]:random_aug_pos0[0] + self.output_size[0],
                                                    random_aug_pos0[1]:random_aug_pos0[1] + self.output_size[1],
                                                    random_aug_pos0[2]:random_aug_pos0[2] + self.output_size[2]]
        random_pos0 = np.asarray([np.around(cor_z[random_aug_pos0[0], random_aug_pos0[1], random_aug_pos0[2]]), \
                                                np.around(cor_x[random_aug_pos0[0], random_aug_pos0[1], random_aug_pos0[2]]), \
                                                np.around(cor_y[random_aug_pos0[0], random_aug_pos0[1], random_aug_pos0[2]])])
        sample['random_position_0'] = random_pos0
        sample['random_crop_image_0'] = image[random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]
        sample['random_fullsize_position_0'] = cor[:, random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]


        background_chosen = True
        #print(shape, shape_t)
        while background_chosen:
            random_aug_pos1= self.random_position(shape,random_aug_pos0,fluct_range=self.fluct_range, small_move=self.small_move)
            if aug_image[random_aug_pos1[0] + self.output_size[0] // 2, random_aug_pos1[1] + self.output_size[1] // 2,
                     random_aug_pos1[2] + self.output_size[2] // 2] != 0:
                background_chosen = False
        sample['random_crop_aug_image_1']=aug_image[random_aug_pos1[0]:random_aug_pos1[0] + self.output_size[0],
                                                    random_aug_pos1[1]:random_aug_pos1[1] + self.output_size[1],
                                                    random_aug_pos1[2]:random_aug_pos1[2] + self.output_size[2]]
        sample['random_fullsize_aug_position_1'] = aug_cor[:, random_aug_pos1[0]:random_aug_pos1[0] + self.output_size[0],
                                                    random_aug_pos1[1]:random_aug_pos1[1] + self.output_size[1],
                                                    random_aug_pos1[2]:random_aug_pos1[2] + self.output_size[2]]
        random_pos1 = np.asarray([np.around(cor_z[random_aug_pos1[0], random_aug_pos1[1], random_aug_pos1[2]]), \
                                                np.around(cor_x[random_aug_pos1[0], random_aug_pos1[1], random_aug_pos1[2]]), \
                                                np.around(cor_y[random_aug_pos1[0], random_aug_pos1[1], random_aug_pos1[2]])])
        sample['random_position_1'] = random_pos1
        sample['random_crop_image_1'] = image[random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                    random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                    random_pos1[2]:random_pos1[2] + self.output_size[2]]
        sample['random_fullsize_position_1'] = cor[:, random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                    random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                    random_pos1[2]:random_pos1[2] + self.output_size[2]]

        #print(random_pos0, random_pos1, (random_pos0-random_pos1)*spacing)
        for i in range(len(rela_position)):
            if sample['random_position_0'][i]<sample['random_position_1'][i]:
                rela_position[i] = 1
        spacing = np.asarray(spacing).squeeze()
        sample['rela_distance']=(random_pos0-random_pos1)*spacing
        sample['rela_fullsize_distance'] = (sample['random_fullsize_position_0']-sample['random_fullsize_position_1'])*spacing[:,np.newaxis,np.newaxis,np.newaxis]
        sample['rela_fullsize_aug_distance'] = (sample['random_fullsize_aug_position_0']-sample['random_fullsize_aug_position_1'])*spacing[:,np.newaxis,np.newaxis,np.newaxis]
        sample['consis_fullsize_distance_0'] = (sample['random_fullsize_position_0']-sample['random_fullsize_aug_position_0'])*spacing[:,np.newaxis,np.newaxis,np.newaxis]
        sample['consis_fullsize_distance_1'] = (sample['random_fullsize_position_1']-sample['random_fullsize_aug_position_1'])*spacing[:,np.newaxis,np.newaxis,np.newaxis]
        sample['rela_poi']=rela_position
        return sample

class RandomPositionDoublePatientCrop(object):
    """
    Randomly crop two images in one sample;
    distance is a vector(could be positive or pasitive), representing the vector
    from image1 to image2.
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, padding=True):
        self.output_size = output_size
        self.padding = padding

    def random_position(self, shape):
        position = []
        for i in range(len(shape)):
            position.append(np.random.randint(0, shape[i] - self.output_size[i]))
        #print('random result',shape, position)
        return np.asarray(position)

    def __call__(self, sample):
        for i in range(2):
            image,spacing= sample['image_{0:}'.format(i)],sample['spacing_{0:}'.format(i)]
            # pad the sample if necessary
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                    self.output_size[2]:
                pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
                pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.padding:
                image = np.pad(image, [(self.output_size[0]//2, self.output_size[0]//2), (self.output_size[1]//2,
                    self.output_size[1]//2), (self.output_size[2]//2, self.output_size[2]//2)], mode='constant', constant_values=0)
            shape = image.shape
            background_chosen = True
            while background_chosen:
                random_position = self.random_position(shape)
                if image[
                    random_position[0] + self.output_size[0] // 2, random_position[1] + self.output_size[1] // 2,
                    random_position[2] + self.output_size[2] // 2] >= 0.001:
                    background_chosen = False
            # print(i, random_position, shape,image[
            #     random_position[0] + self.output_size[0] // 2, random_position[1] + self.output_size[1] // 2,
            #     random_position[2] + self.output_size[2] // 2])
            image_patch = image[random_position[0]:random_position[0] + self.output_size[0],
                     random_position[1]:random_position[1] + self.output_size[1], random_position[2]:random_position[2] + self.output_size[2]]
            sample['random_crop_image_{0:}'.format(i)]=image_patch
            sample['random_position_{0:}'.format(i)]=random_position*np.asarray(spacing)
            sample['random_inher_position_{0:}'.format(i)]=random_position/np.asarray(shape)
        sample['rela_distance']=sample['random_position_0']-sample['random_position_1']
        rela_position = np.zeros(3)
        for i in range(len(rela_position)):
            rela_position[i]=(sample['random_inher_position_0'][i]-sample['random_inher_position_1'][i])*20
        sample['rela_poi'] = rela_position
        return sample

class RandomScale(object):
    def __init__(self, p=0.5, axes=(0,1), max_scale=1):
        self.p = p
        self.axes = axes
        self.max_scale = max_scale


    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            if isinstance(self.max_scale, numbers.Number):
                if self.max_scale < 0:
                    raise ValueError("If degrees is a single number, it must be positive.")
                scale = (1/self.max_scale, self.max_scale)
            else:
                if len(self.max_scale) != 2:
                    raise ValueError("If degrees is a sequence, it must be of len 2.")
                scale = self.max_scale
            scale = random.uniform(scale[0], scale[1])
            image = ndimage.zoom(image, scale,  order=0)
            label = ndimage.zoom(label, scale,  order=0)
            if 'coarseg' in sample:
                coarseg = ndimage.rotate(sample['coarseg'], scale, order=0)
                return {'image': image, 'label': label, 'coarseg': coarseg}
            else:
                return {'image': image, 'label': label}
        else:
            return sample


class ToPositionTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        nsample = {}
        key_float_list = ['rela_distance','rela_fullsize_distance','rela_fullsize_aug_distance','rela_poi','random_position','random_fullsize_position','random_fullsize_aug_position','consis_fullsize_distance']
        for key in sample.keys():
            if 'random_crop_image' in key or 'random_crop_aug_image' in key:
                image = sample[key]
                image= image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
                nsample[key] = torch.from_numpy(image)
            else:
                for key_float in  key_float_list:
                    if key_float in key:
                        nsample[key]=torch.from_numpy(sample[key]).float()
        return nsample

class ToDoublePatientPositionTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1 = sample['random_crop_image_0']
        image2 = sample['random_crop_image_1']
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2]).astype(np.float32)
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1], image2.shape[2]).astype(np.float32)
        sample['random_crop_image_0'] = torch.from_numpy(image1)
        sample['random_crop_image_1'] = torch.from_numpy(image2)
        sample['rela_distance'] = torch.from_numpy(sample['rela_distance']).float()
        sample['rela_poi'] = torch.from_numpy(sample['rela_poi']).float()
        sample['random_position_0'] = torch.from_numpy(sample['random_position_0']).float()
        sample['random_position_1'] = torch.from_numpy(sample['random_position_1']).float()
        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

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
