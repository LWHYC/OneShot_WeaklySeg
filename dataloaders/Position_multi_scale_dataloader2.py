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
random.seed(1)
import h5py
import itertools
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data.sampler import Sampler
from data_process.data_process_func import *

class PositionDataloader(Dataset):
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
            image_fold = random.sample(self.image_name_list, 1)[0]
        else:
            image_fold = self.image_name_list[idx]
        image_path = os.path.join(self._data_root, self.split, image_fold, self._image_filename)
        image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
        spacing = np.asarray(spacing)
        sample = {'image': image, 'spacing':spacing, 'patient_path':image_path}
        if self.transform:
            sample = self.transform(sample)

        return sample

class PositionDoublePatientDataloader(Dataset):
    """ structseg Dataset position """
    def __init__(self, config=None, split='train', num=None, transform=None):
        self._data_root = config['data_root']
        self._image_filename = config['image_name']
        self._iternum = config['iter_num']
        self.support_image = config['support_image']
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.image_dic_list = []
        if split:
            self.image_name_list = os.listdir(self._data_root+'/'+split)
        else:
            ValueError('please input choose correct mode! i.e."train" "valid" "test"')

        if num is not None:
            self.image_name_list = self.image_name_list[:num]
        print("total {} samples".format(len(self.image_name_list)))

    def __len__(self):
        # if self.split == 'train':
        #     return self._iternum
        # else:
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # if self.split == 'train':
        #     double_image_path = random.sample(self.image_name_list, 2)
        # else:
        double_image_path = [self.image_name_list[idx], self.image_name_list[0]]
        if self._image_filename is None:
            image_path_0 = os.path.join(self._data_root, self.split, double_image_path[0])
            image_path_1 = self.support_image
        else:
            image_path_0 = os.path.join(self._data_root, self.split, double_image_path[0], self._image_filename)
            image_path_1 = os.path.join(self.support_image, self._image_filename)
        image_0, spacing_0 = load_nifty_volume_as_array(image_path_0, return_spacing=True)
        spacing_0 = np.asarray(spacing_0)
        image_1, spacing_1 = load_nifty_volume_as_array(image_path_1, return_spacing=True)
        spacing_1 = np.asarray(spacing_1)
        sample = {'image_0': image_0.astype(np.float16), 'spacing_0':spacing_0, 'spacing_1':spacing_1, 'image_1':image_1.astype(np.float16),
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

    def __init__(self, output_size, padding=True, foreground_only=True, elastic_prob=0,scale_prob=0, scale_ratio=0.2):
        self.output_size = np.asarray(output_size)
        self.max_outsize_each_axis = np.max(self.output_size, axis=0)  # 各方向最大的裁减尺度
        self.padding = padding
        self.foregroung_only = foreground_only
        self.elastic_prob = elastic_prob
        self.scale_prob = scale_prob
        self.scale_ratio = scale_ratio
    def random_cor(self, shape):
        position = []
        for i in range(len(shape)):
            position.append(np.random.randint(self.max_outsize_each_axis[i]//2, shape[i] - self.max_outsize_each_axis[i]//2))
        return np.asarray(position)

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         alpha 控制高斯模糊的幅度, 越大，总体
         sigma控制高斯模糊的方差，越小，坐标模糊时局部弹性变化越剧烈
         alpha_affine控制仿射变换的幅度，越大，放射变换幅度可能越大
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape
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

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                             sigma) * alpha  # alpha 控制高斯模糊的幅度,sigma控制高斯模糊的方差，越小局部变化越剧烈
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 原图中每个格点的x,y,z坐标
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))  # 弹性变换后的坐标
        image = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
        return image, M, indices

    def __call__(self, sample):
        image,spacing= sample['image'],sample['spacing']
        # pad the sample if necessary
        if self.padding:
            image = np.pad(image, [(self.max_outsize_each_axis[0] // 2, self.max_outsize_each_axis[0] // 2), (self.max_outsize_each_axis[1] // 2,
                                                                                            self.max_outsize_each_axis[1] // 2),
                                     (self.max_outsize_each_axis[2] // 2, self.max_outsize_each_axis[2] // 2)], mode='constant',
                             constant_values=0)
        if image.shape[0] <= self.max_outsize_each_axis[0] or image.shape[1] <= self.max_outsize_each_axis[1] or image.shape[2] <= \
                self.max_outsize_each_axis[2]:
            pw = max((self.max_outsize_each_axis[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.max_outsize_each_axis[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.max_outsize_each_axis[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        relative_position = np.zeros(3)

        shape = list(image.shape)
        background_chosen = True
        while background_chosen:
            random_pos0 = self.random_cor(shape)
            if image[random_pos0[0] , random_pos0[1] , random_pos0[2]]!=0:
                background_chosen = False
        sample['random_position_0'] = random_pos0.copy()
        for i in range(self.output_size.shape[0]):
            sample['random_crop_image_0_{0:}'.format(i)]=image[
                                                         random_pos0[0]-self.output_size[i,0]//2:random_pos0[0] + self.output_size[i,0]//2,
                                                        random_pos0[1]-self.output_size[i,1]//2:random_pos0[1] + self.output_size[i,1]//2,
                                                        random_pos0[2]-self.output_size[i,2]//2:random_pos0[2] + self.output_size[i,2]//2]

            '''
            如随机数超过elastic_rate，进行弹性变化
            如进行弹性变换，将第一步裁减图片的坐标投影至弹性变换后的坐标系
            '''
            image_t = image.copy()
            if np.random.uniform(0, 1) < self.elastic_prob:
                image_t = image_t.transpose(2, 1, 0)  # 转到x,y,z，因为z方向出来会有镜像情况。进行变换
                image_t, M, indices = self.elastic_transform(image_t, image_t.shape[1], image_t.shape[1] * 0.06,
                                                            image_t.shape[1] * 0.03)
                image_t = image_t.transpose(2, 1, 0)
                indices = list(indices)
                rshape = shape.copy()
                rshape.reverse()
                indices[0] = indices[0].reshape(rshape)
                indices[1] = indices[1].reshape(rshape)
                rp = np.int16(np.round(np.matmul(M, np.asarray([random_pos0[2], random_pos0[1]]+[1]))))
                rp=np.int16(np.round([indices[1][rp[0],rp[1], random_pos0[0]], indices[0][rp[0],rp[1], random_pos0[0]]]))
                random_pos0[1:]=rp

                '''
                如随机数超过scale_rate，进行缩放变化
                如进行缩放变换，将第一步裁减图片的坐标投影至弹性变换后的坐标
              '''
            if np.random.uniform(0, 1) < self.scale_prob:
                zoomfactor = np.random.uniform(1-self.scale_ratio, 1+self.scale_ratio,size=3)
                image_t = ndimage.zoom(image_t, zoom=zoomfactor, order=1)
                random_pos0 = np.int16(np.round(random_pos0*zoomfactor))

            background_chosen = True
            shape_t = image_t.shape
            while background_chosen:
                random_pos1= self.random_cor(shape_t)
                if image_t[random_pos1[0] , random_pos1[1] , random_pos1[2] ] != 0:
                    background_chosen = False
            sample['random_position_1'] = random_pos1
            for ii in range(self.output_size.shape[0]):
                sample['random_crop_image_1_{0:}'.format(ii)] = image_t[
                                                                random_pos1[0] - self.output_size[ii, 0] // 2:
                                                                random_pos1[0] + self.output_size[ii, 0] // 2,
                                                                random_pos1[1] - self.output_size[ii, 1] // 2:
                                                                random_pos1[1] + self.output_size[ii, 1] // 2,
                                                                random_pos1[2] - self.output_size[ii, 2] // 2:
                                                                random_pos1[2] + self.output_size[ii, 2] // 2]

            for i in range(len(relative_position)):
                if sample['random_position_0'][i]<sample['random_position_1'][i]:
                    relative_position[i] = 1
            spacing = np.asarray(spacing)
            sample['rela_distance']=(random_pos0-random_pos1)*spacing
            sample['rela_poi']=relative_position
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
        self.max_outsize_each_axis = self.output_size # 各方向最大的裁减尺度

    def random_cor(self, shape):
        position = []
        for i in range(len(shape)):
            position.append(np.random.randint(shape[i]//2-10, shape[i]//2+10))
            #position.append(np.random.randint(self.max_outsize_each_axis[i]//2, shape[i] - self.max_outsize_each_axis[i]//2))
        return np.asarray(position)

    def __call__(self, sample):
        for i in range(2):
            image,spacing= sample['image_{0:}'.format(i)],sample['spacing_{0:}'.format(i)]
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
            sample['random_position_{0:}'.format(i)]=random_cor*np.asarray(spacing)
            sample['random_inher_position_{0:}'.format(i)]=random_cor/np.asarray(shape)
            image_patch = image[random_cor[0]-self.output_size[0]//2:random_cor[0] + self.output_size[0]//2,
                                random_cor[1]-self.output_size[1]//2:random_cor[1] + self.output_size[1]//2,
                                random_cor[2]-self.output_size[2]//2:random_cor[2] + self.output_size[2]//2]
            sample['random_crop_image_{0:}'.format(i)]=image_patch

        sample['rela_distance']=sample['random_position_0']-sample['random_position_1']
        relative_position = np.zeros(3)
        for i in range(len(relative_position)):
            relative_position[i]=(sample['random_inher_position_0'][i]-sample['random_inher_position_1'][i])*20
        sample['rela_poi'] = relative_position
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
        for key in sample.keys():
            if 'random_crop_image' in key:
                image = sample[key]
                image= image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
                nsample[key] = torch.from_numpy(image)
        nsample['rela_distance'] = torch.from_numpy(sample['rela_distance']).float()
        nsample['rela_poi'] = torch.from_numpy(sample['rela_poi']).float()
        nsample['random_position_0'] = torch.from_numpy(sample['random_position_0']).float()
        nsample['random_position_1'] = torch.from_numpy(sample['random_position_1']).float()
        return nsample

class ToDoublePatientPositionTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        for key in sample.keys():
            if 'random_crop_image' in key:
                image = sample[key]
                image= image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
                sample[key] = torch.from_numpy(image)
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
