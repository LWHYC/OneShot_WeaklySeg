import os
import torch
import torch.nn.functional as F
import numpy as np
import numbers
from tqdm import tqdm
from scipy import ndimage
from glob import glob
from torch.utils.data import Dataset
import random
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from data_process.data_process_func import *
import torchio as tio
from multiprocessing import Pool, cpu_count

def create_sample(image_path, label_path, coarseg_path=False, distance_path=False, class_num=2, transpose=True):
    image = torch.from_numpy(load_nifty_volume_as_array(image_path, transpose=transpose)[np.newaxis,:])
    label = load_nifty_volume_as_array(label_path, transpose=transpose).astype(np.int16)
    label[label>(class_num-1)]=0
    label = convert_to_one_hot(label, class_number=class_num)
    label = torch.from_numpy(label.astype(np.uint8))
    sample={'image':image,'label':label, 'patient_path':image_path}
    if coarseg_path:
        coarseg = load_nifty_volume_as_array(coarseg_path, transpose=transpose).astype(np.int16)
        coarseg = convert_to_one_hot(coarseg, class_number=class_num)
        coarseg = torch.from_numpy(coarseg.astype(np.uint8))
        sample['coarseg']=coarseg
    if distance_path:
        distance = torch.from_numpy(load_nifty_volume_as_array(distance_path, transpose=transpose)[np.newaxis,:])
        sample['distance']=distance
    return sample

class TorchioDataloader(Dataset):
    def __init__(self, config=None, class_num=3, split='train', num=None, transform=None, 
            random_sample=True, transpose=True, load_memory=True, image_fold_ls=[],label_fold=[]):
        self._data_root = config['data_root']
        self._image_filename = config['image_name']
        self._label_filename = config['label_name']
        self._coarseg_filename = config.get('coarseg_name', None)
        self._distance_filename = config.get('dis_name', None)
        self._iternum = config['iter_num']
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.random_sample = random_sample
        self.transpose = transpose
        self.folder_ls = os.path.join(self._data_root, split)
        self.image_dic = {}
        image_task_dic = {}
        self.load_memory = load_memory
        self.image_fold_ls = image_fold_ls
        self.label_fold = label_fold
        self.class_num = class_num
        self.image_name_list = []
        if split :
            if self.load_memory:
                p = Pool(cpu_count())
                if len(self.image_fold_ls)>0:
                    for image_fold in self.image_fold_ls:
                        self.image_name_list = os.listdir(os.path.join(self.folder_ls, image_fold))
                        for i in tqdm(range(len(self.image_name_list))):
                            image_path = os.path.join(self.folder_ls, image_fold, self.image_name_list[i])
                            label_path = os.path.join(self.folder_ls, self.label_fold, self.image_name_list[i])
                            
                            if self._coarseg_filename:
                                coarseg_path = os.path.join(self.folder_ls, image_fold, self.image_name_list[i])
                            else: 
                                coarseg_path = False
                            if self._distance_filename:
                                distance_path = os.path.join(self.folder_ls, 'dis_060606', self.image_name_list[i])
                            else:
                                distance_path = False
                            image_task_dic[image_fold+'_'+self.image_name_list[i]] = p.apply_async(create_sample, args=(image_path, label_path, coarseg_path, distance_path, class_num, transpose, ))
                        p.close()
                        p.join()
                else:
                    self.image_name_list = os.listdir(self.folder_ls)
                    for i in tqdm(range(len(self.image_name_list))):
                        image_path = os.path.join(self.folder_ls, self.image_name_list[i], self._image_filename)
                        label_path = os.path.join(self.folder_ls, self.image_name_list[i], self._label_filename)
                        if self._coarseg_filename:
                            coarseg_path = os.path.join(self.folder_ls,  self.image_name_list[i], self._coarseg_filename)
                        else: 
                            coarseg_path = False
                        if self._distance_filename:
                            distance_path = os.path.join(self.folder_ls, self.image_name_list[i], self._distance_filename)
                        else:
                            distance_path = False
                        image_task_dic[self.image_name_list[i]] = p.apply_async(create_sample, args=(image_path, label_path, coarseg_path, distance_path, class_num, ))
                    p.close()
                    p.join()
                for image_name in image_task_dic.keys():
                    self.image_dic[image_name]=image_task_dic[image_name].get()
            else:
                if len(self.image_fold_ls)>0:
                    self.image_name_list = os.listdir(os.path.join(self.folder_ls, self.image_fold_ls[0]))
                else:
                    self.image_name_list = os.listdir(os.path.join(self.folder_ls))
            if num is not None:
                self.image_name_list = self.image_name_list[:num]
                
        else:
            ValueError('please input choose correct mode! i.e."train" "valid" "test"')
        
        print("total {} samples".format(len(self.image_name_list)))

    def __len__(self):
        if self.random_sample == True:
            return self._iternum
        else:
            return len(self.image_name_list)

    def __getitem__(self, idx):
        if self.load_memory:
            if self.random_sample == True:
                image_name = random.sample(self.image_name_list, 1)[0]
            else:
                image_name = self.image_name_list[idx]
            sample = self.image_dic[image_name].copy()
        else:
            if self.random_sample == True:
                image_name = random.sample(self.image_name_list, 1)[0]
            else:
                image_name = self.image_name_list[idx]
            if len(self.image_fold_ls)>0:
                image_fold = random.sample(self.image_fold_ls, 1)[0]
                image_path = os.path.join(self.folder_ls, image_fold, image_name)
                label_path = os.path.join(self.folder_ls, self.label_fold, image_name)
            else:
                image_path = os.path.join(self.folder_ls, image_name, self._image_filename)
                label_path = os.path.join(self.folder_ls, image_name, self._label_filename)
            if self._coarseg_filename:
                coarseg_path = os.path.join(self.folder_ls,  image_name, self._coarseg_filename)
            else: 
                coarseg_path = False
            if self._distance_filename:
                distance_path = os.path.join(self.folder_ls, image_name, self._distance_filename)
            else:
                distance_path = False
            sample = create_sample(image_path, label_path, coarseg_path, distance_path, self.class_num)
        if self.transform:
            sample = self.transform(sample)
        return sample


class CropBound(object):
    def __init__(self, pad=[0,0,0], mode='label', class_determine=False):
        self.pad = pad
        self.mode = mode
        self.class_determine=class_determine
    def __call__(self, sample):
        # if self.mode=='label':
        if self.class_determine:
            file = sample[self.mode].index_select(0, torch.tensor(self.class_determine))
        else:
            file = sample[self.mode][1::]
        file = torch.sum(file, dim=0) 
        # else:
        #     nfile = sample[self.mode][0]
        #     if self.class_determine:
        #         file = torch.zeros_like(nfile)
        #         for i in range(len(self.class_determine)):
        #             file[nfile==self.class_determine[i]]=(i+1)
        #     else:
        #         file = nfile
        file_size = file.shape # DWH
        nonzeropoint = torch.as_tensor(torch.nonzero(file))
        maxpoint = torch.max(nonzeropoint, 0)[0].tolist()
        minpoint = torch.min(nonzeropoint, 0)[0].tolist()
        for i in range(len(self.pad)):
            maxpoint[i] = min(maxpoint[i] + self.pad[i], file_size[i])
            minpoint[i] = max(minpoint[i] - self.pad[i], 0)
        sample['minpoint']=minpoint
        sample['maxpoint']=maxpoint
        sample['shape'] = file_size
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key]=sample[key][:, minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        return sample

class ExtractCertainClass(object):
    def __init__(self, class_wanted=[1]):
        self.class_wanted = class_wanted
    def __call__(self, sample):
        label = sample['label']
        nlabel = label.index_select(0, torch.tensor([0]+self.class_wanted))
        sample ['label'] = nlabel
        if 'coarseg' in sample:
            ncoarseg = sample['coarseg'].index_select(0, torch.tensor([0]+self.class_wanted))
            sample ['coarseg'] = ncoarseg
                
        return sample

class ExtractCertainClassScribble(object):
    def __init__(self, class_wanted=[1]):
        self.class_wanted = class_wanted
    def __call__(self, sample):
        label = sample['label']
        nlabel = label.index_select(0, torch.tensor([0]+self.class_wanted[1::]))
        sample ['label'] = nlabel
        if 'coarseg' in sample:
            ncoarseg = sample['coarseg'].index_select(0, torch.tensor([0]+self.class_wanted[1::]))
            sample ['coarseg'] = ncoarseg
        return sample

class RandomNoiseAroundClass(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, sample):
        image = sample['image']
        noise = torch.clamp(self.sigma * torch.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        sample['image'] = image
        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        cshape = sample['image'].shape[1::] # DWH
        # pad the sample if necessary
        if cshape[0] <= self.output_size[0] or cshape[1] <= self.output_size[1] or cshape[2] <= \
                self.output_size[2]:
            #print(cshape)
            orishape = cshape
            pw = max((self.output_size[0] - cshape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - cshape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - cshape[2]) // 2 + 3, 0)
            for key in sample.keys():
                if torch.is_tensor(sample[key]):
                    sample[key] = F.pad(sample[key], (pd, pd, ph, ph, pw, pw), mode='constant', value=0)
            #print('Origin shape:', orishape,'Padded shape:', sample['image'].shape)
        (w, h, d) = sample['image'].shape[1::]
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key]=sample[key][:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return sample


class RandomNoise(object):
    def __init__(self, mean=0, std=0.1,include=['image'], prob=0):
        self.prob = prob
        self.add_noise = tio.RandomNoise(mean=mean, std=std, include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample= self.add_noise(sample)
        return sample


class RandomFlip(object):
    def __init__(self, include=['image'], flip_probability=0):
        self.flip_probability = flip_probability
        self.include = include
    def __call__(self, sample):
        axes = np.random.randint(0, 2)
        flip = tio.RandomFlip( axes=axes, flip_probability=self.flip_probability, include = self.include)
        sample= flip(sample)
        return sample

class RandomSpike(object):
    def __init__(self, num_spikes=3, intensity=1.2,include=['image'], prob=0):
        self.prob = prob
        self.add_spike = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity,include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_spike(sample)
        return sample

class RandomGhosting(object):
    def __init__(self, intensity=0.8, include=['image'], prob=0):
        self.prob = prob
        self.add_ghost = tio.RandomGhosting(intensity=intensity, include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_ghost(sample)
        return sample

class RandomElasticDeformation(object):
    def __init__(self, num_control_points=[5,10,10], max_displacement=[7,7,7], include=['image','label'], prob=0):
        self.prob = prob
        self.add_elas = tio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement = max_displacement,
            include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_elas(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, concat_coarseg=False, concat_distance=False):
        self.concat_coarseg=concat_coarseg
        self.concat_distance=concat_distance
    def __call__(self, sample):
        sample['label']=torch.max(sample['label'], 0)[1].long()
        if 'onehot_label' in sample:
            sample['onehot_label']=torch.from_numpy(sample['onehot_label']).long()
        if 'coarseg' in sample:
            sample['coarseg']=torch.max(sample['coarseg'], 0)[1].long()
            if self.concat_coarseg:
                sample['image']=torch.cat((sample['image'],sample['coarseg'].float()), 0)
        if 'distance' in sample:
            if self.concat_distance:
                sample['image']=torch.cat((sample['image'],sample['distance']), 0)
        return sample
