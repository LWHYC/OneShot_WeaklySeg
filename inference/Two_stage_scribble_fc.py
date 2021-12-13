#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
#os._exit(00)
import sys
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.ndimage.measurements import label
import torch
import torch.backends.cudnn as cudnn
from dataloaders.Position_multi_scale_dataloader2 import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from data_process.data_process_func import * 
from prefetch_generator import BackgroundGenerator
from RDR_Seg_public.inference.localization_functions import *
import numpy as np

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def test(config_file):#, label_wanted):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data = config['data'] 
    config_coarse_pnet = config['coarse_pnetwork']
    config_fine_pnet = config['fine_pnetwork']
    coarse_dis_ratio = config_coarse_pnet['distance_ratio']
    fine_dis_ratio = config_fine_pnet['distance_ratio']
    config_test = config['testing']
    image_name = config_data['image_name']
    scribble_name = config_data['scribble_name']
    save_scribble_name = config_data['save_scribble_name']
    judge_noise = config_data['judge_noise']
    save_noise_name = config_data['save_noise_name']
    patch_size = np.asarray(config_data['patch_size'])
    max_scale = patch_size
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed) 
    random_crop = RandomPositionDoublePatientCrop(patch_size, padding=False)
    to_tensor = ToDoublePatientPositionTensor()

    cudnn.benchmark = True
    cudnn.deterministic = True

    # 2, load data

        # 3. creat model
    coarse_net_type = config_coarse_pnet['net_type']
    coarse_net_class = NetFactory.create(coarse_net_type)
    fine_net_type = config_fine_pnet['net_type']
    fine_net_class = NetFactory.create(fine_net_type)
    patch_size = np.asarray(config_data['patch_size'])
    Coarse_Pnet = coarse_net_class(
                    inc=config_coarse_pnet.get('input_channel', 1),
                    patch_size = patch_size,
                    base_chns= config_coarse_pnet.get('base_feature_number', 16),
                    norm='in',
                    depth=config_coarse_pnet.get('depth', False),
                    dilation=config_coarse_pnet.get('dilation', 1),
                    n_classes = config_coarse_pnet['class_num'],
                    droprate=config_coarse_pnet.get('drop_rate', 0.2),
                    )
    Fine_Pnet = fine_net_class(
                    inc=config_fine_pnet.get('input_channel', 1),
                    patch_size = patch_size,
                    base_chns= config_fine_pnet.get('base_feature_number', 16),
                    norm='in',
                    depth=config_fine_pnet.get('depth', False),
                    dilation=config_fine_pnet.get('dilation', 1),
                    n_classes = config_fine_pnet['class_num'],
                    droprate=config_fine_pnet.get('drop_rate', 0.2),
                    )
    Coarse_Pnet = torch.nn.DataParallel(Coarse_Pnet).half().cuda()
    Fine_Pnet = torch.nn.DataParallel(Fine_Pnet).half().cuda()
    if config_test['load_weight']:
        coarse_pnet_weight = torch.load(config_test['coarse_pnet_load_path'],
                                map_location=lambda storage, loc: storage) 
        Coarse_Pnet.load_state_dict(coarse_pnet_weight)
        fine_pnet_weight = torch.load(config_test['fine_pnet_load_path'],
                                map_location=lambda storage, loc: storage) 
        Fine_Pnet.load_state_dict(fine_pnet_weight)
    Coarse_RD = Relative_distance(Coarse_Pnet,out_mode='fc_position', distance_mode='tanh', \
                center_patch_size=[2,8,8], distance_ratio=coarse_dis_ratio)
    Fine_RD = Relative_distance(Fine_Pnet,out_mode='fc_position', feature_refine=False, judge_noise=judge_noise, distance_mode='tanh', \
                center_patch_size=[2,8,8], distance_ratio=fine_dis_ratio)

    for mode in ['train','valid']:
        validData = PositionDoublePatientDataloader(config=config_data,
                                                    split=mode,
                                                    transform=None)
        validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        volume_num=len(os.listdir(os.path.join(config_data.get('data_root'), mode)))

        # 4, start to detect
        mean_iou_lis = []
        mean_error_lis = []
        error_dis = []
        show = False
        iter_patch_num = 5
        iter_move_num = 1
        with torch.no_grad():
            Coarse_Pnet.eval(), Fine_Pnet.eval()
            for ii_batch, sample_batch in enumerate(validLoader):
                print(sample_batch['image_path_0'])
                spacing = sample_batch['spacing_0'].cpu().data.numpy().squeeze()
                scribble_0 = np.zeros_like(sample_batch['image_0'].cpu().data.numpy().squeeze())
                scribble_0_path = sample_batch['image_path_0'][0].replace(image_name, save_scribble_name)#label_name)
                noise_0 = np.zeros_like(sample_batch['image_0'].cpu().data.numpy().squeeze())
                noise_0_path = sample_batch['image_path_0'][0].replace(image_name, save_noise_name)
                scribble_1 = pad(load_volume_as_array(sample_batch['image_path_1'][0].replace(image_name, scribble_name)),max_scale)
                sample_batch['image_0'] = pad(sample_batch['image_0'].cpu().data.numpy().squeeze(), max_scale)
                sample_batch['image_1'] = pad(sample_batch['image_1'].cpu().data.numpy().squeeze(), max_scale)
                ss = sample_batch['image_0'].shape
                step = config_data['step']
                class_wanted = config_data['class_wanted']
                for class_cur in range(len(class_wanted)):
                    ognb = class_wanted[class_cur]
                    print('lw', ognb)
                    support_mask = np.array(np.where(scribble_1==ognb)).transpose()
                    for iii in range(0, support_mask.shape[0]-1, 32):
                        
                        '''crop several support patch'''
                        support_batch = []
                        cur_support_cor = support_mask[iii:min(iii+32, support_mask.shape[0]):step[class_cur]] # 4,3
                        for i in range(cur_support_cor.shape[0]):
                            support_cor = cur_support_cor[i]
                            support_batch.append(sample_batch['image_1'][support_cor[0] - patch_size[0] // 2:support_cor[0] + patch_size[0] // 2,
                                        support_cor[1] - patch_size[1] // 2:support_cor[1] + patch_size[1] // 2,
                                        support_cor[2] - patch_size[2] // 2:support_cor[2] + patch_size[2] // 2][np.newaxis])
                        support_batch = np.asarray(support_batch)
                        Coarse_RD.cal_support_position(support_batch)
                        Fine_RD.cal_support_position(support_batch)
                        cur_position,predic_position,query_batch = [],[],[]

                        '''randomly select several initial points'''
                        for ii in range(iter_patch_num):
                            sample = random_crop(sample_batch)
                            random_position = np.int16(sample['random_position_0']).squeeze()
                            random_cor = np.around(random_position/spacing).astype(np.int16)
                            cur_position.append(random_position)
                            query_batch.append(sample_batch['image_0'][
                                            random_cor[0] - patch_size[0] // 2:random_cor[0] + patch_size[0] // 2,
                                            random_cor[1] - patch_size[1] // 2:random_cor[1] + patch_size[1] // 2,
                                            random_cor[2] - patch_size[2] // 2:random_cor[2] + patch_size[2] // 2][np.newaxis])
                        query_batch = np.asarray(query_batch) #[iter_patch_num,1,d,w,h]
                        relative_position = Coarse_RD.cal_RD(query_patch=query_batch, mean=True)['relative_position']
                        cur_position = np.mean(np.asarray(cur_position), axis=0) + relative_position #

                        ''' fine movement'''
                        for move_step in range(iter_move_num):
                            for dim in range(3):
                                cur_position[:,dim] = np.minimum(np.maximum(cur_position[:,dim], spacing[dim]*patch_size[dim]//2), spacing[dim]*(ss[dim]-patch_size[dim]//2-1))
                            fine_query_batch = []
                            cur_cor = np.around(cur_position/spacing).astype(np.int16) #像素坐标
                            
                            for iiii in range(cur_position.shape[0]):
                                fine_query_batch.append(sample_batch['image_0'][
                                            cur_cor[iiii,0] - patch_size[0] // 2:cur_cor[iiii,0] + patch_size[0] // 2,
                                            cur_cor[iiii,1] - patch_size[1] // 2:cur_cor[iiii,1] + patch_size[1] // 2,
                                            cur_cor[iiii,2] - patch_size[2] // 2:cur_cor[iiii,2] + patch_size[2] // 2][np.newaxis,:])
                            fine_query_batch = np.asarray(fine_query_batch)
                            relative_position = Fine_RD.cal_RD(fine_query_batch)['relative_position']
                            cur_position = np.float16(cur_position)+ relative_position # [6,3]
                        predic_cor = np.around(cur_position.copy()/spacing).astype(np.int16) #[6,3]
                        for dim in range(3):
                            predic_cor[:, dim] = np.minimum(np.maximum(predic_cor[:,dim], max_scale[dim]//2), ss[dim]-max_scale[dim]//2-1)    
                        
                        ''' judge noise'''
                        if judge_noise:
                            noise_query_batch = []
                            for iii in range(predic_cor.shape[0]):
                                noise_query_batch.append(sample_batch['image_0'][
                                            predic_cor[iii,0] - patch_size[0] // 2:predic_cor[iii,0] + patch_size[0] // 2,
                                            predic_cor[iii,1] - patch_size[1] // 2:predic_cor[iii,1] + patch_size[1] // 2,
                                            predic_cor[iii,2] - patch_size[2] // 2:predic_cor[iii,2] + patch_size[2] // 2][np.newaxis,:])
                            noise_query_batch = np.asarray(noise_query_batch)
                            noise = Fine_RD.cal_noise(noise_query_batch)

                        ''' project cor and noise'''
                        predic_cor = predic_cor.tolist()
                        for iiii in range(len(predic_cor)):
                            scribble_0[predic_cor[iiii][0]-max_scale[0]//2,predic_cor[iiii][1]-max_scale[1]//2,predic_cor[iiii][2]-max_scale[2]//2]=ognb
                            if judge_noise:
                                noise_0[predic_cor[iiii][0]-max_scale[0]//2,predic_cor[iiii][1]-max_scale[1]//2,predic_cor[iiii][2]-max_scale[2]//2]=noise[iiii]

                print(scribble_0_path)
                save_array_as_nifty_volume(scribble_0.astype(np.float32), scribble_0_path)#, pixel_spacing = spacing.tolist() )
                if judge_noise:
                    save_array_as_nifty_volume(noise_0.astype(np.float32), noise_0_path)

    



if __name__ == '__main__':
    config_file = str('../config/test_scribble_full_size_double_stage_abdomen.txt')
    assert (os.path.isfile(config_file))
    test(config_file) 
