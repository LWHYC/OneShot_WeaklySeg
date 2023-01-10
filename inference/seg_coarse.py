#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
#os._exit(00)
import sys
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
from data_process.data_process_func import * 
from inference.localization_functions import *
from inference.geoseg import geode_seg
from inference.ssbr import selfsuport_bundaryrefine
from inference.msd import multi_simi_denoise
import numpy as np
import argparse
 
def test(config_file):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data = config['data'] 
    config_coarse_pnet = config['coarse_pnetwork']
    config_fine_pnet = config['fine_pnetwork']
    config_test = config['testing']
    save_seg_name = config_data['save_seg_name']
    
    patch_size = np.asarray(config_data['patch_size'])
    judge_noise = config_data['judge_noise']
    patch_size = patch_size
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed) 
    

    cudnn.benchmark = True
    cudnn.deterministic = True

    # 3. creat model
    coarse_net_type = config_coarse_pnet['net_type']
    coarse_net_class = NetFactory.create(coarse_net_type)
    fine_net_type = config_fine_pnet['net_type']
    fine_net_class = NetFactory.create(fine_net_type)

    Coarse_Pnet = coarse_net_class(
                    inc=config_coarse_pnet.get('input_channel', 1),
                    patch_size = patch_size,
                    base_chns= config_coarse_pnet.get('base_feature_number', 16),
                    norm='in',
                    depth=config_coarse_pnet.get('depth', False),
                    dilation=config_coarse_pnet.get('dilation', 1),
                    n_classes = config_coarse_pnet['class_num'],
                    )
    Fine_Pnet = fine_net_class(
                    inc=config_fine_pnet.get('input_channel', 1),
                    patch_size = patch_size,
                    base_chns= config_fine_pnet.get('base_feature_number', 16),
                    norm='in',
                    depth=config_fine_pnet.get('depth', False),
                    dilation=config_fine_pnet.get('dilation', 1),
                    n_classes = config_fine_pnet['class_num']
                    )
    Coarse_Pnet = torch.nn.DataParallel(Coarse_Pnet).half().cuda().eval()
    Fine_Pnet = torch.nn.DataParallel(Fine_Pnet).half().cuda().eval()
    if config_test['load_weight']:
        if os.path.isfile(config_test['coarse_pnet_load_path']):
            print("=> loading checkpoint '{}'".format(config_test['coarse_pnet_load_path']))
            print("=> loading checkpoint '{}'".format(config_test['fine_pnet_load_path']))
            checkpoint = torch.load(config_test['coarse_pnet_load_path'])
            Coarse_Pnet.load_state_dict(checkpoint['state_dict'])
            checkpoint = torch.load(config_test['fine_pnet_load_path'])
            Fine_Pnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' ".format(config_test['coarse_pnet_load_path']))
            print("=> loaded checkpoint '{}' ".format(config_test['fine_pnet_load_path']))
        else:
            raise(ValueError("=> no checkpoint found at '{}'".format(config_test['coarse_pnet_load_path'])))
    Coarse_RD = Relative_distance(Coarse_Pnet, config=config_data, distance_mode='tanh', \
                distance_ratio=config_coarse_pnet['distance_ratio'])
    Fine_RD = Relative_distance(Fine_Pnet, config=config_data, judge_noise=judge_noise,
                noise_wanted_ls = ['center_feature5', 'center_feature6', 'center_feature7', 'center_feature8'], 
                distance_mode='tanh', distance_ratio=config_fine_pnet['distance_ratio'])

    validData = Dataloader_(config=config_data, transform=None)
    validLoader = DataLoader(validData, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 4, project the scribble and seg
    
    with torch.no_grad():
        Coarse_RD.cal_support_position()
        Fine_RD.cal_support_position()
        for ii_batch, sample_batch in enumerate(validLoader):
            process_bar(ii_batch+1, len(validLoader))
            seg_save_path = sample_batch['image_path'][0].replace(sample_batch['image_path'][0].split('/')[-1], save_seg_name)
            spacing = sample_batch['spacing'].cpu().data.numpy().squeeze()

            ''' 1.Project the scribble and thresh it with Multi-Similarity-Denoising'''
            print('1.Project the scribble and thresh it with Multi-Similarity-Denoising')
            scribble_array = multi_simi_denoise(config_data, sample_batch, Coarse_RD, Fine_RD)

            ''' 2.Generate coarse seg with GeoS'''
            print('2.Generate coarse seg with GeoS', spacing)
            coarse_geoseg = geode_seg(scribble_array, sample_batch['image'].astype(np.float32), tuple(spacing), config_data['class_wanted'][:-1], config_data['class_wanted'][-1])
            nsample_batch = {}
            nsample_batch['coarseg'] = F.one_hot(torch.from_numpy(coarse_geoseg).long(), num_classes=np.max(coarse_geoseg)+1).permute(3,0,1,2)
            nsample_batch['image'] = torch.from_numpy(sample_batch['image']).unsqueeze(0)

            ''' 3.Refine the coarse seg with Self-Support Boundary Refinement'''
            print('3.Refine the coarse seg with Self-Support Boundary Refinement')
            seg = selfsuport_bundaryrefine(Fine_Pnet, nsample_batch, config_data['class_wanted'][:-1], config_data['stride'], patch_size)

            data = sitk.ReadImage(sample_batch['image_path'][0])
            seg_data = sitk.GetImageFromArray(seg.astype(np.int32))
            seg_data.SetSpacing(data.GetSpacing())
            seg_data.SetOrigin(data.GetOrigin())
            seg_data.SetDirection(data.GetDirection())
            sitk.WriteImage(seg_data, seg_save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='', help='Config file path')
    args = parser.parse_args()
    assert(os.path.isfile(args.config_path))
    test(args.config_path)
