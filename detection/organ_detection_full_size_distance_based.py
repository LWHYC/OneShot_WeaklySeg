#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
#os._exit(00)
import sys
from scipy.ndimage.measurements import label
sys.path.append(os.path.abspath(__file__))  #返回当前.py文件的绝对路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.tensor
import torch.backends.cudnn as cudnn
from dataloaders.Position_multi_scale_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from NetFactory import NetFactory
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from data_process.data_process_func import *
from util.visualization.show_param import show_param
from prefetch_generator import BackgroundGenerator
from detection.detection_functions import *
import numpy as np

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def test(config_file):#, label_wanted):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data = config['data']  # 包含数据的各种信息,如data_shape,batch_size等
    config_pnet = config['pnetwork']
    config_test = config['testing']

    patch_size = np.asarray(config_data['patch_size'])
    max_scale = patch_size
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed)  # 给定seed value,决定了后面的伪随机序列
    random_crop = RandomPositionDoublePatientCrop(patch_size, padding=False)
    to_tensor = ToDoublePatientPositionTensor()

    cudnn.benchmark = True
    cudnn.deterministic = True

    # 2, load data
    validData = PositionDoublePatientDataloader(config=config_data,
                                                split='valid',
                                                transform=None)
    validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    volume_num=len(os.listdir(os.path.join(config_data.get('data_root'), 'valid')))

    # 3. creat model
    net_type = config_pnet['net_type']
    net_class = NetFactory.create(net_type)
    pnet = net_class(
                    inc=config_pnet.get('input_channel', 1),
                    base_chns= config_pnet.get('base_feature_number', 16),
                    norm='in',
                    depth=config_pnet.get('depth', False),
                    dilation=config_pnet.get('dilation', 1),
                    n_classes = config_pnet['class_num'],
                    droprate=config_pnet.get('drop_rate', 0.2),
                    )
    pnet = torch.nn.DataParallel(pnet, device_ids=[0,1,2,3]).cuda()
    if config_test['load_weight']:
        pnet_weight = torch.load(config_test['pnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # position net
        pnet.load_state_dict(pnet_weight)
    #show_param(pnet)

    # 4, start to detect
    mean_iou_lis = []
    mean_error_lis = []
    for ognb in range(1,8):
        label_wanted = ognb
        print('lw', label_wanted)
        show = False
        iou_sum = 0
        error_sum = np.zeros([2, 3])
        iter_patch_num = 10
        iter_move_num = 1
        cor = {}
        with torch.no_grad():
            pnet.eval()
            for ii_batch, sample_batch in enumerate(validLoader):
                label_0 = pad(load_volume_as_array(sample_batch['image_path_0'][0].replace('crop_norm_SLF1', 'crop_label')), max_scale)
                ss = label_0.shape
                label_1 = pad(load_volume_as_array(sample_batch['image_path_1'][0].replace('crop_norm_SLF1', 'crop_label')),max_scale)
                # print(sample_batch['image_path_0'], sample_batch['image_path_1'])
                sample_batch['image_0'] = pad(sample_batch['image_0'].cpu().data.numpy().squeeze(), max_scale)
                sample_batch['image_1'] = pad(sample_batch['image_1'].cpu().data.numpy().squeeze(), max_scale)
                real_extreme_cor = extract_certain_organ_cor(label_0, label_wanted=label_wanted,extreme_point_num=2)
                support_extreme_cor = extract_certain_organ_cor(label_1, label_wanted=label_wanted, extreme_point_num=6)
                predic_extreme_cor = np.zeros([support_extreme_cor.shape[0],3])
                support_batch = []
                for i in range(support_extreme_cor.shape[0]):
                    '''
                    分别裁减几个support极端点所在patch，预测其坐标
                    '''
                    support_position = support_extreme_cor[i]
                    support_batch.append(sample_batch['image_1'][support_position[0] - patch_size[0] // 2:support_position[0] + patch_size[0] // 2,
                                support_position[1] - patch_size[1] // 2:support_position[1] + patch_size[1] // 2,
                                support_position[2] - patch_size[2] // 2:support_position[2] + patch_size[2] // 2][np.newaxis])
                support_batch = np.asarray(support_batch)
                support_cor = pnet(torch.from_numpy(support_batch).float())['position'].cpu().numpy().squeeze()
                center_support_patch = crop_patch_around_center(support_batch, r=[2,4,4])
                center_support_fg_mask = center_support_patch>0.1

                initial_position = []
                cur_position = []
                predic_position = []
                query_batch = []
                # for iii in range(iter_move_num):
                for ii in range(iter_patch_num):
                    '''
                    多次随机裁减预测距离，最终取平均
                    '''
                    sample = random_crop(sample_batch)
                    sample = to_tensor(sample)
                    random_position = np.int16(sample['random_position_0']).squeeze()
                    cur_position.append(random_position)
                    query_batch.append(sample_batch['image_0'][
                                    random_position[0]//3 - patch_size[0] // 2:random_position[0]//3 + patch_size[0] // 2,
                                    random_position[1] - patch_size[1] // 2:random_position[1] + patch_size[1] // 2,
                                    random_position[2] - patch_size[2] // 2:random_position[2] + patch_size[2] // 2][np.newaxis])
                query_batch = np.asarray(query_batch)
                initial_position = cur_position.copy()
                query_cor = pnet(torch.from_numpy(query_batch).float())['position'].cpu().numpy().squeeze()#[10,3,D,W,H]
                full_size_relative_position = 300*np.tanh(support_cor-np.mean(query_cor, axis=0)) #
                center_re_pos_patch = crop_patch_around_center(full_size_relative_position, r=[2,8,8])
                center_query_patch = crop_patch_around_center(query_batch, r = [2,8,8])
                center_point_relative_position = np.mean(center_re_pos_patch, axis=(2,3,4)) # [6, 3]
                cur_position = np.mean(np.asarray(cur_position), axis=0) + center_point_relative_position # [6, 3]

                for move_step in range(iter_move_num):
                    ''' 多走几次'''
                    cur_position[:,0] = np.minimum(np.maximum(cur_position[:,0], 45), 3*(ss[0]-15))
                    cur_position[:,1] = np.minimum(np.maximum(cur_position[:,1], 95), ss[1]-95)
                    cur_position[:,2] = np.minimum(np.maximum(cur_position[:,2], 95), ss[2]-95)
                    fine_query_batch = []
                    cur_position = np.int16(np.round(cur_position))
                    for iii in range(cur_position.shape[0]):
                        fine_query_batch.append(sample_batch['image_0'][
                                    cur_position[iii,0]//3 - patch_size[0] // 2:cur_position[iii,0]//3 + patch_size[0] // 2,
                                    cur_position[iii,1] - patch_size[1] // 2:cur_position[iii,1] + patch_size[1] // 2,
                                    cur_position[iii,2] - patch_size[2] // 2:cur_position[iii,2] + patch_size[2] // 2][np.newaxis])
                    fine_query_batch = np.asarray(fine_query_batch)
                    query_cor = pnet(torch.from_numpy(fine_query_batch).float())['position'].cpu().numpy().squeeze()#[6,3,D,W,H]
                    full_size_relative_position = 300*np.tanh(support_cor-query_cor) # [6,3,D,W,H]
                    center_re_pos_patch = crop_patch_around_center(full_size_relative_position, r=[2,8,8])
                    center_query_patch = crop_patch_around_center(query_batch, r = [2,8,8])
                    center_point_relative_position = np.mean(center_re_pos_patch, axis=(2,3,4)) # [6, 3]
                    cur_position = np.float16(cur_position)+ center_point_relative_position # [6, 3]
                predic_extreme_cor = cur_position.copy()
                predic_extreme_cor[:,0]= predic_extreme_cor[:,0]/3
                if show :
                    show_detection(sample_batch, support_position, initial_position=initial_position, predicted_position=predic_position)
                real_predic_extreme_cor = np.asarray([np.min(predic_extreme_cor,axis=0),np.max(predic_extreme_cor, axis=0)])
                pred_iou = iou(real_extreme_cor,  real_predic_extreme_cor)
                pred_error = np.abs(real_extreme_cor-real_predic_extreme_cor)
                #print('predic iou:',pred_iou, 'predic error:',pred_error)
                iou_sum+=pred_iou
                error_sum+=pred_error
            mean_iou = np.around(iou_sum/volume_num, decimals=3)
            mean_error = np.around(np.mean(np.asarray([1.5,0.5,0.5])*np.sum(error_sum, axis=0)/volume_num), decimals=2)
            mean_error_each = np.around(np.asarray([3, 1, 1]) * error_sum/ volume_num, decimals=2)
            print('mean iou:',mean_iou, 'mean error:',mean_error, 'mean error each:',mean_error_each)
            mean_iou_lis.append(mean_iou)
            mean_error_lis.append(np.mean(np.asarray([1.5,0.5,0.5])*np.sum(error_sum, axis=0)/len(
                os.listdir(os.path.join(config_data.get('data_root'), 'valid')))))
    print(np.mean(np.array(mean_iou_lis)), np.mean(np.array(mean_error_lis)))



if __name__ == '__main__':
    # for iii in [1,3,6,7]:
    #     label_wanted = iii #int(input('label_wanted:'))
    config_file = str('/home/oem/home/data/zhaoqianfei/Relative_Distance_Regression/Relative_Distance_Regression_v2/config/test_position_full_size.txt')
    assert (os.path.isfile(config_file))
    test(config_file)
