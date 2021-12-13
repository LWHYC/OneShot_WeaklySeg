#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
sys.path.append(os.path.abspath(__file__))  #返回当前.py文件的绝对路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.tensor
import torch.backends.cudnn as cudnn
from dataloaders.Position_multi_scale_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
import matplotlib.pyplot as plt
from data_process.data_process_func import *
from util.visualization.show_param import show_param
from prefetch_generator import BackgroundGenerator
from detection.detection_functions import *
import time



class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def test(config_file):#, label_wanted):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data = config['data']  # 包含数据的各种信息,如data_shape,batch_size等
    config_coarse_fnet = config['coarse_fnetwork']  # 网络参数,如net_name,base_feature_name,class_num等
    config_coarse_pnet = config['coarse_pnetwork']
    config_fine_fnet = config['fine_fnetwork']
    config_fine_pnet = config['fine_pnetwork']
    config_test = config['testing']

    patch_size = np.asarray([i for i in eval(config_data['patch_size']).values()])
    max_scale = np.max(patch_size, axis=0)
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed)  # 给定seed value,决定了后面的伪随机序列
    random_crop = RandomPositionDoublePatientCrop(patch_size, padding=False)
    to_tensor = ToDoublePatientPositionTensor()

    cudnn.benchmark = True
    cudnn.deterministic = True

    # 2, load data
    # print('2.Load data')
    validData = PositionDoublePatientDataloader(config=config_data,
                                                split='valid',
                                                transform=None)
    validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 3. creat model
    # print('3.Creat model')
    net_type = config_coarse_fnet['net_type']
    net_class = NetFactory.create(net_type)
    coarse_fnet = net_class(inc=config_coarse_fnet.get('input_channel', 1),
                     base_chns=config_coarse_fnet.get('base_feature_number', 16),
                     droprate=config_coarse_fnet.get('drop_rate', 0.2),
                     norm='in',
                     depth=config_coarse_fnet.get('depth', False),
                     dilation=config_coarse_fnet.get('dilation', 1),
                     )
    net_type = config_coarse_pnet['net_type']
    net_class = NetFactory.create(net_type)
    out_put_size = int(np.sum(patch_size.prod(axis=1)) / 512)  # 由于后面是全连接网络，所以coarse_fnet的输出需拉直
    coarse_pnet = net_class(
        inc=4 * config_coarse_pnet.get('base_feature_number', 16) * out_put_size,
        n_classes=config_coarse_pnet['class_num'],
        base_chns=config_coarse_pnet.get('base_feature_number', 16),
        droprate=config_coarse_pnet.get('drop_rate', 0.2),
    )
    net_type = config_fine_fnet['net_type']
    net_class = NetFactory.create(net_type)
    fine_fnet = net_class(inc=config_fine_fnet.get('input_channel', 1),
                     base_chns=config_fine_fnet.get('base_feature_number', 16),
                     droprate=config_fine_fnet.get('drop_rate', 0.2),
                     norm='in',
                     depth=config_fine_fnet.get('depth', False),
                     dilation=config_fine_fnet.get('dilation', 1),
                     )
    net_type = config_fine_pnet['net_type']
    net_class = NetFactory.create(net_type)
    out_put_size = int(np.sum(patch_size.prod(axis=1)) / 512)  # 由于后面是全连接网络，所以coarse_fnet的输出需拉直
    fine_pnet = net_class(
        inc=4 * config_fine_pnet.get('base_feature_number', 16) * out_put_size,
        n_classes=config_fine_pnet['class_num'],
        base_chns=config_fine_pnet.get('base_feature_number', 16),
        droprate=config_fine_pnet.get('drop_rate', 0.2),
        dis_range=30
    )
    coarse_fnet = torch.nn.DataParallel(coarse_fnet, device_ids=[0]).cuda()
    coarse_pnet = torch.nn.DataParallel(coarse_pnet, device_ids=[0]).cuda()
    fine_fnet = torch.nn.DataParallel(fine_fnet, device_ids=[0]).cuda()
    fine_pnet = torch.nn.DataParallel(fine_pnet, device_ids=[0]).cuda()
    if config_test['load_weight']:
        coarse_fnet_weight = torch.load(config_test['coarse_fnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # coarse_feature net
        coarse_pnet_weight = torch.load(config_test['coarse_pnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # position net
        fine_fnet_weight = torch.load(config_test['fine_fnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # coarse_feature net
        fine_pnet_weight = torch.load(config_test['fine_pnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # position net
        coarse_fnet.load_state_dict(coarse_fnet_weight)
        coarse_pnet.load_state_dict(coarse_pnet_weight)
        fine_fnet.load_state_dict(fine_fnet_weight)
        fine_pnet.load_state_dict(fine_pnet_weight)

    # show_param(coarse_fnet)  # 计算网络总参数量
    # show_param(coarse_pnet)

    # 4, start to detect
    # print('4.Start to detect')
    a = []
    b = []
    for ognb in [1,3,6,7]:
        label_wanted = ognb
        print('lw', label_wanted)
        show = False
        iou_sum = 0
        error_sum = np.zeros([2, 3])
        iter_patch_num = 15
        iter_move_num = 1
        coarse_feature = {}
        fine_feature = {}
        with torch.no_grad():
            coarse_fnet.eval(), coarse_pnet.eval()
            t0 = time.time()
            for ii_batch, sample_batch in enumerate(validLoader):
                label_0 = pad(load_volume_as_array(sample_batch['image_path_0'][0].replace('crop_norm_SLF1', 'crop_label')), max_scale)
                ss = label_0.shape
                label_1 = pad(load_volume_as_array(sample_batch['image_path_1'][0].replace('crop_norm_SLF1', 'crop_label')),max_scale)
                #print(sample_batch['image_path_0'], sample_batch['image_path_1'])
                sample_batch['image_0'] = pad(sample_batch['image_0'].cpu().data.numpy().squeeze(), max_scale)
                sample_batch['image_1'] = pad(sample_batch['image_1'].cpu().data.numpy().squeeze(), max_scale)
                real_extreme_cor = extract_certain_organ_cor(label_0, label_wanted=label_wanted,extreme_point_num=2)
                real_extreme_cor_2 = extract_certain_organ_cor(label_0, label_wanted=label_wanted,extreme_point_num=6)
                support_extreme_cor = extract_certain_organ_cor(label_1, label_wanted=label_wanted, extreme_point_num=6)
                predic_extreme_cor = np.zeros([support_extreme_cor.shape[0],3])
                t1 = time.time()
                for i in range(support_extreme_cor.shape[0]):
                    '''
                    分别预测几个极端点
                    '''
                    support_position = support_extreme_cor[i]
                    predic_position = []
                    for ii in range(patch_size.shape[0]):
                        support_batch = sample_batch['image_1'][support_position[0] - patch_size[ii, 0] // 2:support_position[0] + patch_size[ii,0] // 2,
                                    support_position[1] - patch_size[ii, 1] // 2:support_position[1] + patch_size[ii, 1] // 2,
                                    support_position[2] - patch_size[ii, 2] // 2:support_position[2] + patch_size[ii, 2] // 2]
                        support_batch = torch.from_numpy(support_batch).unsqueeze(dim=0).unsqueeze(dim=0).float()
                        coarse_support_feature = coarse_fnet(support_batch)['8x']
                        fine_support_feature = fine_fnet(support_batch)['8x']
                        if ii ==0:
                            coarse_feature[0] = coarse_support_feature.view(coarse_support_feature.shape[0], -1)  # 生成
                            fine_feature[0] = fine_support_feature.view(fine_support_feature.shape[0], -1)
                        else:
                            coarse_feature[0] = torch.cat(
                                [coarse_feature[0], coarse_support_feature.view(coarse_support_feature.shape[0], -1)], dim=1)
                            fine_feature[0] = torch.cat(
                                [fine_feature[0], fine_support_feature.view(fine_support_feature.shape[0], -1)], dim=1)
                    for ii in range(iter_patch_num):
                        '''
                        多次随机裁减预测距离，最终取平均
                    '''
                        sample = random_crop(sample_batch)
                        sample = to_tensor(sample)
                        cur_position = np.int16(sample['random_position_0']).squeeze()
                        initial_position = cur_position.copy()
                        for iiii in range(patch_size.shape[0]):
                            img_batch = sample_batch['image_0'][
                                        cur_position[0]//3 - patch_size[iiii, 0] // 2:cur_position[0]//3 + patch_size[
                                            iiii, 0] // 2,
                                        cur_position[1] - patch_size[iiii, 1] // 2:cur_position[1] + patch_size[
                                            iiii, 1] // 2,
                                        cur_position[2] - patch_size[iiii, 2] // 2:cur_position[2] + patch_size[
                                            iiii, 2] // 2]
                            img_batch = torch.from_numpy(img_batch).unsqueeze(dim=0).unsqueeze(dim=0).float()
                            answer_feature = coarse_fnet(img_batch)['8x']
                            if iiii == 0:
                                coarse_feature[1] = answer_feature.view(answer_feature.shape[0], -1)  # 生成
                            else:
                                coarse_feature[1] = torch.cat([coarse_feature[1], answer_feature.view(answer_feature.shape[0], -1)],
                                                    dim=1)
                        predic = coarse_pnet(coarse_feature[0], coarse_feature[1])['distance'].cpu().data.numpy().squeeze()
                        cur_position = np.round(cur_position + predic).astype(np.int16)
                        cur_position[0] = np.min((np.max((cur_position[0], 45)), 3*(ss[0]-15)))
                        cur_position[1] = np.min((np.max((cur_position[1], 63)), ss[1]-63))
                        cur_position[2] = np.min((np.max((cur_position[2], 63)), ss[2]-63))
                        fine_stage_image = pad(sample_batch['image_0'], patch_size[iiii])
                        for iii in range(iter_move_num):
                            cur_position = np.round(cur_position).astype(np.int16)
                            for iiii in range(patch_size.shape[0]):
                                img_batch = fine_stage_image[
                                            cur_position[0] // 3 :cur_position[0] // 3 + patch_size[iiii, 0] ,
                                            cur_position[1] :cur_position[1] + patch_size[iiii, 1],
                                            cur_position[2]:cur_position[2] + patch_size[iiii, 2]]
                                img_batch = torch.from_numpy(img_batch).unsqueeze(dim=0).unsqueeze(dim=0).float()
                                answer_feature = fine_fnet(img_batch)['8x']
                                if iiii == 0:
                                    fine_feature[1] = answer_feature.view(answer_feature.shape[0], -1)  # 生成
                                else:
                                    fine_feature[1] = torch.cat(
                                        [fine_feature[1], answer_feature.view(answer_feature.shape[0], -1)],
                                        dim=1)
                                predic = fine_pnet(fine_feature[0], fine_feature[1])['distance'].cpu().data.numpy().squeeze()
                                cur_position = cur_position+predic
                                cur_position[0] = np.min((np.max((cur_position[0], 45)), 3*(ss[0]-15)))
                                cur_position[1] = np.min((np.max((cur_position[1], 63)), ss[1]-63))
                                cur_position[2] = np.min((np.max((cur_position[2], 63)), ss[2]-63))
                        predic_position.append(cur_position)
                    t2 = time.time()
                    # print(t2-t0, t2-t1)
                    predic_position = cal_average_except_minmax(predic_position, False)
                    initial_position[0]= initial_position[0]/3 #因为预测的是物理距离，层间距是3mm，所以除以3
                    predic_position[0] = predic_position[0]/3
                    predic_extreme_cor[i]=predic_position
                    predic_position = np.round(predic_position).astype(np.int16)
                    # print('initial position',initial_position-[8,32,32], 'predicted position', predic_position-[8,32,32])
                    if show :
                        show_detection(sample_batch, support_position, initial_position=initial_position, predicted_position=predic_position)
                real_predic_extreme_cor = np.asarray([np.min(predic_extreme_cor,axis=0),np.max(predic_extreme_cor, axis=0)])
                pred_iou = iou(real_extreme_cor,  real_predic_extreme_cor)
                pred_error = np.abs(real_extreme_cor-real_predic_extreme_cor)
                # print('predic iou:',pred_iou, 'predic error:',pred_error)
                # print(real_extreme_cor_2-[8,32,32], '\n', real_predic_extreme_cor-[8,32,32])
                iou_sum+=pred_iou
                error_sum+=pred_error
            print('mean iou:', iou_sum/len(os.listdir(os.path.join(config_data.get('data_root'), 'valid'))))
            print('mean error:', np.mean(np.asarray([1.5,0.5,0.5])*np.sum(error_sum, axis=0)/len(
                os.listdir(os.path.join(config_data.get('data_root'), 'valid')))))
            print('mean error each:', np.asarray([3, 1, 1]) * error_sum/ len(
                os.listdir(os.path.join(config_data.get('data_root'), 'valid'))))
            a.append(iou_sum/len(os.listdir(os.path.join(config_data.get('data_root'), 'valid'))))
            b.append(np.mean(np.asarray([1.5,0.5,0.5])*np.sum(error_sum, axis=0)/len(
                os.listdir(os.path.join(config_data.get('data_root'), 'valid')))))
    print(np.mean(np.array(a)), np.mean(np.array(b)))
            



if __name__ == '__main__':

    
    # label_wanted = gggg # int(input('label_wanted:'))
    config_file = str('config/test_position_full_size_trible_stage.txt')
    assert (os.path.isfile(config_file))
    test(config_file)# label_wanted=label_wanted)
