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
import numpy as np


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def test(config_file):#, label_wanted):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data = config['data']  # 包含数据的各种信息,如data_shape,batch_size等
    config_fnet = config['fnetwork']  # 网络参数,如net_name,base_feature_name,class_num等
    config_pnet = config['pnetwork']
    config_test = config['testing']

    patch_size = np.asarray([i for i in eval(config_data['patch_size']).values()])
    max_scale = np.max(patch_size, axis=0)
    # print(max_scale)
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
    net_type = config_fnet['net_type']
    net_class = NetFactory.create(net_type)
    fnet = net_class(inc=config_fnet.get('input_channel', 1),
                     base_chns=config_fnet.get('base_feature_number', 16),
                     droprate=config_fnet.get('drop_rate', 0.2),
                     norm='in',
                     depth=config_fnet.get('depth', False),
                     dilation=config_fnet.get('dilation', 1),
                     )
    net_type = config_pnet['net_type']
    net_class = NetFactory.create(net_type)
    out_put_size = int(np.sum(patch_size.prod(axis=1)) / 512)  # 由于后面是全连接网络，所以Fnet的输出需拉直
    pnet = net_class(
        inc=4 * config_pnet.get('base_feature_number', 16) * out_put_size,
        n_classes=config_pnet['class_num'],
        base_chns=config_pnet.get('base_feature_number', 16),
        droprate=config_pnet.get('drop_rate', 0.2),
    )
    fnet = torch.nn.DataParallel(fnet, device_ids=[0, 1]).cuda()
    pnet = torch.nn.DataParallel(pnet, device_ids=[0, 1]).cuda()
    if config_test['load_weight']:
        fnet_weight = torch.load(config_test['fnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # feature net
        pnet_weight = torch.load(config_test['pnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # position net
        fnet.load_state_dict(fnet_weight)
        pnet.load_state_dict(pnet_weight)


    # 4, start to detect
    a = []
    b = []
    for ognb in [1,3,6,7]:
        label_wanted = ognb
        print('lw', label_wanted)
        show = False
        iou_sum = 0
        error_sum = np.zeros([2, 3])
        iter_patch_num = 15
        iter_move_num = 3
        feature = {}
        with torch.no_grad():
            fnet.eval(), pnet.eval()
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
                        support_feature = fnet(support_batch)['8x']
                        if ii ==0:
                            feature[0] = support_feature.view(support_feature.shape[0], -1)  # 生成
                        else:
                            feature[0] = torch.cat([feature[0], support_feature.view(support_feature.shape[0], -1)], dim=1)
                    for ii in range(iter_patch_num):
                        '''
                        多次随机裁减预测距离，最终取平均
                    '''
                        sample = random_crop(sample_batch)
                        sample = to_tensor(sample)
                        cur_position = np.int16(sample['random_position_0']).squeeze()
                        initial_position = cur_position.copy()
                        for iii in range(iter_move_num):
                            for iiii in range(patch_size.shape[0]):
                                img_batch = sample_batch['image_0'][
                                            cur_position[0]//3 - patch_size[iiii, 0] // 2:cur_position[0]//3 + patch_size[
                                                iiii, 0] // 2,
                                            cur_position[1] - patch_size[iiii, 1] // 2:cur_position[1] + patch_size[
                                                iiii, 1] // 2,
                                            cur_position[2] - patch_size[iiii, 2] // 2:cur_position[2] + patch_size[
                                                iiii, 2] // 2]
                                img_batch = torch.from_numpy(img_batch).unsqueeze(dim=0).unsqueeze(dim=0).float()
                                answer_feature = fnet(img_batch)['8x']
                                if iiii == 0:
                                    feature[1] = answer_feature.view(support_feature.shape[0], -1)  # 生成
                                else:
                                    feature[1] = torch.cat([feature[1], answer_feature.view(answer_feature.shape[0], -1)],
                                                        dim=1)
                                predic = pnet(feature[0], feature[1])['distance'].cpu().data.numpy().squeeze()
                                cur_position = np.round(cur_position + predic).astype(np.int16)

                                cur_position[0] = np.min((np.max((cur_position[0], 45)), 3*(ss[0]-15)))
                                cur_position[1] = np.min((np.max((cur_position[1], 63)), ss[1]-63))
                                cur_position[2] = np.min((np.max((cur_position[2], 63)), ss[2]-63))

                                #print(iii, 'cur position',cur_position)
                        predic_position.append( cur_position)
                    predic_position = cal_average_except_minmax(predic_position, extract_m=False)
                    initial_position[0]= initial_position[0]/3 #因为预测的是物理距离，层间距是3mm，所以除以3
                    predic_position[0] = predic_position[0]/3
                    predic_extreme_cor[i]=predic_position
                    predic_position = np.int16(np.round(predic_position))
                    #print('initial position',initial_position, 'predicted position', predic_position)
                    if show :
                        show_detection(sample_batch, support_position, initial_position=initial_position, predicted_position=predic_position)
                real_predic_extreme_cor = np.asarray([np.min(predic_extreme_cor,axis=0),np.max(predic_extreme_cor, axis=0)])
                pred_iou = iou(real_extreme_cor,  real_predic_extreme_cor)
                pred_error = np.abs(real_extreme_cor-real_predic_extreme_cor)
                #print('predic iou:',pred_iou, 'predic error:',pred_error)
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
    config_file = str('config/test_position.txt')
    assert (os.path.isfile(config_file))
    test(config_file)
