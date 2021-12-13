#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
#os._exit(00)
import sys
sys.path.append(os.path.abspath(__file__))  #返回当前.py文件的绝对路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.ndimage.measurements import label
import torch
import torch.tensor
import torch.backends.cudnn as cudnn
from dataloaders.Position_multi_scale_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
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
    config_coarse_pnet = config['coarse_pnetwork']
    config_middle_pnet = config['middle_pnetwork']
    config_fine_pnet = config['fine_pnetwork']
    coarse_dis_ratio = config_coarse_pnet['distance_ratio']
    middle_dis_ratio = config_middle_pnet['distance_ratio']
    fine_dis_ratio = config_fine_pnet['distance_ratio']
    config_test = config['testing']
    image_name = config_data['image_name']
    label_name = config_data['label_name']
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
    validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    volume_num=len(os.listdir(os.path.join(config_data.get('data_root'), 'valid')))

    # 3. creat model
    coarse_net_type = config_coarse_pnet['net_type']
    coarse_net_class = NetFactory.create(coarse_net_type)
    middle_net_type = config_middle_pnet['net_type']
    middle_net_class = NetFactory.create(middle_net_type)
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
    Middle_Pnet = middle_net_class(
                    inc=config_middle_pnet.get('input_channel', 1),
                    patch_size = patch_size,
                    base_chns= config_middle_pnet.get('base_feature_number', 16),
                    norm='in',
                    depth=config_middle_pnet.get('depth', False),
                    dilation=config_middle_pnet.get('dilation', 1),
                    n_classes = config_middle_pnet['class_num'],
                    droprate=config_middle_pnet.get('drop_rate', 0.2),
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
    Coarse_Pnet = torch.nn.DataParallel(Coarse_Pnet, device_ids=[0]).half().cuda()
    Middle_Pnet = torch.nn.DataParallel(Middle_Pnet, device_ids=[0]).half().cuda()
    Fine_Pnet = torch.nn.DataParallel(Fine_Pnet, device_ids=[0]).half().cuda()
    if config_test['load_weight']:
        coarse_pnet_weight = torch.load(config_test['coarse_pnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # position net
        Coarse_Pnet.load_state_dict(coarse_pnet_weight)
        middle_pnet_weight = torch.load(config_test['middle_pnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # position net
        Middle_Pnet.load_state_dict(middle_pnet_weight)
        # fine_pnet_weight = torch.load(config_test['fine_pnet_load_path'],
        #                          map_location=lambda storage, loc: storage)  # position net
        # Fine_Pnet.load_state_dict(fine_pnet_weight)
    #show_param(Coarse_Pnet)
    Coarse_RD = Relative_distance_style(Coarse_Pnet,out_mode='fc_position', feature_refine=False, McLraft_refine=False, \
                style_refine=False, distance_mode='tanh', \
                center_patch_size=[2,8,8], distance_ratio=coarse_dis_ratio)
    Middle_RD = Relative_distance_style(Middle_Pnet,out_mode='fc_position',feature_refine=False,distance_mode='tanh', \
                center_patch_size=[2,8,8], distance_ratio=middle_dis_ratio)
    Fine_RD = Relative_distance_style(Fine_Pnet,out_mode='fc_position',distance_mode='tanh', \
                center_patch_size=[2,8,8], distance_ratio=fine_dis_ratio)

    # 4, start to detect
    mean_iou_lis = []
    mean_error_lis = []
    error_dis = []
    for ognb in [1]:
        label_wanted = ognb
        print('lw', label_wanted)
        show = False
        iou_sum = 0
        error_sum = np.zeros([2, 3])
        error_dis.append([])
        iter_patch_num = 15
        iter_move_num = [1,0]
        cor = {}
        with torch.no_grad():
            Coarse_Pnet.eval(), Fine_Pnet.eval()
            for ii_batch, sample_batch in enumerate(validLoader):
                spacing = sample_batch['spacing_0'].cpu().data.numpy().squeeze()
                label_0 = pad(load_volume_as_array(sample_batch['image_path_0'][0].replace(image_name, label_name)), max_scale)
                ss = label_0.shape
                label_1 = pad(load_volume_as_array(sample_batch['image_path_1'][0].replace(image_name, label_name)),max_scale)
                print(sample_batch['image_path_0'])
                sample_batch['image_0'] = pad(sample_batch['image_0'].cpu().data.numpy().squeeze(), max_scale)
                sample_batch['image_1'] = pad(sample_batch['image_1'].cpu().data.numpy().squeeze(), max_scale)
                real_corner_cor = extract_certain_organ_cor(label_0, label_wanted=label_wanted,extreme_point_num=2)
                real_extreme_cor = extract_certain_organ_cor(label_0, label_wanted=label_wanted,extreme_point_num=6)
                support_extreme_cor = extract_certain_organ_cor(label_1, label_wanted=label_wanted, extreme_point_num=6)
                predic_extreme_cor = np.zeros([support_extreme_cor.shape[0],3])
                support_batch = []
                for i in range(support_extreme_cor.shape[0]):
                    '''
                    分别裁减几个support极端点所在patch，预测其坐标
                    '''
                    support_cor = support_extreme_cor[i]
                    support_batch.append(sample_batch['image_1'][support_cor[0] - patch_size[0] // 2:support_cor[0] + patch_size[0] // 2,
                                support_cor[1] - patch_size[1] // 2:support_cor[1] + patch_size[1] // 2,
                                support_cor[2] - patch_size[2] // 2:support_cor[2] + patch_size[2] // 2][np.newaxis])
                support_batch = np.asarray(support_batch)
                center_support_patch = crop_patch_around_center(support_batch, r=[8,8,8])
                center_support_fg_mask = center_support_patch>0.1
                Coarse_RD.cal_support_position(support_batch)
                Middle_RD.cal_support_position(support_batch)
                # Fine_RD.cal_support_position(support_batch)

                cur_position,predic_position,query_batch = [],[],[]
                for ii in range(iter_patch_num):
                    '''
                    多次随机裁减预测距离，最终取平均
                    '''
                    sample = random_crop(sample_batch)
                    sample = to_tensor(sample)
                    random_position = np.int16(sample['random_position_0']).squeeze()
                    random_cor = np.around(random_position/spacing).astype(np.int16)
                    cur_position.append(random_position)
                    query_batch.append(sample_batch['image_0'][
                                    random_cor[0] - patch_size[0] // 2:random_cor[0] + patch_size[0] // 2,
                                    random_cor[1] - patch_size[1] // 2:random_cor[1] + patch_size[1] // 2,
                                    random_cor[2] - patch_size[2] // 2:random_cor[2] + patch_size[2] // 2][np.newaxis])
                
                query_batch = np.asarray(query_batch) #[iter_patch_num,1,d,w,h]
                relative_position = Coarse_RD.cal_RD(query_patch=query_batch, mean=True, coarse=True)['relative_position']
                cur_position = np.mean(np.asarray(cur_position), axis=0) + relative_position # [6&2, 3]
                
                for move_step in range(iter_move_num[0]):
                    ''' 多走几次'''
                    cur_position[:,0] = np.minimum(np.maximum(cur_position[:,0], spacing[0]*patch_size[0]-1), spacing[0]*(ss[0]-patch_size[0])+1)
                    cur_position[:,1] = np.minimum(np.maximum(cur_position[:,1], spacing[1]*patch_size[1]-1), spacing[1]*(ss[1]-patch_size[1])+1)
                    cur_position[:,2] = np.minimum(np.maximum(cur_position[:,2], spacing[2]*patch_size[2]-1), spacing[2]*(ss[2]-patch_size[2])+1)
                    middle_query_batch = []
                    
                    cur_cor = np.around(cur_position/spacing).astype(np.int16) #像素坐标
                    for iii in range(cur_position.shape[0]):
                        middle_query_batch.append(sample_batch['image_0'][
                                    cur_cor[iii,0] - patch_size[0] // 2:cur_cor[iii,0] + patch_size[0] // 2,
                                    cur_cor[iii,1] - patch_size[1] // 2:cur_cor[iii,1] + patch_size[1] // 2,
                                    cur_cor[iii,2] - patch_size[2] // 2:cur_cor[iii,2] + patch_size[2] // 2][np.newaxis])
                    middle_query_batch = np.asarray(middle_query_batch)
                    relative_position = Middle_RD.cal_RD(query_patch=middle_query_batch, mean=False)['relative_position']
                    cur_position = np.float16(cur_position)+ relative_position # [6,3]
                for move_step in range(iter_move_num[1]):
                    ''' 多走几次'''
                    cur_position[:,0] = np.minimum(np.maximum(cur_position[:,0], spacing[0]*patch_size[0]-1), spacing[0]*(ss[0]-patch_size[0])+1)
                    cur_position[:,1] = np.minimum(np.maximum(cur_position[:,1], spacing[1]*patch_size[1]-1), spacing[1]*(ss[1]-patch_size[1])+1)
                    cur_position[:,2] = np.minimum(np.maximum(cur_position[:,2], spacing[2]*patch_size[2]-1), spacing[2]*(ss[2]-patch_size[2])+1)
                    fine_query_batch = []
                    cur_cor = np.around(cur_position/spacing).astype(np.int16) #像素坐标
                    
                    for iii in range(cur_position.shape[0]):
                        fine_query_batch.append(sample_batch['image_0'][
                                    cur_cor[iii,0] - patch_size[0] // 2:cur_cor[iii,0] + patch_size[0] // 2,
                                    cur_cor[iii,1] - patch_size[1] // 2:cur_cor[iii,1] + patch_size[1] // 2,
                                    cur_cor[iii,2] - patch_size[2] // 2:cur_cor[iii,2] + patch_size[2] // 2][np.newaxis])
                    fine_query_batch = np.asarray(fine_query_batch)
                    relative_position = Fine_RD.cal_RD(fine_query_batch)
                    cur_position = np.float16(cur_position)+ relative_position # [6,3]
                predic_extreme_cor = cur_position.copy()/spacing
                if predic_extreme_cor.shape[0]==6:
                    predic_corner_cor = transfer_extremepoint_to_cornerpoint(predic_extreme_cor)
                predic_corner_cor = np.asarray([np.min(predic_extreme_cor,axis=0),np.max(predic_extreme_cor, axis=0)])
                pred_iou = iou(real_corner_cor,  predic_corner_cor)
                pred_error = spacing*(real_corner_cor-predic_corner_cor) # 2*3
                print('predic iou:',pred_iou,'\n', 'predic error:',pred_error)
                #print(predic_extreme_cor,'\n',real_extreme_cor)
                iou_sum+=pred_iou
                error_sum+=np.abs(pred_error)
                error_dis[-1].append(pred_error)
                if show :
                    in_predic_extreme_cor = np.int16(np.around(predic_extreme_cor))
                    for i in range(support_extreme_cor.shape[0]):
                        savename = './{0:}.png'.format(str(ii_batch)+str(i))
                        save_detection(sample_batch, support_extreme_cor[i], 
                            query_point_position=real_extreme_cor[i],predicted_point_position=in_predic_extreme_cor[i], fname=savename)
            # error_dis[-1]=np.asarray(error_dis[-1])
            # fig,(ax0,ax1,ax2, ax3,ax4,ax5) = plt.subplots(nrows=6,figsize=(9,6)) 
            # ax0.hist(error_dis[-1][:,0,0],10,histtype='bar',facecolor='yellowgreen',alpha=0.75)
            # ax1.hist(error_dis[-1][:,0,1],10,histtype='bar',facecolor='pink',alpha=0.75)
            # ax2.hist(error_dis[-1][:,0,2],10,histtype='bar',facecolor='red',alpha=0.75)
            # ax3.hist(error_dis[-1][:,1,0],10,histtype='bar',facecolor='yellowgreen',alpha=0.75)
            # ax4.hist(error_dis[-1][:,1,1],10,histtype='bar',facecolor='pink',alpha=0.75)
            # ax5.hist(error_dis[-1][:,1,2],10,histtype='bar',facecolor='red',alpha=0.75)       
            # fig.subplots_adjust(hspace=0.4)  
            # plt.show()
            mean_iou = np.around(iou_sum/volume_num, decimals=3)
            mean_error = np.around(np.mean(error_sum/volume_num), decimals=2)
            mean_error_each = np.around(error_sum/volume_num, decimals=2)
            mean_iou_lis.append(mean_iou)
            mean_error_lis.append(mean_error)
            print('mean iou:',mean_iou, 'mean error:',mean_error, 'mean error each:',mean_error_each)
    print(np.mean(np.array(mean_iou_lis)), np.mean(np.array(mean_error_lis)))
    



if __name__ == '__main__':
    config_file = str('../config/test_position_full_size_trible_stage_style_panc.txt')
    assert (os.path.isfile(config_file))
    test(config_file) 
