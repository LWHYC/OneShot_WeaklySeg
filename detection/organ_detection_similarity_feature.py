#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import sys
sys.path.append('/home/disk/LWH/Code/Relative_Distance_Regression_v2/')
from dataloaders.Position_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
import matplotlib.pyplot as plt
from data_process.data_process_func import *
from util.visualization.show_param import show_param
from prefetch_generator import BackgroundGenerator
from detection.detection_functions import *
from networks.NetFactory import NetFactory
import time
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def test(config_file, label_wanted):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data = config['data']  # 包含数据的各种信息,如data_shape,batch_size等
    config_test = config['testing']

    image_name = config_data['image_name']
    label_name = config_data['label_name']
    
    config_net   = config['network']    # 网络参数,如net_name,base_feature_name,class_num等
    config_test = config['testing']

    net_type    = config_net['net_type']
    class_num   = config_net['class_num']

    test_patch_size = config_data['test_patch_size']
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed)  # 给定seed value,决定了后面的伪随机序列

    # 2, load data
    print('2.Load data')
    validData = PositionDoublePatientDataloader(config=config_data,
                                                split='test',
                                                transform=None)
    validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 3. creat model
    print('3.Creat model')

    net_class = NetFactory.create(net_type)
    net = net_class(inc=config_net.get('input_channel', 1),
                    n_classes = class_num,
                    base_chns= config_net.get('base_feature_number', 16),
                    droprate=config_net.get('drop_rate', 0.2),
                    norm='in',
                    depth=config_net.get('depth', False),
                    dilation=config_net.get('dilation', 1),
                    #separate_direction='axial',
                    )
    
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    # net = torch.nn.DataParallel(net).cuda(1)

    if config_test['load_weight']:
        weight = torch.load(config_test['load_path'], map_location=lambda storage, loc: storage)

        model2_dict = net.state_dict()
        state_dict = {k:v for k,v in weight.items() if k in model2_dict.keys()}
        model2_dict.update(state_dict)
        net.load_state_dict(model2_dict)

    show_param(net) #计算网络总参数量

    # 4, start to detect
    with torch.no_grad():
        print('4.Start to detect')
        show = False
        iou_sum = 0
        error_sum = np.zeros([2, 3])
        t0 = time.time()
        for ii_batch, sample_batch in enumerate(validLoader):
            label_0 = pad(load_volume_as_array(sample_batch['image_path_0'][0].replace(image_name, label_name)), test_patch_size)
            label_1 = pad(load_volume_as_array(sample_batch['image_path_1'][0].replace(image_name, label_name)),test_patch_size)
            print('label wanted', label_wanted, sample_batch['image_path_0'], sample_batch['image_path_1'])

            sample_batch['image_0'] = pad(sample_batch['image_0'].cpu().data.numpy().squeeze(), test_patch_size)
            sample_batch['image_1'] = pad(sample_batch['image_1'].cpu().data.numpy().squeeze(), test_patch_size)
            target_extreme_cor = extract_certain_organ_cor(label_1,label_wanted=label_wanted, extreme_point_num=6)
            real_extreme_cor = extract_certain_organ_cor(label_0, label_wanted=label_wanted,extreme_point_num=2)
            predic_extreme_cor = np.zeros([target_extreme_cor.shape[0],3])
            t1 = time.time()
            for i in range(target_extreme_cor.shape[0]):
                '''
                分别预测最小点与最大点
                '''
                target_position = target_extreme_cor[i]
                target_batch = sample_batch['image_1'][target_position[0] - test_patch_size[0] // 2:target_position[0] + test_patch_size[0] // 2,
                            target_position[1] - test_patch_size[1] // 2:target_position[1] + test_patch_size[1] // 2,
                            target_position[2] - test_patch_size[2] // 2:target_position[2] + test_patch_size[2] // 2]

                predicted_position=get_position_feature(target_patch=target_batch, image=sample_batch['image_0'], net=net, stepSize=[6,6,6])

                predic_extreme_cor[i]=np.asarray(predicted_position)
                predicted_position = np.round(predicted_position)
                print( 'target position', target_position-[24,24,24],'predicted position', predicted_position-[24,24,24])

                if show :
                    plt.figure(figsize=(150,150))
                    plt.subplot(121)
                    plt.imshow(sample_batch['image_0'][target_position[0]], cmap='gray')
                    plt.plot(target_position[2], target_position[1], '*', c='r')
                    plt.title('target position')
                    plt.subplot(122)
                    plt.imshow(sample_batch['image_1'][predicted_position[0]], cmap='gray')
                    plt.plot(predicted_position[2], predicted_position[1], '*', c='r')
                    plt.title('detected position')
                    plt.show()
            t2 = time.time()
            print(t2-t0, t2-t1)
            real_predic_extreme_cor = np.asarray([np.min(predic_extreme_cor,axis=0),np.max(predic_extreme_cor, axis=0)])
            print(real_predic_extreme_cor)
            pred_iou = iou(real_extreme_cor,  real_predic_extreme_cor)
            pred_error = np.abs(real_extreme_cor-real_predic_extreme_cor)
            print('predic iou:',pred_iou, 'predic error:',pred_error)
            iou_sum+=pred_iou
            error_sum+=pred_error
        print(label_wanted,'mean iou:', iou_sum/len(os.listdir(os.path.join(config_data.get('data_root'), 'valid'))))
        print(label_wanted,'mean error:', np.asarray([1.5,1.5,1.5])*np.sum(error_sum, axis=0)/len(
            os.listdir(os.path.join(config_data.get('data_root'), 'valid'))))
        print(label_wanted,'mean error each:', np.asarray([3, 3, 3]) * error_sum/ len(
            os.listdir(os.path.join(config_data.get('data_root'), 'valid'))))
        print('mean err', np.mean(np.asarray([3, 3, 3]) * error_sum/ len(os.listdir(os.path.join(config_data.get('data_root'), 'valid')))))




if __name__ == '__main__':
    config_file = str('/home/disk/LWH/Code/Relative_Distance_Regression_v2/config/test_position_ae.txt')
    assert (os.path.isfile(config_file))
    label_wanted = 1 #int(input('label wanted :'))
    test(config_file, label_wanted=label_wanted)
