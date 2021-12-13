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

    test_patch_size = config_data['test_patch_size']
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed)  # 给定seed value,决定了后面的伪随机序列

    # 2, load data
    print('2.Load data')
    validData = PositionDoublePatientDataloader(config=config_data,
                                                split='valid',
                                                transform=None)
    validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 4, start to detect
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
        query_extreme_cor = extract_certain_organ_cor(label_1,label_wanted=label_wanted, extreme_point_num=6)
        real_extreme_cor = extract_certain_organ_cor(label_0, label_wanted=label_wanted,extreme_point_num=2)
        predic_extreme_cor = np.zeros([query_extreme_cor.shape[0],3])
        t1 = time.time()
        for i in range(query_extreme_cor.shape[0]):
            '''
            分别预测最小点与最大点
            '''
            query_position = query_extreme_cor[i]
            target_batch = sample_batch['image_1'][query_position[0] - test_patch_size[0] // 2:query_position[0] + test_patch_size[0] // 2,
                        query_position[1] - test_patch_size[1] // 2:query_position[1] + test_patch_size[1] // 2,
                        query_position[2] - test_patch_size[2] // 2:query_position[2] + test_patch_size[2] // 2]
            
            
            predicted_position=get_position_sift(target_patch=target_batch, image=sample_batch['image_0'],stepSize=[4,4,4])
            
            predic_extreme_cor[i]=np.asarray(predicted_position)
            predicted_position = np.round(predicted_position)
            print( 'query position', query_position-[24, 24, 24],'predicted position', predicted_position-[24, 24, 24])

            if show :
                plt.figure(figsize=(150,150))
                plt.subplot(121)
                plt.imshow(sample_batch['image_1'][query_position[0]], cmap='gray')
                plt.plot(query_position[2], query_position[1], '*', c='r')
                plt.title('target position')
                plt.subplot(122)
                plt.imshow(sample_batch['image_0'][predicted_position[0]], cmap='gray')
                plt.plot(predicted_position[2], predicted_position[1], '*', c='r')
                plt.title('detected position')
                plt.show()
        t2 = time.time()
        print('t1:{} t2:{}'.format(t2-t0, (t2-t1)))
        real_predic_extreme_cor = np.asarray([np.min(predic_extreme_cor,axis=0),np.max(predic_extreme_cor, axis=0)])
        print('extreme_cor',real_extreme_cor.reshape(1,-1),'predic_cor',real_predic_extreme_cor.reshape(1,-1))
        pred_iou = iou(real_extreme_cor,  real_predic_extreme_cor)
        pred_error = np.abs(real_extreme_cor-real_predic_extreme_cor)
        print('predic iou:',pred_iou, 'predic error:',pred_error.reshape(1,-1), 'error', np.mean(pred_error.reshape(1,-1)))
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
