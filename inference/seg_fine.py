# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import sys
import os
import argparse

sys.path.append("..")
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders.Torchio_dataloader import *
from util.train_test_func import *
from torch.utils.data import DataLoader
from torchvision import transforms
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
import pandas as pd
from test_single.test import test_single_case
from medpy import metric
import numpy as np
from util.evaluation_index import assd, dc,recall, precision
from data_process.data_process_func import save_array_as_nifty_volume, one_hot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='', help='Config file path')
    args = parser.parse_args()
    assert(os.path.isfile(args.config_path))
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(args.config_path)
    config_data = config['data']  # config of data,e.g. data_shape,batch_size.
    config_net = config['network']  # config of net, e.g. net_name,base_feature_name,class_num.
    config_test = config['testing']
    random.seed(config_test.get('random_seed', 1))
    net_type = config_net['net_type']
    class_num = config_net['class_num']
    stride = config_data['stride']
    test_patch_size = config_data['test_patch_size']
    save_seg = config_data['save_seg']
    concat_coarseg = config_data['concat_coarseg']
    remo_noise = True
    cal_dice = True
    cal_assd = False
    cal_hd95 = False
    cal_recall = True
    cal_precision = True
    save_metric = False

    # 2, load data
    print('2.Load data')
    Datamode = ['train']
    patient_number = len(read_file_list(config_data.get('test_image_list')))
    dice_array = np.zeros([patient_number, class_num])
    assd_array = np.zeros([patient_number, class_num])
    hd95_array = np.zeros([patient_number, class_num])
    recall_array = np.zeros([patient_number, class_num])
    precision_array = np.zeros([patient_number, class_num])
    Data = TorchioDataloader(config=config_data,
                                    class_num=class_num,
                                    wanted_class=config_data.get('class_wanted'),
                                    transform=transforms.Compose([
                                CropBound(pad=[48,48,48], mode='coarseg'),
                                ToTensor(concat_coarseg=concat_coarseg)
                                ]),
                                load_memory = False,
                                random_sample=False,
                                image_list=config_data.get('test_image_list'),
                                label_list=config_data.get('test_label_list'),
                                coarseg_list=config_data.get('test_coarseg_list'))
    Dataloader = DataLoader(Data, batch_size=1, shuffle=False, num_workers=1)

    # 3. creat model
    print('3.Creat model')
    net_class = NetFactory.create(net_type)
    net = net_class(inc=config_net.get('input_channel', 1),
                    n_classes = class_num,
                    base_chns= config_net.get('base_feature_number', 16),
                    droprate=config_net.get('drop_rate', 0.2),
                    norm='in',
                    depth=False,
                    dilation=1
                    )
    net = torch.nn.DataParallel(net).cuda()
    if config_test['load_pretrained_model']:
        checkpoint = torch.load(config_test['pretrained_model_path'])
        net.load_state_dict(checkpoint['state_dict'])
    # 4, start to seg
    print('''start to seg ''')
    with torch.no_grad():
        net.eval()
        for ii_batch, sample_batch in enumerate(Dataloader):
            img_batch, label_batch, image_path = sample_batch['image'].cpu().data.numpy().squeeze(), sample_batch['label'].cpu().data.numpy().squeeze(),sample_batch['image_path']
            minpoint = sample_batch['minpoint']
            maxpoint = sample_batch['maxpoint']
            orishape = sample_batch['shape']
            prediction = np.zeros(orishape)
            if 'coarseg' in sample_batch.keys():
                crop_coarseg = sample_batch['coarseg'].cpu().data.numpy().squeeze()
            if concat_coarseg:
                valid_img_batch = np.concatenate((valid_img_batch, crop_coarseg), dim=0)
            img_batch = img_batch[np.newaxis, :]
            predic_all = test_single_case(net, img_batch ,stride=stride, patch_size=test_patch_size)
            crop_prediction = predic_all['label']
            if remo_noise:
                nprediction = np.zeros_like(crop_prediction, dtype=np.int16)
                oh_crop_prediction = one_hot(crop_prediction, class_num)
                oh_crop_coarseg = one_hot(crop_coarseg, class_num)
                for i in range(1, class_num):
                    oh_minpoint, oh_maxpoint= get_bound_coordinate(oh_crop_coarseg[i], pad=[4,8,8])
                    cur_oh_crop_prediction = oh_crop_prediction[i, oh_minpoint[0]:oh_maxpoint[0], oh_minpoint[1]:oh_maxpoint[1], oh_minpoint[2]:oh_maxpoint[2]]
                    nprediction[oh_minpoint[0]:oh_maxpoint[0], oh_minpoint[1]:oh_maxpoint[1], oh_minpoint[2]:oh_maxpoint[2]] += np.int16(i*cur_oh_crop_prediction)
                crop_prediction = nprediction
            prediction[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]] = crop_prediction
            prediction *= 1
            nprediction = recover_certain_class(prediction, wanted_class=config_data.get('class_wanted'))
            save_array_as_nifty_volume(nprediction, image_path[0].replace(image_path[0].split('/')[-1], config_data['save_seg_name'])) # change the save 
            print(sample_batch['image_path'])
            print(prediction.shape)
            oh_label = one_hot(label_batch, class_num)
            oh_predic = one_hot(crop_prediction, class_num)
            if cal_dice:
                Dice = np.zeros(class_num)
                for i in range(class_num):
                    Dice[i] = dc(oh_predic[i], oh_label[i])
                dice_array[ii_batch] = Dice
                print('patient order', ii_batch, ' dice:', Dice)
            if cal_assd:
                Assd = np.zeros(class_num)
                for i in range(class_num):
                    Assd[i] = assd(oh_predic[i], oh_label[i], 1)
                assd_array[ii_batch] = Assd
                print('patient order', ii_batch, ' dice:', Assd)
            if cal_hd95:
                Hd95 = np.zeros(class_num)
                for i in range(class_num):
                    try:
                        Hd95[i] = metric.binary.hd95(oh_predic[i], oh_label[i], voxelspacing=[3,1,1])
                    except:
                        Hd95[i] = 23
                hd95_array[ii_batch] = Hd95
                print('patient order', ii_batch, ' Hd95:', Hd95)
            if cal_recall:
                Recall = np.zeros(class_num)
                for i in range(class_num):
                    Recall[i] = recall(oh_predic[i], oh_label[i])
                recall_array[ii_batch] = Recall
                print('patient order', ii_batch, ' recall:', Recall)
            if cal_precision:
                Precision = np.zeros(class_num)
                for i in range(class_num):
                    Precision[i] = precision(oh_predic[i], oh_label[i])
                precision_array[ii_batch] = Precision
                print('patient order', ii_batch, ' precision:', Precision)

        if cal_dice:
            dice_array[:, 0] = np.mean(dice_array[:, 1::], 1)
            dice_mean = np.around(100*np.mean(dice_array, 0), decimals=1)
            dice_std = np.around(100*np.std(dice_array, 0),decimals=1)
            print('mode: mean dice:{0:}, std of dice:{1:}'.format( dice_mean, dice_std))
        if cal_assd:
            assd_array[:, 0] = np.mean(assd_array[:, 1::], 1)
            assd_mean = np.around(np.mean(assd_array, 0),decimals=1)
            assd_std = np.around(np.std(assd_array, 0),decimals=1)
            print('mode: mean assd:{0:}, std of assd:{1:}'.format( assd_mean, assd_std))
        if cal_hd95:
            hd95_array[:, 0] = np.mean(hd95_array[:, 1::], 1)
            hd95_mean = np.around(np.mean(hd95_array, 0), decimals=2)
            hd95_std = np.around(np.std(hd95_array, 0),decimals=2)
            print('mode: mean hd95:{0:}, std of hd95:{1:}'.format( hd95_mean, hd95_std))
        if cal_recall:
            recall_array[:, 0] = np.mean(recall_array[:, 1::], 1)
            recall_mean = np.mean(recall_array, 0)
            recall_std = np.std(recall_array, 0)
            print('mode: mean recall:{0:}, std of recall:{1:}'.format( recall_mean, recall_std))
        if cal_precision:
            precision_array[:, 0] = np.mean(precision_array[:, 1::], 1)
            precision_mean = np.mean(precision_array, 0)
            precision_std = np.std(precision_array, 0)
            print('mode: mean precision:{0:}, std of precision:{1:}'.format( precision_mean, precision_std))



