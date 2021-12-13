# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import sys
import os
import argparse

from scipy.ndimage.measurements import label
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
from test.test import test_single_case
import numpy as np
from util.evaluation_index import dc
from data_process.data_process_func import save_array_as_nifty_volume, get_largest_component
from scipy.ndimage import morphology
from util.assd_evaluation import one_hot


if __name__ == '__main__':
    config_file = str('config/test_tcia.txt')
    assert (os.path.isfile(config_file))
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data = config['data']  # config of data,e.g. data_shape,batch_size.
    config_net = config['network']  # config of net, e.g. net_name,base_feature_name,class_num.
    config_test = config['testing']
    random.seed(config_test.get('random_seed', 1))
    net_type = config_net['net_type']
    class_num = config_net['class_num']
    stride = config_data['stride']
    test_patch_size = config_data['test_patch_size']
    save_seg = config_data['save_seg']
    seg_name = config_data['seg_name']
    concat_coarseg = config_data['concat_coarseg']
    class_wanted = config_data['class_wanted']
    remo_noise = True
    cal_dice = True

    # 2, load data
    print('2.Load data')
    Datamode = ['train']
    for mode in Datamode:
        patient_number = len(os.listdir(os.path.join(config_data['data_root'], mode)))
        dice_array = np.zeros([patient_number, class_num])
        assd_array = np.zeros([patient_number, class_num])
        hd95_array = np.zeros([patient_number, class_num])
        recall_array = np.zeros([patient_number, class_num])
        precision_array = np.zeros([patient_number, class_num])
        Data = TorchioDataloader(config=config_data,
                                       split=mode,
                                       class_num=16,
                                       transform=transforms.Compose([
                                    ExtractCertainClass(class_wanted=[class_wanted]),
                                    CropBound(pad=[2,8,8], mode='coarseg'),
                                    ToTensor(concat_coarseg=concat_coarseg)
                                   ]),
                                   load_memory = False,
                                  random_sample=False)
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
        net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
        if config_test['load_pretrained_model']:
            weight = torch.load(config_test['pretrained_model_path'], map_location=lambda storage, loc: storage)
            net.load_state_dict(weight)
        # 4, start to seg
        print('''start to seg ''')
        with torch.no_grad():
            net.eval()
            for ii_batch, sample_batch in enumerate(Dataloader):
                img_batch, label_batch, patient_path = sample_batch['image'].cpu().data.numpy().squeeze(), sample_batch['label'].cpu().data.numpy().squeeze(),sample_batch['patient_path']
                minpoint = sample_batch['minpoint']
                maxpoint = sample_batch['maxpoint']
                orishape = sample_batch['shape']
                prediction = np.zeros(orishape)
                if concat_coarseg:
                    coarseg = sample_batch['coarseg'].cpu().data.numpy().squeeze()
                else:
                    coarseg = False
                img_batch = img_batch[np.newaxis, :]
                predic_all = test_single_case(None, net, img_batch, concat_coarseg=concat_coarseg,coarseg=coarseg,stride=stride,
                                                       num_classes=class_num, patch_size=test_patch_size)
                crop_prediction = predic_all['label']
                if remo_noise:
                    nprediction = morphology.binary_erosion(crop_prediction, np.ones([8,8,8])).astype(np.int16)
                    nprediction = get_largest_component(nprediction)
                    nprediction = morphology.binary_dilation(nprediction, np.ones([12,12,12])).astype(np.int16)
                    crop_prediction *= nprediction
                prediction[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]] = crop_prediction
                if save_seg :
                    prediction *= class_wanted
                    save_array_as_nifty_volume(prediction, patient_path[0].replace(config_data['image_name'], config_data['seg_name']))
                print(sample_batch['patient_path'])
                print(prediction.shape)
                one_hot_label = one_hot(label_batch, class_num)
                one_hot_predic = one_hot(crop_prediction, class_num)
                if cal_dice:
                    Dice = np.zeros(class_num)
                    for i in range(class_num):
                        Dice[i] = dc(one_hot_predic[i], one_hot_label[i])
                    dice_array[ii_batch] = Dice
                    print('patient order', ii_batch, ' dice:', Dice)

            if cal_dice:
                dice_array[:, 0] = np.mean(dice_array[:, 1::], 1)
                dice_mean = np.around(100*np.mean(dice_array, 0), decimals=1)
                dice_std = np.around(100*np.std(dice_array, 0),decimals=1)
                print('{0:} mode: mean dice:{1:}, std of dice:{2:}'.format(mode, dice_mean, dice_std))



