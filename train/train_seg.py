#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import argparse
sys.path.append(os.path.abspath(__file__))  #返回当前.py文件的绝对路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from dataloaders.Torchio_rm_noise_dataloader import *
from torch.utils.data import DataLoader, dataloader
import matplotlib.pyplot as plt
from util.train_test_func import *
from util.parse_config import parse_config
from networks.NetFactory import  NetFactory
from losses.loss_function import *
from test.test import test_single_case, cal_dice

def train(config_file):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data  = config['data']    # 包含数据的各种信息,如data_shape,batch_size等
    config_net   = config['network']    # 网络参数,如net_name,base_feature_name,class_num等
    config_train = config['training']

    train_patch_size = config_data['train_patch_size']
    test_patch_size = config_data['test_patch_size']
    stride = config_data['stride']
    device_ids = config_data['device_ids']
    batch_size = config_data.get('batch_size', 4)
    concat_coarseg = config_data['concat_coarseg']
    concat_distance = config_data['concat_distance']
    net_type    = config_net['net_type']
    class_num   = config_net['class_num']


    lr = config_train.get('learning_rate', 1e-3)
    best_dice = config_train.get('best_dice', 0.5)
    random_seed = config_train.get('random_seed', 1)
    num_worker = config_train.get('num_worker', 0)
    load_memory = config_train.get('load_memory', False)

    random.seed(random_seed)     # 给定seed value,决定了后面的伪随机序列
    cudnn.benchmark = True
    cudnn.deterministic = True
    best_dice_iter = 0
    cur_dice = 0
    noise_thresh = np.array([0.95,0.99])
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # 2, load data
    print('2.Load data')
    trainData = TorchioDataloader(config=config_data,
                                   class_num=16,
                                   split='train',
                                   transform=transforms.Compose([
                                       ExtractCertainClass(class_wanted=[6]),
                                       CropBound(pad=[2,24,24], mode='coarseg'),
                                       RandomCrop(train_patch_size),
                                       ToTensor(concat_coarseg=concat_coarseg, concat_distance=concat_distance),
                                   ]),
                                   load_memory=load_memory)
    validData = TorchioDataloader(config=config_data,
                                   class_num=16,
                                   split='valid',
                                   transform=transforms.Compose([
                                    ExtractCertainClass(class_wanted=[6]),
                                    CropBound(pad=[2,8,8], mode='label'),
                                    ToTensor(concat_coarseg=concat_coarseg, concat_distance=concat_distance)
                                   ]),
                                   random_sample=False,
                                   load_memory=load_memory)
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    validLoader = DataLoader(validData, batch_size=1, shuffle=False, num_workers=1)
    cropbound=CropBound(pad=[2,24,24], mode='coarseg')
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
                    )
    net = torch.nn.DataParallel(net).cuda()
    if config_train['load_pretrained_model']:
        pretrained_dict = torch.load(config_train['pretrained_model_path'], map_location=lambda storage, loc: storage)
        model_dict=net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    dice_eval = TestDiceLoss(n_class=class_num)
    loss_func = CrossEntropy_and_Dice_Loss(n_class=class_num,alpha=1)

    Adamoptimizer = optim.Adam(net.parameters(), lr=lr, weight_decay= config_train.get('decay', 1e-7))
    Adamscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Adamoptimizer, mode='max', factor=0.9, patience=1,threshold=0.001)
    # 4, start to train
    print('4.Start to train')
    start_it  = config_train.get('start_iteration', 0)
    dice_save = {}
    lr_retain_epoch = 0
    for epoch in range(start_it, config_train['maximal_epoch']):  
        print('#######epoch:', epoch)
        optimizer = Adamoptimizer
        net.train()
        for i_batch, sample_batch in enumerate(trainLoader):
            img_batch, label_batch, coarseg_batch = sample_batch['image'], sample_batch['label'], sample_batch['coarseg']
            img_batch, label_batch, coarseg_batch = img_batch.cuda(), label_batch.cuda(), coarseg_batch.cuda()
            predic = net(img_batch)['prob']
            train_loss = loss_func(predic, coarseg_batch)
            optimizer.zero_grad()  
            train_loss.backward() 
            optimizer.step() 
            if epoch%config_train['train_step']==0 and i_batch%config_train['print_step']==0:
                train_dice = dice_eval(predic, label_batch)
                train_dice = train_dice.cpu().data.numpy()
                train_loss = train_loss.cpu().data.numpy()
                if i_batch ==0:
                    train_dice_array=train_dice[1::][np.newaxis,:]
                else:
                    train_dice_array = np.append(train_dice_array, train_dice[1::][np.newaxis,:], axis=0)
                print('train batch:',i_batch,' train dice:', train_dice, 'train_loss:', train_loss)

        if epoch>=2:
            with torch.no_grad():
                net.eval()
                for key in trainLoader.dataset.image_dic.keys():
                    sample = trainLoader.dataset.get_specific_data(key)
                    coarseg = sample['coarseg'].numpy()
                    sample = cropbound(sample)
                    crop_label = sample['label'].numpy()
                    crop_img_batch, crop_coarseg,minpoint,maxpoint,ori_shape = sample['image'].numpy(),sample['coarseg'].numpy(),sample['minpoint'],sample['maxpoint'],sample['shape']
                    if len(crop_img_batch.shape)!=4:
                        crop_img_batch = crop_img_batch[np.newaxis,:]
                    predic_all = test_single_case(net, crop_img_batch, concat_coarseg=concat_coarseg,stride=stride,
                                                       num_classes=class_num, patch_size=test_patch_size)
                    crop_predic_prob = predic_all['prob']
                    crop_update_coarseg = trainLoader.dataset.update_corrupted_label(crop_predic_prob, crop_label, crop_coarseg, noise_thresh)
                    coarseg[:, minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]] = crop_update_coarseg
                    trainLoader.dataset.image_dic[key]['coarseg'] = torch.from_numpy(coarseg)
                noise_thresh*=0.98
                noise_thresh[0]=max(noise_thresh[0], 0.85)
                noise_thresh[1::]=np.clip(noise_thresh[1::], 0.9, 1)
                print('update finished')
                
        if  epoch % config_train['test_step']==0:
            with torch.no_grad():
                net.eval()
                for ii_batch, sample_batch in enumerate(validLoader):
                    valid_img_batch, valid_label_batch = sample_batch['image'].cpu().data.numpy().squeeze(), sample_batch['label'].cpu().data.numpy().squeeze().astype(np.uint16)
                    if len(valid_img_batch.shape)!=4:
                        valid_img_batch = valid_img_batch[np.newaxis,:]
                    if 'distance' in sample_batch.keys():
                        distance_batch=sample_batch['distance'].cpu().data.numpy().squeeze()[np.newaxis,:]
                        valid_img_batch = np.concatenate((valid_img_batch, distance_batch), 0)
                    if concat_coarseg:
                        coarseg = sample_batch['coarseg'].cpu().data.numpy().squeeze()
                    else:
                        coarseg = False
                    predic_all = test_single_case(None, net, valid_img_batch, concat_coarseg=concat_coarseg,coarseg=coarseg,stride=stride,
                                                       num_classes=class_num, patch_size=test_patch_size)
                    predic_label = predic_all['label']
                    valid_dice = cal_dice(predic_label, valid_label_batch, class_num)
                    if ii_batch ==0:
                        valid_dice_array = valid_dice[np.newaxis,:]
                    else:
                        valid_dice_array = np.append(valid_dice_array, valid_dice[np.newaxis,:], axis=0)
                    print('valid batch:',ii_batch,' valid dice:', valid_dice)

            epoch_dice = {'valid_dice':valid_dice_array.mean(axis=0), 'train_dice':train_dice_array.mean(axis=0)}
            t = time.strftime('%X %x %Z')
            print(t, 'epoch', epoch, 'dice:', epoch_dice)
            train_dice_mean = epoch_dice['train_dice'].mean(axis=0)
            epoch_dice['train_dice_mean'] = train_dice_mean
            valid_dice_classes = epoch_dice['valid_dice']
            valid_dice_mean = valid_dice_classes.mean(axis=0)
            epoch_dice['valid_dice_mean'] = valid_dice_mean
            dice_save[epoch] = epoch_dice

            '當前批次模型儲存'
            if os.path.exists(config_train['model_save_prefix'] + "_cur_{0:}.pkl".format(cur_dice)):
                os.remove(config_train['model_save_prefix'] + "_cur_{0:}.pkl".format(cur_dice))
            cur_dice = epoch_dice['valid_dice_mean']
            torch.save(net.state_dict(), config_train['model_save_prefix'] + "_cur_{0:}.pkl".format(cur_dice))

            '最优模型儲存'
            if epoch_dice['valid_dice_mean'] > best_dice:
                if best_dice_iter>0:
                    if os.path.exists(config_train['model_save_prefix'] + "_{0:}.pkl".format(best_dice)):
                        os.remove(config_train['model_save_prefix'] + "_{0:}.pkl".format(best_dice))
                best_dice = epoch_dice['valid_dice_mean']
                torch.save(net.state_dict(), config_train['model_save_prefix'] + "_{0:}.pkl".format(best_dice))
                best_dice_iter+=1 
            Adamscheduler.step(epoch_dice['valid_dice_mean'])


if __name__ == '__main__':
    config_file = 'config/train_tcia.txt' # you could change it to config/train_structseg.txt
    assert(os.path.isfile(config_file))
    train(config_file)
