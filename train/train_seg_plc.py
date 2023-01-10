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
from tensorboardX import SummaryWriter
from torchvision import transforms
from dataloaders.Torchio_rm_noise_dataloader import *
from torch.utils.data import DataLoader, dataloader
from util.train_test_func import *
from util.parse_config import parse_config
from networks.NetFactory import  NetFactory
from losses.loss_function import *
from test_single.test import test_single_case, cal_dice
from util.save_model import Save_checkpoint

def train(config_file):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data  = config['data']   
    config_net   = config['network']   
    config_train = config['training']

    os.environ['CUDA_VISIBLE_DEVICES']= config_train.get('device_ids')
    train_patch_size = config_data['train_patch_size']
    test_patch_size = config_data['test_patch_size']
    stride = config_data['stride']
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

    random.seed(random_seed)    
    cudnn.benchmark = True
    cudnn.deterministic = True
    best_dice_iter = 0
    cur_dice = 0
    noise_thresh = np.array([0.99,0.99,0.99,0.99])
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    save_model = Save_checkpoint()

    # 2. creat model
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
        if os.path.isfile(config_train['pretrained_model_path']):
            print("=> loading checkpoint '{}'".format(config_train['pretrained_model_path']))
            checkpoint = torch.load(config_train['pretrained_model_path'])
            best_dice = checkpoint['best_dice']
            net.load_state_dict(checkpoint['state_dict'])
            Adamoptimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' ".format(config_train['pretrained_model_path']))
        else:
            raise(ValueError("=> no checkpoint found at '{}'".format(config_train['pretrained_model_path'])))
        
                
    dice_eval = TestDiceLoss(n_class=class_num)
    loss_func = CrossEntropy_and_Dice_Loss(n_class=class_num, lamda=1)

    Adamoptimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay= config_train.get('decay', 1e-7))
    Adamscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Adamoptimizer, mode='max', factor=0.9, patience=1,threshold=0.001)

    # 3, load data
    print('2.Load data')
    trainData = TorchioDataloader(config=config_data,
                                   class_num=class_num,
                                   wanted_class = config_data.get('class_wanted'),
                                   transform=transforms.Compose([
                                       RandomCrop(train_patch_size, fg_focus_prob=0.5),
                                       ToTensor(concat_coarseg=concat_coarseg, concat_distance=concat_distance),
                                   ]),
                                   load_memory=load_memory, image_list=config_data.get('train_image_list'),
                                    label_list=config_data.get('train_label_list'),coarseg_list=config_data.get('train_coarseg_list'))
    validData = TorchioDataloader(config=config_data,
                                   class_num=class_num,
                                   wanted_class = config_data.get('class_wanted'),
                                   transform=transforms.Compose([
                                    CropBound(pad=[32,32,32], mode='label'),
                                    ToTensor(concat_coarseg=concat_coarseg, concat_distance=concat_distance)
                                   ]),
                                   random_sample=False,
                                   load_memory=load_memory, image_list=config_data.get('valid_image_list'),
                                   label_list=config_data.get('valid_label_list'))
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    validLoader = DataLoader(validData, batch_size=1, shuffle=False, num_workers=1)
    cropbound=CropBound(pad=[4,24,24], mode='coarseg')
    
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

        if epoch>=0:
            with torch.no_grad():
                net.eval()
                for idx in range(len(trainLoader.dataset.image_list)):
                    sample = trainLoader.dataset.get_specific_data(idx)
                    coarseg = sample['coarseg'].numpy()
                    image_path = sample['image_path']
                    sample = cropbound(sample)
                    crop_label = sample['label'].numpy()
                    crop_img_batch, crop_coarseg,minpoint,maxpoint,ori_shape = sample['image'].numpy(),sample['coarseg'].numpy(),sample['minpoint'],sample['maxpoint'],sample['shape']
                    if len(crop_img_batch.shape)!=4:
                        crop_img_batch = crop_img_batch[np.newaxis,:]
                    predic_all = test_single_case(net, crop_img_batch,stride=stride, patch_size=test_patch_size)
                    crop_predic_prob = predic_all['prob']
                    crop_update_coarseg = trainLoader.dataset.update_corrupted_label(crop_predic_prob, crop_label, crop_coarseg, noise_thresh)
                    coarseg[:, minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]] = crop_update_coarseg
                    trainLoader.dataset.image_dic[image_path]['coarseg'] = torch.from_numpy(coarseg)
                noise_thresh*=0.99
                noise_thresh[0]=max(noise_thresh[0], 0.99)
                noise_thresh=np.clip(noise_thresh, 0.95, 1)
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
                        valid_img_batch = np.concatenate((valid_img_batch, coarseg), dim=0)
                    predic_all = test_single_case(net, valid_img_batch,stride=stride, patch_size=test_patch_size)
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

            filename = config_train['model_save_name'] + "_cur.tar".format(valid_dice_mean)
            if valid_dice_mean > best_dice:
                best_dice =valid_dice_mean
                is_best = True
                bestname = config_train['model_save_name'] + "_best.tar".format(valid_dice_mean)
            else:
                is_best = False
                bestname = None
            save_model.save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_dice': best_dice,
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=filename, bestname=bestname)
            Adamscheduler.step(epoch_dice['valid_dice_mean'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='', help='Config file path')
    args = parser.parse_args()
    assert(os.path.isfile(args.config_path))
    train(args.config_path)
