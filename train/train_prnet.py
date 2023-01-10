#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import argparse
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from dataloaders.Position_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from torch.utils.tensorboard import SummaryWriter
from networks.NetFactory import NetFactory
from util.save_model import Save_checkpoint
import argparse

def random_all(random_seed):
    random.seed(random_seed) 
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def train(config_file):
    # 1, load configuration parameters
    writer = SummaryWriter()
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data  = config['data']    
    config_prnet  = config['prnetwork'] 
    config_train = config['training']

    patch_size = config_data['patch_size']
    batch_size = config_data.get('batch_size', 4)
    dis_ratio = torch.FloatTensor(config_data.get('distance_ratio')).unsqueeze(dim=0).cuda()
    lr = config_train.get('learning_rate', 1e-3)
    best_loss = config_train.get('best_loss', 0.5)
    random_seed = config_train.get('random_seed', 1)
    num_workers = config_train.get('num_workers', 1)
    load_memory = config_train.get('load_memory', False)
    small_move  = config_train.get('small_move', False)
    fluct_range = config_train.get('fluct_range')
    random_all(random_seed)    
    random_crop = RandomDoubleCrop(patch_size, small_move=small_move, fluct_range=fluct_range)
    to_tensor = ToPositionTensor()
    save_model = Save_checkpoint()
    cudnn.benchmark = True
    cudnn.deterministic = True

    # 2, load data
    print('2.Load data')
    trainData = PositionDataloader(config=config_data,
                                   image_name_list=config_data['train_image_list'],
                                   transform=transforms.Compose([
                                       RandomDoubleCrop(patch_size, small_move=small_move, fluct_range=fluct_range),
                                       RandomDoubleMask(max_round=16,mask_size=np.array(patch_size)//4),
                                       ToPositionTensor(),
                                   ]),
                                   load_memory=load_memory,
                                   random_sample=True,
                                   out_size = config_data['patch_size'])
    config_data['iter_num']=50                            
    validData = PositionDataloader(config=config_data,
                                   image_name_list=config_data['valid_image_list'],
                                   transform=None,
                                   load_memory=load_memory,     
                                   random_sample=True,
                                   out_size = config_data['patch_size'])
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validLoader = DataLoader(validData, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)

    # 3. creat model
    print('3.Creat model')
    net_type   = config_prnet['net_type']
    net_class = NetFactory.create(net_type)
    print('dis ratio', dis_ratio)
    patch_size = np.asarray(config_data['patch_size'])
    prnet = net_class(
                    inc=config_prnet.get('input_channel', 1),
                    patch_size=patch_size,
                    base_chns= config_prnet.get('base_feature_number', 16),
                    norm='in',
                    depth=config_prnet.get('depth', False),
                    dilation=config_prnet.get('dilation', 1),
                    n_classes = config_prnet['class_num'],
                    droprate=config_prnet.get('drop_rate', 0.2),
                    )
    if config_train.get('parallel_training'):
        prnet = torch.nn.DataParallel(prnet).cuda()
    else:
        prnet = prnet.cuda()

    loss_func = nn.MSELoss()
    Adamoptimizer = optim.Adam(prnet.parameters(), lr=lr, weight_decay=config_train.get('decay', 1e-7))
    Adamscheduler = torch.optim.lr_scheduler.StepLR(Adamoptimizer, step_size=15, gamma=0.8)

    if config_train['load_weight']:
        if os.path.isfile(config_train['prnet_load_path']):
            print("=> loading checkpoint '{}'".format(config_train['prnet_load_path']))
            checkpoint = torch.load(config_train['prnet_load_path'])
            best_loss = checkpoint['best_loss']
            prnet.load_state_dict(checkpoint['state_dict'])
            Adamoptimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' ".format(config_train['prnet_load_path']))
        else:
            raise(ValueError("=> no checkpoint found at '{}'".format(config_train['prnet_load_path'])))
    

    # 4, start to train
    print('4.Start to train')
    start_it  = config_train.get('start_iteration', 0)
    print_iter = 0
    for epoch in range(start_it, config_train['maximal_epoch']): 
        print('#######epoch:', epoch)
        optimizer = Adamoptimizer

        'train'
        prnet.train()
        for i_batch, sample_batch in enumerate(trainLoader):
            mask_img_batch0,img_batch0,mask_img_batch1, \
                    img_batch1,fc_label_batch = sample_batch['random_mask_crop_image_0'].cuda(), \
                    sample_batch['random_crop_image_0'].cuda(), \
                    sample_batch['random_mask_crop_image_1'].cuda(), \
                    sample_batch['random_crop_image_1'].cuda(), \
                    sample_batch['rela_distance'].cuda()
            predic_0 = prnet(mask_img_batch0)
            predic_1 = prnet(mask_img_batch1)
            predic_ae_0, predic_cor_fc_0 = torch.sigmoid(predic_0['ae']), predic_0['fc_position']
            predic_ae_1, predic_cor_fc_1 = torch.sigmoid(predic_1['ae']), predic_1['fc_position']
            ae_train_loss = loss_func(predic_ae_0, img_batch0) + loss_func(predic_ae_1, img_batch1)
            fc_predic = dis_ratio*torch.tanh(predic_cor_fc_0-predic_cor_fc_1)
            fc_train_loss = loss_func(fc_predic, fc_label_batch)
            train_loss = ae_train_loss+fc_train_loss
            optimizer.zero_grad()  
            train_loss.backward()  
            optimizer.step() 
            if epoch%config_train['train_step']==0 and i_batch%config_train['print_step']==0:
                ae_train_loss = ae_train_loss.cpu().data.numpy()
                fc_train_loss = fc_train_loss.cpu().data.numpy()
                fc_predic = fc_predic.cpu().data.numpy()[0]
                fc_label_batch = fc_label_batch.cpu().data.numpy()[0]
                train_loss = train_loss.cpu().data.numpy()
                writer.add_scalar('Loss/train_fc', fc_train_loss, print_iter)
                writer.add_scalar('Loss/train_ae', ae_train_loss, print_iter)
                print_iter+=1
                if i_batch ==0:
                    train_loss_array=train_loss
                else:
                    train_loss_array = np.append(train_loss_array, train_loss)
                print('train batch:',i_batch,'loss:', ae_train_loss, fc_train_loss, fc_predic, fc_label_batch)
        Adamscheduler.step()

        'valid'
        if  epoch % config_train['test_step']==0:
            with torch.no_grad():
                prnet.eval()
                for ii_batch, sample_batch in enumerate(validLoader):
                    if sample_batch['image'].dim()==5:
                        sample_batch['image']=sample_batch['image'][0]
                    for  ii_iter in range(config_train['test_iter']):
                        sample = random_crop(sample_batch)
                        sample = to_tensor(sample)
                        img_batch0,img_batch1,fc_label_batch \
                        =sample['random_crop_image_0'].cuda().unsqueeze(dim=0), \
                        sample['random_crop_image_1'].cuda().unsqueeze(dim=0), \
                        sample['rela_distance'].cuda().unsqueeze(dim=0)
                        predic_0 = prnet(img_batch0)
                        predic_1 = prnet(img_batch1)                                  
                        predic_ae_0, predic_cor_fc_0 = torch.sigmoid(predic_0['ae']), predic_0['fc_position']
                        predic_ae_1, predic_cor_fc_1 = torch.sigmoid(predic_1['ae']), predic_1['fc_position']
                        fc_predic = dis_ratio*torch.tanh(predic_cor_fc_0-predic_cor_fc_1)
                        ae_valid_loss = loss_func(predic_ae_0, img_batch0).cpu().data.numpy() + loss_func(predic_ae_1, img_batch1).cpu().data.numpy()
                        fc_valid_loss = loss_func(fc_predic, fc_label_batch).cpu().data.numpy()
                        valid_loss = fc_valid_loss
                        if ii_batch ==0 and ii_iter==0:
                            valid_loss_array = valid_loss
                        else:
                            valid_loss_array = np.append(valid_loss_array, valid_loss)
                    print('valid batch:',ii_batch,' valid loss:', ae_valid_loss, fc_valid_loss)

            epoch_loss = {'valid_loss':valid_loss_array.mean(), 'train_loss':train_loss_array.mean()}
            writer.add_scalar('Loss/valid_fc', valid_loss_array.mean(), epoch)
            t = time.strftime('%X %x %Z')
            print(t, 'epoch', epoch, '\nloss:\n', epoch_loss)

            'save model'
            filename = config_train['prnet_save_name'] + "_cur.tar"
            if epoch_loss['valid_loss'] < best_loss:
                best_loss =epoch_loss['valid_loss']
                is_best = True
                bestname = config_train['prnet_save_name'] + "_best.tar"
            else:
                is_best = False
                bestname = None
            save_model.save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': prnet.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=filename, bestname=bestname)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='', help='Config file path')
    args = parser.parse_args()
    assert(os.path.isfile(args.config_path))
    train(args.config_path)
