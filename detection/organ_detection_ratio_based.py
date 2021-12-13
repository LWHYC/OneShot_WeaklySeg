#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import time
import torch.optim as optim
import torch.tensor
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms
from dataloaders.Position_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from NetFactory import NetFactory
import matplotlib.pyplot as plt
from losses.loss_function import TestDiceLoss, AttentionExpDiceLoss
from util.visualization.visualize_loss import loss_visualize
from util.visualization.show_param import show_param
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def random_all(random_seed):
    random.seed(random_seed)  # 给定seed value,决定了后面的伪随机序列
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def move(image, inher_position, test_patch_size):
    cur_inher_position = np.asarray(image.shape)*np.asarray(inher_position)
    cur_inher_position = cur_inher_position.astype(np.int16)
    cur_image_patch = image[cur_inher_position[0]:cur_inher_position[0]+test_patch_size[0], cur_inher_position[1]:cur_inher_position[1]+test_patch_size[1],
                      cur_inher_position[2]:cur_inher_position[2]+test_patch_size[2]]
    return cur_image_patch


def test(config_file):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data = config['data']  # 包含数据的各种信息,如data_shape,batch_size等
    config_fnet = config['fnetwork']  # 网络参数,如net_name,base_feature_name,class_num等
    config_pnet = config['pnetwork']
    config_test = config['testing']

    test_patch_size = config_data['test_patch_size']
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed)  # 给定seed value,决定了后面的伪随机序列
    random_crop = RandomPositionDoublePatientCrop(test_patch_size)
    to_tensor = ToDoublePatientPositionTensor()

    cudnn.benchmark = True
    cudnn.deterministic = True

    # 2, load data
    print('2.Load data')
    validData = PositionDoublePatientDataloader(config=config_data,
                                                split='valid',
                                                transform=None)

    def worker_init_fn(worker_id):
        random.seed(random_seed + worker_id)

    validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 3. creat model
    print('3.Creat model')
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
    out_put_size = int(np.asarray(config_data['test_patch_size']).prod() / 512)  # 由于后面是全连接网络，所以Fnet的输出需拉直
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

    show_param(fnet)  # 计算网络总参数量
    show_param(pnet)

    # 4, start to train
    print('4.Start to train')
    with torch.no_grad():
        fnet.eval(), pnet.eval()
        for ii_batch, sample_batch in enumerate(validLoader):
            print(sample_batch['image_path_0'], sample_batch['image_path_1'])
            move_step = 0
            sample_batch['image_0'] = sample_batch['image_0'].cpu().data.numpy().squeeze()
            sample_batch['image_1'] = sample_batch['image_1'].cpu().data.numpy().squeeze()
            sample = random_crop(sample_batch)
            sample = to_tensor(sample)
            target_batch, cur_batch, rela_poi, target_inher_position, cur_inher_position= sample['random_crop_image_0'].cuda().unsqueeze(0), \
                                                  sample['random_crop_image_1'].cuda().unsqueeze(0), sample[
                                                      'rela_poi'].cuda(), sample['random_inher_position_0'], sample['random_inher_position_1']
            target_feature = fnet(target_batch)
            target_position = np.int16(target_inher_position*np.asarray(sample_batch['image_0'].shape))
            initial_position = np.int16(cur_inher_position*np.asarray(sample_batch['image_1'].shape))
        # while move_step<5:
            cur_feature = fnet(cur_batch)
            predic = pnet(target_feature, cur_feature).cpu().data.numpy().squeeze()/20
            #print(predic)
            cur_inher_position = cur_inher_position + predic
            cur_position = np.int16(cur_inher_position*np.asarray(sample_batch['image_1'].shape))
            print('target inher position',target_inher_position,'cur inher position',cur_inher_position,'cur inher loss',target_inher_position-cur_inher_position)
            print('target position', target_position, 'cur position', cur_position)
            cur_batch = move(sample_batch['image_1'], cur_inher_position, test_patch_size=test_patch_size)
            cur_batch = torch.from_numpy(cur_batch).cuda().unsqueeze(0).unsqueeze(0).float()
            move_step+=1
            print(sample_batch['image_1'].shape, initial_position)
            plt.figure(dpi=300)
            plt.subplot(131)
            plt.imshow(sample_batch['image_0'][target_position[0]], cmap='gray')
            plt.plot(target_position[1], target_position[2], '*', c='r')
            plt.title('target position')
            plt.subplot(132)
            plt.imshow(sample_batch['image_1'][initial_position[0]], cmap='gray')
            plt.plot(initial_position[1], initial_position[2], '*', c='r')
            plt.title('initial position')
            plt.subplot(133)
            plt.imshow(sample_batch['image_1'][cur_position[0]], cmap='gray')
            plt.plot(cur_position[1], cur_position[2], '*', c='r')
            plt.title('detected position')
            plt.show()



if __name__ == '__main__':
    config_file = str('/media/lwh/09b9fdae-489a-484b-8cc7-a0a31e663bdd/代码/DeepLearning/Pytorch-project/Poistion_Detection/config/test_position.txt')
    assert (os.path.isfile(config_file))
    test(config_file)
