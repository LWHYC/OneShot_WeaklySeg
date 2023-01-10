import h5py
import math
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from collections import OrderedDict


def test_single_case(net, image, stride, patch_size):
    _, w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(0,0),(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    _,ww,hh,dd = image.shape

    sz = math.ceil((ww - patch_size[0]) / stride[0]) + 1
    sy = math.ceil((hh - patch_size[1]) / stride[1]) + 1
    sx = math.ceil((dd - patch_size[2]) / stride[2]) + 1
    # print("{}, {}, {}".format(sz, sy, sx))
    predic_all = {}
    cnt = np.zeros(image.shape[1::]).astype(np.float32)

    for z in range(0, sz):
        zs = min(stride[0]*z, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride[1] * y,hh-patch_size[1])
            for x in range(0, sx):
                xs = min(stride[2] * x, dd-patch_size[2])
                test_patch = image[:,zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]]
                test_patch = (np.expand_dims(test_patch,axis=0)).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                predic = net(test_patch)  # 字典，包含有网络所输出的结果,如prob_1
                for key in predic.keys():
                    if torch.is_tensor(predic[key]):
                        if len(predic[key].shape) == 5:
                            if key in predic_all.keys():
                                predic_all[key][:, zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]] += predic[key].cpu().data.numpy()[0]
                            else:
                                predic_all[key] = np.zeros((predic[key].shape[1], ) + image.shape[1::]).astype(np.float32)
                                predic_all[key][:, zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]] += predic[key].cpu().data.numpy()[0]
                cnt[zs:zs+patch_size[0], ys:ys+patch_size[1], xs:xs+patch_size[2]] += 1
    for key in list(predic_all.keys()):
        predic_all[key] /= np.expand_dims(cnt, axis=0)
        predic_all[key]  = predic_all[key] [:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        if 'prob' in key:
            predic_all[key.replace('prob','label')]=np.argmax(predic_all[key], axis=0).astype(np.int16)

    return predic_all


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp)+0.01)
        total_dice[i - 1] += dice

    return total_dice

