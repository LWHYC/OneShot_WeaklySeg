# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
from __future__ import absolute_import, print_function
import os
import sys
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from util.parse_config import parse_config
from data_process.data_process_func import load_nifty_volume_as_array
from util.evaluation_index import dc, assd

def get_largest_component(img):
    s = ndimage.generate_binary_structure(3,1) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) 
    max_label = np.where(sizes == sizes.max())[0] + 1
    return labeled_array == max_label

def binary_dice3d(s,g):
    assert(len(s.shape)==2)
    prod = np.multiply(s, g)
    s0 = prod.sum(axis=-1)
    s1 = s.sum(axis=-1)
    s2 = g.sum(axis=-1)
    dice = 2.0*s0/(s1 + s2 + 0.00001)
    return dice[1::]

def dice_of_binary_volumes(s_name, g_name):
    s = load_nifty_volume_as_array(s_name)
    g = load_nifty_volume_as_array(g_name)
    dice = binary_dice3d(s, g)
    return dice

def one_hot(img, nb_classes):
    hot_img = np.zeros([nb_classes]+list(img.shape))
    for i in range(nb_classes):
        hot_img[i][np.where(img == i)] = 1
    return hot_img

def evaluation(folder, classnum=6, save=False):
    patient_list = os.listdir(folder)
    dice_all_data = []
    assd_all_data = []
    for patient in patient_list:
        s_name = os.path.join(folder, patient , 'geo_seg.nii.gz')
        g_name = os.path.join(folder, patient , 'rex_label.nii.gz')
        #s_volume = np.int64(np.load(s_name))
        s_volume = load_nifty_volume_as_array(s_name)
        g_volume = load_nifty_volume_as_array(g_name)
        s_volume = one_hot(s_volume, nb_classes=classnum)
        g_volume = one_hot(g_volume, nb_classes=classnum)
        dice_list=[]
        assd_list=[]
        for i in range(classnum):
            temp_dice = dc(s_volume[i], g_volume[i])
            temp_assd = assd(s_volume[i], g_volume[i],voxelspacing=[3, 1, 1])
            dice_list.append(temp_dice)
            assd_list.append(temp_assd)
        dice_all_data.append(dice_list)
        assd_all_data.append(assd_list)
        print(patient, dice_list)
        if save:
         np.savetxt(os.path.join(folder, patient + '/Inter_dice.txt'), np.asarray(dice_list))
    dice_all_data = np.asarray(dice_all_data)
    dice_mean = [dice_all_data.mean(axis = 0)]
    dice_std  = [dice_all_data.std(axis = 0)]
    if save:
        np.savetxt(folder + '/dice_all.txt', dice_all_data)
        np.savetxt(folder + '/dice_mean.txt', dice_mean)
        np.savetxt(folder + '/dice_std.txt', dice_std)
    print('dice mean ', dice_mean)
    print('dice std  ', dice_std)
    assd_all_data = np.asarray(assd_all_data)
    assd_mean = [assd_all_data.mean(axis = 0)]
    assd_std  = [assd_all_data.std(axis = 0)]
    print('assd mean ', assd_mean)
    print('assd std  ', assd_std)
    if save:
        np.savetxt(folder + '/assd_all.txt', assd_all_data)
        np.savetxt(folder + '/assd_mean.txt', assd_mean)
        np.savetxt(folder + '/assd_std.txt', assd_std)

# if __name__ == '__main__':
#     if(len(sys.argv) != 2):
#         print('Number of arguments should be 2. e.g.')
#         print('    python util/dice_evaluation.py config.txt')
#         exit()
folder = '../../../Data/TCIA/train/'
evaluation(folder, classnum=16, save=False)
