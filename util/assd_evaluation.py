# -*- coding: utf-8 -*-
import os
import math
import random
import numpy as np
from scipy import ndimage

def one_hot(img, nb_classes):
    hot_img = np.zeros([nb_classes]+list(img.shape))
    for i in range(nb_classes):
        hot_img[i][np.where(img == i)] = 1
    return hot_img

def binary_assd3d(s, g, spacing = [1.0, 1.0, 1.0]):
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg  and Hs==Hg and Ws==Wg)
    scale = [1.0, 1/spacing[1], 1/spacing[2]]
    s_resample = ndimage.interpolation.zoom(s, scale, order = 0)
    g_resample = ndimage.interpolation.zoom(g, scale, order = 0)
    point_list_s = volume_to_surface(s_resample)    # 含所有边界点的列表
    point_list_g = volume_to_surface(g_resample)
    new_spacing = [spacing[0], 1.0, 1.0]
    dis_array1 = assd_distance_from_one_surface_to_another(point_list_s, point_list_g, new_spacing)
    dis_array2 = assd_distance_from_one_surface_to_another(point_list_g, point_list_s, new_spacing)
    assd = (dis_array1.sum() + dis_array2.sum())/(len(dis_array1) + len(dis_array2))
    return assd
    
def assd_distance_from_one_surface_to_another(point_list_s, point_list_g, spacing):
    dis_square = 0.0
    n_max = 500
    if(len(point_list_s) > n_max):
        point_list_s = random.sample(point_list_s, n_max)
    distance_array = np.zeros(len(point_list_s))
    for i in range(len(point_list_s)):
        ps = point_list_s[i]
        ps_nearest = 1e10
        for pg in point_list_g:
            dd = spacing[0]*(ps[0] - pg[0])
            dh = spacing[1]*(ps[1] - pg[1])
            dw = spacing[2]*(ps[2] - pg[2])
            temp_dis_square = dd*dd + dh*dh + dw*dw
            if(temp_dis_square < ps_nearest):
                ps_nearest = temp_dis_square
        distance_array[i] = math.sqrt(ps_nearest)
    return distance_array

def volume_to_surface(img):
    strt = ndimage.generate_binary_structure(3,2)
    img  = ndimage.morphology.binary_closing(img, strt, 5)
    point_list = []
    [D, H, W] = img.shape
    offset_d  = [-1, 1,  0, 0,  0, 0]
    offset_h  = [ 0, 0, -1, 1,  0, 0]
    offset_w  = [ 0, 0,  0, 0, -1, 1]
    for d in range(1, D-1):
        for h in range(1, H-1):
            for w in range(1, W-1):
                if(img[d, h, w] > 0):
                    edge_flag = False
                    for idx in range(6):
                        if(img[d + offset_d[idx], h + offset_h[idx], w + offset_w[idx]] == 0):  # 在6个方向上迈一步只要有一个为0,该点就为边界点
                            edge_flag = True
                            break
                    if(edge_flag):
                        point_list.append([d, h, w])
    return point_list





# if __name__ == '__main__':
#     if(len(sys.argv) != 2):
#         print('Number of arguments should be 2. e.g.')
#         print('    python util/dice_evaluation.py config.txt')
#         exit()