import os
import sys
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import GeodisTK
import os
from data_process.data_process_func import get_bound_coordinate
import numpy as np
from multiprocessing import Pool

def geodesic_distance(img, seed, distance_threshold, spacing, bound_crop=False):
    threshold = distance_threshold
    if(seed.sum() > 0):
        print(img.dtype, seed.dtype, spacing)
        geo_dis = GeodisTK.geodesic3d_raster_scan(img, seed, spacing, 1.0, 2)
        geo_dis[np.where(geo_dis > threshold)] = threshold
        dis = 1-geo_dis/threshold  # recale to 0-1
        if bound_crop:
            bound_cor = np.array(get_bound_coordinate(seed))
            bound_cor = get_bound_coordinate(seed, pad=((bound_cor[1]-bound_cor[0])).astype(np.int32))
            aa = np.zeros_like(dis)
            aa[bound_cor[0][0]:bound_cor[1][0], bound_cor[0][1]:bound_cor[1][1], bound_cor[0][2]:bound_cor[1][2]] = 1
            dis *= aa
    else:
        dis = np.zeros_like(img, np.float32)
    
    return dis

def transfer_arg2label(arg, label_ls):
    oh_arg = np.zeros([max(label_ls)+1]+list(arg.shape)).astype(np.uint8)
    for i in range(len(label_ls)):
        oh_arg[label_ls[i]][arg==(i+1)]=1
    label = oh_arg.argmax(0).astype(np.int16)
    return label

def extract_label(dis, label_ls):   
    label = np.argmax(dis, axis=0).astype(np.float32)
    nlabel = transfer_arg2label(label, label_ls)
    for clabel in label_ls:
        nnlabel = (nlabel==clabel).astype(np.uint8)
        nlabel[nlabel==clabel]=0
        nlabel[nnlabel==1]=clabel
    return nlabel

def geode_seg(scribble, image, spacing, fglabel_ls, bglabel):
    geo_dis = []

    # Generating geodes distance based on localized scribbles
    print('bg dis')
    nnlabel = (scribble==bglabel).astype(np.uint8)
    geo_dis.append(geodesic_distance(image, nnlabel, distance_threshold=1, spacing=spacing))

    for i in range(len(fglabel_ls)):
        print(i, 'fg dis')
        nlabel = (scribble==fglabel_ls[i]).astype(np.uint8)
        geo_dis.append(geodesic_distance(image, nlabel, distance_threshold=1, spacing=spacing, bound_crop=False))
    

    # Generating pseudo label from distance map
    geoseg = extract_label(np.array(geo_dis), fglabel_ls)
    return geoseg   