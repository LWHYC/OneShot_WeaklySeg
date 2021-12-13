import GeodisTK
from scipy import  ndimage
import os
#from scipy import ndimage
from data_process.data_process_func import load_nifty_volume_as_array,save_array_as_nifty_volume
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
def geodesic_distance(img, seed, distance_threshold):
    threshold = distance_threshold
    if(seed.sum() > 0):
        geo_dis = GeodisTK.geodesic3d_raster_scan(img, seed, 1.0, 2)
        geo_dis[np.where(geo_dis > threshold)] = threshold
        dis = 1-geo_dis/threshold  # recale to 0-1
    else:
        dis = np.zeros_like(img, np.float32)
    return dis

def get_center_cor(img):
    '''
    get 2d binary img center corardiate
    :param img: 2d binary img
    :return:
    '''
    contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0]) # 计算第一条轮廓的各阶矩，字典模式
    center_x = int(M["m10"]/M["m00"])
    center_y = int(M["m01"]/M["m00"])
    return center_x, center_y

def euclidean_distance(seed, dis_threshold, spacing=[5, 1, 1]):
    threshold = dis_threshold
    if(seed.sum() > 0):
        euc_dis = ndimage.distance_transform_edt(seed==0, sampling=spacing)
        euc_dis[euc_dis > threshold] = threshold
        dis = 1-euc_dis/threshold
    else:
        dis = np.zeros_like(seed, np.float32)
    return dis

'''
读取label，根据label在每张slice中心，生成距离图
'''
if  __name__ == '__main__':
    data_root = '/home/uestc-c1501c/StructSeg/Lung_GTV'
    mode = ['train', 'valid']
    show = True # 是否可视化康康
    geo = False
    blur = True
    geodis_savename = 'eue_dis.nii.gz' #存储距离图的名称
    for submode in mode:
        for item in os.listdir(os.path.join(data_root, submode)):
            item_path = os.path.join(data_root, submode, item)
            label = load_nifty_volume_as_array(os.path.join(item_path, 'label.nii.gz'))
            mask = np.where(label==1)
            nlabel = np.zeros_like(label)
            seeds = np.zeros_like(label)
            nlabel[mask]=1
            print(item, np.sum(nlabel))
            image = load_nifty_volume_as_array(os.path.join(item_path, 'data.nii.gz'))
            for i in range(nlabel.shape[0]):
                if np.sum(nlabel[i]) >=10:
                    center_x, center_y = get_center_cor(nlabel[i])
                    seeds[i, center_y-2:center_y+2, center_x-2:center_x+2]=1
            if blur:
                nlabel = nlabel.astype(np.float)
                for i in range(nlabel.shape[0]):
                    #nlabel[i] = cv2.dilate(nlabel[i], (16,16))
                    nlabel[i]=morphology.dilation(nlabel[i], np.ones([3,3]))
                    nlabel[i] = cv2.GaussianBlur(nlabel[i], (5, 5), 10)

            if geo:
                dis = geodesic_distance(image, seeds, 300)
                dis *= nlabel
            else:
                dis = euclidean_distance(seeds, 30, [5,1,1])
                dis *= nlabel
            if show:
                f, plots = plt.subplots(1, 3)
                plots[0].imshow(label[-50])
                plots[1].imshow(nlabel[-50])
                plots[2].imshow(dis[-50])
                plt.show()
            save_array_as_nifty_volume(dis, os.path.join(item_path, geodis_savename))