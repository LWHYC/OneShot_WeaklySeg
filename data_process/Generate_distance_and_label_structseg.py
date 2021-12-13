import os
import sys
import argparse
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import GeodisTK
import SimpleITK as sitk
from scipy import  ndimage
import os
from data_process.data_process_func import load_nifty_volume_as_array,save_array_as_nifty_volume
import numpy as np
from skimage import morphology
from multiprocessing import Pool

def geodesic_distance(img, seed, distance_threshold):
    threshold = distance_threshold
    if(seed.sum() > 0):
        geo_dis = GeodisTK.geodesic3d_raster_scan(img, seed, [3,1,1], 0.5, 1)
        geo_dis[np.where(geo_dis > threshold)] = threshold
        dis = 1-geo_dis/threshold  # recale to 0-1
    else:
        dis = np.zeros_like(img, np.float32)
    return dis


def get_largest_component(img, print_info=False, threshold=False):
    """
    Get the largest component of a binary volume
    inputs:
        img: the input 3D_train volume
        threshold: a size threshold
    outputs:
        out_img: the output volume
    """
    s = ndimage.generate_binary_structure(3, 1)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if (print_info):
        print('component size', sizes_list)
    if (len(sizes) <= 1):
        out_img = img
    else:
        if threshold:
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if (temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            out_img = labeled_array == max_label1
    return out_img

def transfer_arg2label(arg, label_ls):
    oh_arg = np.zeros([max(label_ls)+1]+list(arg.shape)).astype(np.uint8)
    for i in range(len(label_ls)):
        oh_arg[label_ls[i]][arg==(i+1)]=1
    label = oh_arg.argmax(0).astype(np.int16)
    return label

def extract_label(file_path, dis, label_ls):   
    label = np.argmax(dis, axis=0).astype(np.float32)
    nlabel = transfer_arg2label(label, label_ls)
    for clabel in label_ls:
        nnlabel = (nlabel==clabel).astype(np.uint8)
        nnlabel = morphology.binary_erosion(nnlabel, np.ones([3,5,5])).astype(np.int16)
        nnlabel = get_largest_component(nnlabel)
        nnlabel = morphology.binary_dilation(nnlabel, np.ones([3,5,5])).astype(np.int16)
        nlabel[nlabel==clabel]=0
        nlabel[nnlabel==1]=clabel
    save_path=os.path.join(file_path, 'geo_seg.nii.gz')
    ndata = sitk.GetImageFromArray(nlabel)
    sitk.WriteImage(ndata, save_path)
    return

def generate_distance_and_label(label_path, img_path, item_path, fgname_ls, fglabel_ls, bglabel):
    label = load_nifty_volume_as_array(label_path).astype(np.uint8)
    image = load_nifty_volume_as_array(img_path).astype(np.float32)
    fg_dis = []

    # Generating geodes distance based on localized scribbles
    for i in range(len(fglabel_ls)):
        nlabel = (label==fglabel_ls[i]).astype(np.uint8)
        fg_dis.append(geodesic_distance(image, nlabel,1))
    nnlabel = (label==bglabel).astype(np.uint8)
    backdis = geodesic_distance(image, nnlabel, 1)
    fg_dis = np.array(fg_dis)
    geo_dis = [backdis]

    # Saving distance. It could be visualized directly by Itk-Snap
    for i in range(fg_dis.shape[0]):
        curfg_dis = fg_dis[i]
        geo_dis.append(curfg_dis)
        save_path = os.path.join(item_path, fgname_ls[i])
        save_array_as_nifty_volume(curfg_dis, save_path)
    save_path = os.path.join(item_path, 'geo_dis_back.nii.gz')
    save_array_as_nifty_volume(backdis, save_path)

    # Saving pseudo label from distance map
    extract_label(item_path, geo_dis, fglabel_ls)
    print(item_path)
    return


if  __name__ == '__main__':
    data_root = '../../../Data/HaN_structseg/'
    mode = ['train','valid', 'test']
    fglabel_ls = [1,6,7] # the foreground class of each label
    fgname_ls = ['geo_dis_brainstem.nii.gz', 'geo_dis_leftparotid.nii.gz', 'geo_dis_rightparotid.nii.gz']
    bglabel=8  # you can change it to your background label
    for submode in mode:
        for item in os.listdir(os.path.join(data_root, submode)):
            item_path = os.path.join(data_root, submode, item)
            label_path=os.path.join(item_path, 'os_scribble.nii.gz')
            img_path=os.path.join(item_path, 'norm_rdata.nii.gz')
            generate_distance_and_label(label_path, img_path, item_path, fgname_ls, fglabel_ls, bglabel)
            print(item_path)