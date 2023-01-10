import numpy as np
from data_process.data_process_func import * 
from util.train_test_func import *
from inference.localization_functions import *


def multi_simi_denoise(config_data, sample_batch, Coarse_RD, Fine_RD):

    noise_thresh = config_data['noise_thresh']
    patch_size = np.asarray(config_data['patch_size'])
    judge_noise = config_data['judge_noise']
    iter_patch_num = 5
    random_crop = RandomPositionCrop(patch_size, padding=False)
    spacing = sample_batch['spacing'].cpu().data.numpy().squeeze()
    scribble_array = np.zeros_like(sample_batch['image'].cpu().data.numpy().squeeze())

    noise_array_dic = {}
    noise_array = np.ones_like(sample_batch['image'].cpu().data.numpy().squeeze())
    for key in ['center_feature5', 'center_feature6', 'center_feature7', 'center_feature8']: # the multi-feature from prnet
        noise_array_dic[key] = np.zeros_like(sample_batch['image'].cpu().data.numpy().squeeze())
    sample_batch['image'] = pad(sample_batch['image'].cpu().data.numpy().squeeze(), patch_size)
    ss = sample_batch['image'].shape
    query_position,query_batch = [],[]
    
    '''randomly select several initial points'''
    for _ in range(iter_patch_num):
        sample = random_crop(sample_batch)
        random_position = np.int16(sample['random_position']).squeeze()
        random_cor = np.around(random_position/spacing).astype(np.int16)
        query_position.append(random_position)
        query_batch.append(sample_batch['image'][
                        random_cor[0] - patch_size[0] // 2:random_cor[0] + patch_size[0] // 2,
                        random_cor[1] - patch_size[1] // 2:random_cor[1] + patch_size[1] // 2,
                        random_cor[2] - patch_size[2] // 2:random_cor[2] + patch_size[2] // 2][np.newaxis])
    query_batch = np.asarray(query_batch) # [iter_patch_num,1,d,w,h]

    
    Coarse_RD.cal_query_position(query_batch, mean=True)

    for key in Coarse_RD.support_position.keys(): # Calculate relative distance for every 32 support points
        ognb = int(key.split('_')[0])
        
        ''' coarse movement'''
        relative_position = Coarse_RD.cal_RD(key=key) # [32,3]
        cur_position = np.mean(np.asarray(query_position), axis=0) + relative_position # [32, 3]

        ''' fine movement'''
        for dim in range(3):
            cur_position[:,dim] = np.minimum(np.maximum(cur_position[:,dim], spacing[dim]*patch_size[dim]//2), spacing[dim]*(ss[dim]-patch_size[dim]//2-1))
        fine_query_batch = []
        cur_cor = np.around(cur_position/spacing).astype(np.int16)
        
        for iiii in range(cur_position.shape[0]):
            fine_query_batch.append(sample_batch['image'][
                        cur_cor[iiii,0] - patch_size[0] // 2:cur_cor[iiii,0] + patch_size[0] // 2,
                        cur_cor[iiii,1] - patch_size[1] // 2:cur_cor[iiii,1] + patch_size[1] // 2,
                        cur_cor[iiii,2] - patch_size[2] // 2:cur_cor[iiii,2] + patch_size[2] // 2][np.newaxis,:])
        fine_query_batch = np.asarray(fine_query_batch)
        Fine_RD.cal_query_position(fine_query_batch)
        relative_position = Fine_RD.cal_RD(key=key)
        cur_position = np.float16(cur_position)+ relative_position # [32,3]
        predic_cor = np.around(cur_position.copy()/spacing).astype(np.int16) #[32,3]
        for dim in range(3):
            predic_cor[:, dim] = np.minimum(np.maximum(predic_cor[:,dim], patch_size[dim]//2), ss[dim]-patch_size[dim]//2-1)    
        
        ''' judge noise'''
        if judge_noise:
            noise_query_batch = []
            for iii in range(predic_cor.shape[0]): # 32
                noise_query_batch.append(sample_batch['image'][
                            predic_cor[iii,0] - patch_size[0] // 2:predic_cor[iii,0] + patch_size[0] // 2,
                            predic_cor[iii,1] - patch_size[1] // 2:predic_cor[iii,1] + patch_size[1] // 2,
                            predic_cor[iii,2] - patch_size[2] // 2:predic_cor[iii,2] + patch_size[2] // 2][np.newaxis,:])
            noise_query_batch = np.asarray(noise_query_batch)
            noise_dic = Fine_RD.cal_noise(noise_query_batch, key=key)

        ''' project cor and noise'''
        predic_cor = predic_cor.tolist()
        for iiii in range(len(predic_cor)):
            scribble_array[predic_cor[iiii][0]-patch_size[0]//2,predic_cor[iiii][1]-patch_size[1]//2,predic_cor[iiii][2]-patch_size[2]//2]=ognb
            if judge_noise:
                for nkey in noise_dic.keys():
                    noise_array_dic[nkey][predic_cor[iiii][0]-patch_size[0]//2,predic_cor[iiii][1]-patch_size[1]//2,predic_cor[iiii][2]-patch_size[2]//2]=noise_dic[nkey][iiii]
    
    ''' thresh the scribble'''
    if judge_noise:
        for key in noise_array_dic.keys():
            noise_array *= noise_array_dic[key].astype(np.float32)
        scribble_array *= 1*(noise_array>noise_thresh)

    return scribble_array