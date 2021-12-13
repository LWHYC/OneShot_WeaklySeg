#!/usr/bin/env bash
eps=200
baselr=0.0001
server=SH-IDC1-10-5-38-[151-152]
gpus=1
batchsz=2
weightCE=5.0
num_workers=1
#load_model='/mnt/lustre/wangna2/Vessel/save_model1986/0730_avseg_unetpp3d_cps160/unetpp_epoch_80.model'
#load_model='/mnt/lustre/wangna2/Vessel/save_model/1118_avseg_unetpp3d_adam_cps160/unetpp_epoch_18.model'
#load_model="/mnt/lustreold/wangna2/Vessel/save_model/1201_avseg_unetpp3d_adam_cps[48, 320, 320]/unetpp_epoch_70.model"
#load_model="/mnt/lustreold/wangna2/Vessel/save_model/0128_avseg_unet_GroupGC_cps[64, 288, 288]/unetpp_epoch_10.model"
#load_model="/mnt/lustre/wangna2/Vessel/save_model/0401_avseg_unet_GroupGC_MixCropsize_alldata_cps[160, 192, 192]/unetpp_epoch_9.model"
# /home/leiwenhui/leiwenhui/Data/Pancreas
# /mnt/lustre/leiwenhui.vendor/Data/Pancreas
srun -p MIA -n1 --mpi=pmi2 --cpus-per-task=$num_workers --gres=gpu:$gpus --job-name=odtf \
python -u organ_detection_trible_stage_full_size_distance_based.py 
# --gpu='0,1,2,3' --optim='ADAM' --num_workers=$num_workers --lr=$baselr --epochs=$eps --step='10, 20, 30, 40' --batch-size=$batchsz --weightCE=$weightCE \
# --data_dir=$data_dir --load_model="${load_model}" --attention_type=$attention_type --save_model_dir=$save_model_dir --log=$log



#srun -p mia -n1 --mpi=pmi2 --job-name=avbseg python -u train_unet3d_att.py --gpu='0,1,2,3' \
#--optim='ADAM' --num_workers=$num_workers --lr=$baselr --epochs=$eps --step='10, 20, 30, 40' --batch-size=$batchsz --weightCE=$weightCE \
#--data_dir=$data_dir --load_model="${load_model}" --attention_type=$attention_type --save_model_dir=$save_model_dir --log=$log



##!/usr/bin/env bash
#eps=200
#baselr=0.0001
#server=SH-IDC1-10-5-38-[151-152]
#gpus=4
#batchsz=4
#weightCE=5.0
#num_workers=4
#attention_type="Group_GC"
##load_model='/mnt/lustre/wangna2/Vessel/save_model1986/0730_avseg_unetpp3d_cps160/unetpp_epoch_80.model'
##load_model='/mnt/lustre/wangna2/Vessel/save_model/1118_avseg_unetpp3d_adam_cps160/unetpp_epoch_18.model'
#load_model="/mnt/lustre/wangna2/Vessel/save_model/1201_avseg_unetpp3d_adam_cps[48, 320, 320]/unetpp_epoch_70.model"
#data_dir='/mnt/lustre/wangna2/Vessel/AV_data/bin_vessel_network'
#save_model_dir='/mnt/lustre/wangna2/Vessel/save_model/0128_avseg_unet_GroupGC_2'
#log='/mnt/lustre/wangna2/Vessel/log/train_0128_avseg_unet_GroupGC_2.log'
#
#
#srun -p mia -n1 --mpi=pmi2 --gres=gpu:$gpus --job-name=uGgc2 python -u train_unet3d_att.py --gpu='0,1,2,3,4,5,6,7' \
#--optim='ADAM' --num_workers=$num_workers --lr=$baselr --epochs=$eps --step='10, 20, 30, 40' --batch-size=$batchsz --weightCE=$weightCE \
#--data_dir=$data_dir --attention_type=$attention_type --save_model_dir=$save_model_dir --log=$log







##!/usr/bin/env bash
#eps=200
#baselr=0.0001
#server=SH-IDC1-10-5-38-[151-152]
#gpus=2
#batchsz=2
#weightCE=5.0
#num_workers=2
#attention_type="Aug_GC"
##load_model='/mnt/lustre/wangna2/Vessel/save_model1986/0730_avseg_unetpp3d_cps160/unetpp_epoch_80.model'
##load_model='/mnt/lustre/wangna2/Vessel/save_model/1118_avseg_unetpp3d_adam_cps160/unetpp_epoch_18.model'
#load_model="/mnt/lustre/wangna2/Vessel/save_model/1201_avseg_unetpp3d_adam_cps[48, 320, 320]/unetpp_epoch_70.model"
#data_dir='/mnt/lustre/wangna2/Vessel/AV_data/bin_vessel_network'
#save_model_dir='/mnt/lustre/wangna2/Vessel/save_model/0128_avseg_unet_AugGC'
#log='/mnt/lustre/wangna2/Vessel/log/train_0128_avseg_unet_AugGC.log'
#
#
#srun -p mia -n1 --mpi=pmi2 --gres=gpu:$gpus --job-name=uAgc python -u train_unet3d_att.py --gpu='0,1,2,3,4,5,6,7' \
#--optim='ADAM' --num_workers=$num_workers --lr=$baselr --epochs=$eps --step='10, 20, 30, 40' --batch-size=$batchsz --weightCE=$weightCE \
#--data_dir=$data_dir --attention_type=$attention_type --save_model_dir=$save_model_dir --log=$log