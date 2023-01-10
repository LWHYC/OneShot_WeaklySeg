CUDA_VISIBLE_DEVICES=0 python train/train_prnet.py --config_path config/train/train_prnet_tcia_large.txt &&
CUDA_VISIBLE_DEVICES=0 python train/train_prnet.py --config_path config/train/train_prnet_tcia_small.txt &&
CUDA_VISIBLE_DEVICES=0 python train/train_prnet.py --config_path config/train/train_prnet_structseg_large.txt &&
CUDA_VISIBLE_DEVICES=0 python train/train_prnet.py --config_path config/train/train_prnet_structseg_small.txt 