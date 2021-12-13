## It's the official code of ['One-shot Weakly-Supervised Segmentation in Medical Images'](https://arxiv.org/abs/2111.10773)
## Abstract
Deep neural networks usually require accurate and a large number of annotations to achieve outstanding performance in medical image segmentation. One-shot segmentation and weakly-supervised learning are promising research directions that lower labeling effort by learning a new class from only one annotated image and utilizing coarse labels instead, respectively. Previous works usually fail to leverage the anatomical structure and suffer from class imbalance and low contrast problems. Hence, we present an innovative framework for 3D medical image segmentation with one-shot and weakly-supervised settings. Firstly a propagation-reconstruction network is proposed to project scribbles from annotated volume to unlabeled 3D images based on the assumption that anatomical patterns in different human bodies are similar. Then a dual-level feature denoising module is designed to refine the scribbles based on anatomical- and pixel-level features. After expanding the scribbles to pseudo masks, we could train a segmentation model for the new class with the noisy label training strategy. Experiments on one abdomen and one head-and-neck CT dataset show the proposed method obtains significant improvement over the state-of-the-art methods and performs robustly even under severe class imbalance and low contrast.  
![image](https://github.com/LWHYC/OneShot_WeaklySeg/blob/main/train_framework.png)

## Requirements
Pytorch >= 1.4, SimpleITK >= 1.2, scipy >= 1.3.1, nibabel >= 2.5.0, GeodisTK and some common packages.

## Usages
- Prepare StructSeg2019 task1 data and split them into three folders: train, valid and test. (Each patient's CT image and label should be in a individual folder in train, valid or test folder) ;
### Train 
- Preprocess the data by `data_process/Resample_and_norm_data.py`;
- Change the `data_root` in `config/train_prnet_structseg.txt` to your data root;
- To train the PRNet, run `python train/train_prnet_structseg.py`. Your model is saved as `prnet_save_name` in `config/train_prnet_structseg.txt`. You should train two models: coarse and fine. Details could befound in ['Contrastive Learning of Relative Position Regression for One-Shot Object Localization in 3D Medical Images'](https://arxiv.org/abs/2012.07043).
### Generate Pesudo Labels
- After the model training, you should select one volume in `train` folder, and draw few scribbles on it as the support volume.
- Run `python inference/Two_stage_scribble_fc.py` to localize scribble on unlabeled volumes.
- Run `python data_process/Generate_distance_and_label_structseg.py` to generate geodesic distance.
- Run `python/Extract_label_from_geodesic2.py` to generate pseudo segmentation.
### Train Segmentation Model
- Run `python train_seg.py` to train a seg model.
