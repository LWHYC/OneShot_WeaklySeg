## It's the official code of ['One-shot Weakly-Supervised Segmentation in Medical Images'](https://arxiv.org/abs/2111.10773)
## Abstract
Deep neural networks typically  require accurate and a large number of annotations to achieve outstanding performance in medical image segmentation. One-shot and weakly-supervised learning are promising research directions that reduce labeling effort by learning a new class from only one annotated image and using coarse labels instead, respectively. In this work, we present an innovative framework for 3D medical image segmentation with one-shot and weakly-supervised settings. Firstly a propagation-reconstruction network is proposed to propagate scribbles from one annotated volume to unlabeled 3D images based on the assumption that anatomical patterns in different human bodies are similar. Then a multi-level similarity denoising module is designed to refine the scribbles based on embeddings from anatomical- to pixel-level. After expanding the scribbles to pseudo masks, we observe the miss-classified voxels mainly occur at the border region and propose to extract self-support prototypes for the specific refinement. Based on these weakly-supervised segmentation results, we further train a segmentation model for the new class with the noisy label training strategy. Experiments on one abdomen and one head-and-neck CT dataset show the proposed method obtains significant improvement over the state-of-the-art methods and performs robustly even under severe class imbalance and low contrast. 
![image](https://github.com/LWHYC/OneShot_WeaklySeg/blob/main/train_frame.jpg)

## Requirements
Pytorch >= 1.4, SimpleITK >= 1.2, scipy >= 1.3.1, nibabel >= 2.5.0, GeodisTK and some common packages.

## Usages
### Dataset
You could download the processed dataset from: [StructSeg](https://structseg2019.grand-challenge.org/Home/) task1 (Organ-at-risk segmentation from head & neck CT scans): [BaiDu Yun](https://pan.baidu.com/s/1VV8VqJ39wKvlF-mh8b6IVg?pwd=ic6g) or [Google Drive](https://drive.google.com/file/d/1TlMfWvgSd3kAh3Eq80DVoboZ42FbLMvE/view?usp=sharing) into `data/` and unzip them. For TCIA-Pancreas, please cite the original paper (Deeporgan: Multi-level deep convolutional networks for automated pancreas segmentation).
#### Extend to Your Dataset
Prepare your data in `data/Your_Data_Name/`. The data format should be like:
```
    data/Your_Data_Name/
    ├── train
    │   ├── 1
    │     ├── rimage.nii.gz
    │     ├── rlabel.nii.gz            
    │   ├── 2
    │   ├── ...
    ├── valid
    │   ├── n
    │     ├── rimage.nii.gz
    │     ├── rlabel.nii.gz
    │   ├── ...
    └── test
        ├── N
          ├── rimage.nii.gz
          ├── rlabel.nii.gz
        ├── ...
```
Actually, you can customize the names of your images and labels. Just record their pathes in the corresponding txt files in `config/data/Your_Data_Name`. You can refer to the files in `config/data/TCIA/` as an example.
### Pretrained Model
The pretrained model for PRNet and Unet is avaliable [here](https://drive.google.com/drive/folders/1RGxeU_VRczBYuor-gKYQh2iQ50obRZKV?usp=sharing). Just place them in the `weights/`.
### Train PRNet
- To train the PRNet, run `bash train_prnet.sh`.
### Generate Coarse Segmentations
- Run `bash test_coarseg_seg.sh` for coarse segmentation.  
It will generate a coarse segmentation file named `coarseg.nii.gz` in each scan folder.
### Train PLC Segmentation Network
- Run `bash train_plc.sh`
### Generate Fine Segmentations
- Run `bash test_fine_seg.sh`.  
It will generate a coarse segmentation file named `fineseg.nii.gz` in each scan folder.
