Prepare your data here. The data format should be like:
```
    data/Your_Data_Name/
    ├── train
    │   ├── 1
    │     ├── rimage.nii.gz
    │     ├── rlabel.nii.gz            
    │   ├── 2
    │   ├── ...
    ├── valid
    │   ├── a
    │     ├── rimage.nii.gz
    │     ├── rlabel.nii.gz
    │   ├── ...
    └── test
        ├── b
          ├── rimage.nii.gz
          ├── rlabel.nii.gz
        ├── ...
```
Actually, you can customize the names of your images and labels. Just record their pathes in the corresponding txt files in `config/data/Your_Data_Name`. 
