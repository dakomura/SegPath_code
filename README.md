# SegPath generation

This repository provides scripts to generate annotation masks for tissue/cell segmentation using immunofluorescence restaining.

## Prerequisites

Python 3.7 or newer

- numpy
- matplotlib
- seaborn
- joblib
- pandas
- scipy
- openslide
- pillow
- tqdm
- cellpose
- mlflow
- opencv
- pytorch
- torchvision
- pytorch lightning
- torchmetrics
- segmentation_models_pytorch
- albumentations
- scikit-image
- kornia
- optuna
- dali
- SimpleITK
- imreg_dft

## scripts

### `1.registration_patch_extraction.py` 
This script extracts patches from Whole Slide Images (.ndpi) of tissue microarray sections after rigid and non-rigid registration between H&E-stained and immunofluorescence (IF)-restained sections.

usage:
```
python 1.registration_patch_extraction.py targetdir outdir [option] 
```

Input Variable | Description
--- | --- 
--init-scale | scale used for rough registration
--regist_scale | scale for fine-grained registration
--img_size | output image size
--mask_th | cutoff IHC intensity for mask generation (0-255)
--overwrite | overwrite output image files

`targetdir` must contain subdirectories, each of which have the following three .ndpi files.
1. HE-stained WSI file the file must contain either 'DAPI' or 'Opal' in its name.
1. DAPI-stained WSI file (the file must contain 'DAPI' in its name.)
1. IF-stained WSI file (the file must contain 'Opal' in its name.)

The slides must be scanned at 40x magnification.

The last directory name of targetdir is used in other scripts as `antibody`.

output:
- HE-stained patch file, which ends with `_HE.png`
- IF-stained patch file, which ends with `_IHC_nonrigid.png`

The HE and IF-image pair files have the same prefix in their name.

### `2_CELL.run_cellpose.py` 
This script runs Cellpose to the extrated patches (for cell segmentation).

usage:
```
python 2_CELL.run_cellpose.py input_dir [option] 
```

Input Variable | Description
--- | --- 
--pos_th | IF intensity cutoff for mask generation(0-255)
--diameter | expected nucleus diameter(px)
--bs | batch size for cellpose
--overlap | overlap rate for positive cell
--cpu | CPU mode
--reuse | reuse cellpose results
--skip | skip if the output file exists
--cellpose_th | Cell probability threshold

`input_dir` is the output directory created by `1.registration_patch_extraction.py` 

output:
- image of IF-stained regions detected by Cellpose, which ends with `_IHC_cellpose_nonrigid.png`
- cell mask, which ends with `_mask.pkl`

### `3_CELL.mask_generation.py` 
This script generates the segmentation masks based on the patches from IF-restained sections and the Cellpose output. 

usage:
```
python 3_CELL.mask_generation.py input_dir 
```

`input_dir` is the output directory containig files created by `2_CELL.run_cellpose.pyy` 

output:
- mask of IF-stained regions detected by Cellpose, which ends with `_IHC_cellpose_mask.png`


### `3_RBC.mask_generation.py` 
This script generates the segmentation masks for red blood cells based on the patches from IF-restained sections. 

usage:
```
python 3_RBC.mask_generation.py input_file [option] 
```
Input Variable | Description
--- | --- 
--msize_opal | minimum size of IF positive region
--th_opal | IF intensity cutoff

`input_file` is the output HE-stained patch file created by `1.registration_patch_extraction.py`

output:
- RBC mask file, which ends with `_IHC_nonrigid_mask2.png`

### `3_REGION.mask_generation.py `  
This script generates the segmentation masks for tissues based on the patches from IF-restained sections (requires MLFlow). 

usage:
```
python 3_REGION.mask_generation.py input_dir [option] 
```
Input Variable | Description
--- | --- 
--th_opal | IF intensity cutoff

Note: Please modify the source code so that the RBC segmentation model can be loaded from MLFlow server (l.22-25, l.100).

`input_dir` is the output directory created by `1.registration_patch_extraction.py` 

output:
- mask file, which ends with `_IHC_nonrigid_mask2.png`

### `4.QC_make_summary.py` 
This script calculates blur level and the correlation between DAPI and Hematoxylin signal.

usage:
```
python 4.QC_make_summary.py input_dir 
```

`input_dir` is the output directory created by `1.registration_patch_extraction.py` 

output:
- csv file which contains blur level and the correlation between DAPI and Hematoxylin signal for each patch file.

### `5.filter_QC.py` 
This script filters out patches based blur level and the correlation between DAPI and Hematoxylin signal.

usage:
```
python 5.filter_QC.py input_dir antibody 
```

`input_dir` is the output directory created by `1.registration_patch_extraction.py` 
`antibody` is used in the output csv.

output:
- csv file which contains blur level and the correlation between DAPI and Hematoxylin signal for each patch file.

### `6.train_segmentation_model.py` 
This script trains the segmentation models (requires MLFlow).
usage:
```
python 6.train_segmentation_model.py antibody [option]
```
Input Variable | Description
--- | --- 
--user | user name for MLFlow
--data_dir | input data directory
--resume | resume file for Optuna Study
--img_size | input image size
--post | postfix for MLflow name
--loss | loss type(combo/dice/bce/ftv/focal/auto)
--lparam1 | loss parameter 1
--lparam2 | loss parameter 2
--nepoch | number of epochs
--n_trials | number of optuna trials
--nbatch_tr | training batch size
--accum_grad | use accumulate gradient
--oversampling | oversampling for training data
--num_gpus | number of GPU used for training
--debug | debug mode (only 5% samples are used for train/val

Note: Please modify the source code so that the RBC segmentation model can be loaded from MLFlow server (l.192-195).

output:
- Segmentation model (stored in MLFlow).
