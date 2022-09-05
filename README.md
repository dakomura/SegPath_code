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

### `3_CELL.mask_generation.py` 
This script generates the segmentation masks based on the patches from IF-restained sections and the Cellpose output. 

usage:
```
python 3_CELL.mask_generation.py input_dir 
```

### `3_RBC.mask_generation.py` 
This script generates the segmentation masks for red blood cells based on the patches from IF-restained sections. 

usage:
```
python 3_RBC.mask_generation.py input_WSI 
```
Input Variable | Description
--- | --- 
--msize_opal | minimum size of IF positive region
--th_opal | IF intensity cutoff


### `3_REGION.mask_generation.py ` 
This script generates the segmentation masks for tissues based on the patches from IF-restained sections. 

usage:
```
python 3_REGION.mask_generation.py input_dir 
```
Input Variable | Description
--- | --- 
--th_opal | IF intensity cutoff

### `4.QC_make_summary.py` 
This script calculates blur level and the correlation between DAPI and Hematoxylin signal.

usage:
```
python 4.QC_make_summary.py input_dir 
```

### `5.filter_QC.py` 
This script filters out patches based blur level and the correlation between DAPI and Hematoxylin signal.

usage:
```
python 5.filter_QC.py input_dir antibody 
```


### `6.train_segmentation_model.py` 
This script trains the segmentation models (requires MLFlow).
