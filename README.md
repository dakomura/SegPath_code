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

### `2_CELL.run_cellpose.py` 
This script runs Cellpose to the extrated patches (for cell segmentation).

### `3_CELL.mask_generation.py` 
This script generates the segmentation masks based on the patches from IF-restained sections and the Cellpose output. 

### `3_RBC.mask_generation.py` 
This script generates the segmentation masks for red blood cells based on the patches from IF-restained sections. 

### `3_REGION.mask_generation.py ` 
This script generates the segmentation masks for tissues based on the patches from IF-restained sections. 

### `4.QC_make_summary.py` 
This script calculates blur level and the correlation between DAPI and Hematoxylin signal.

### `5.filter_QC.py` 
This script filters out patches based blur level and the correlation between DAPI and Hematoxylin signal.

### `6.train_segmentation_model.py` 
This script trains the segmentation models (requires MLFlow).
