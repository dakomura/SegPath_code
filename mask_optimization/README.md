# Mask optimization
This subdirectory provides scripts to mask optimization.

## scripts

### `1.apply_segmentation.py` 
This script runs the segmentation models trained on the current dataset and save the results (requires MLFlow).

usage:
```
python 1.apply_segmentation.py antibody [option] 
```

Input Variable | Description
--- | --- 
--batch | batch size
--rbc | RBC segmentation
--data_dir | input directory
--out_dir | output directory

Note: Please modify the source code so that the segmentation model can be loaded from MLFlow server (l.92-l.116).

output:
- segmentation mask predicted by the segmentation model, which ends with `.npy`

### `2.ridge_regression.py` 
This script applies ridge regression between the IF-mask intensity and the output probabiities.

usage:
```
python 2.ridge_regression.py antibody 
```

output:
- dataframe containing the number of pixels predicted as positive by the segmentation model and coefficient of ridge regression, which ends with `.npy`

### `3.IF_cutoff_optimization.py` 
This script optimizes IF intensity cufoff. 

usage:
```
python 3.IF_cutoff_optimization.py antibody 
```

output:
- dataframe containing the updated IF threshold, which ends with `_otsu.pkl`

### `4_CELL.MCC_calculation.py` 
This script calculates Matthew’s correlation coefficient (MCC) between the prediction and mask for leucocytes, myeloid cells, lymphocytes, plasma cells, and endothelial cells.

usage:
```
python 4_CELL.MCC_calculation.py antibody 
```

output:
- dictionary containing MCC, which ends with `.pkl`


### `5_CELL.overlap_ratio_optimization.py` 
This script optimizes the nucleus overlap cut-off based on the MCC.
usage:
```
python 5_CELL.overlap_ratio_optimization.py antibody 
```

output:
- updated mask files, which ends with `_IHC_cellpose_mask.png` or `_IHC_nonrigid_mask2.png`

